"""Main loop: push-to-talk -> transcribe -> Claude -> TTS -> playback."""

import argparse
import asyncio
import os
import queue
import re
import shlex
import sys
import threading
import time
from pathlib import Path

_SENTENCE_BOUNDARY = re.compile(r"[.!?](\s|$)")

from rich.console import Console

from mops_voice.config import load_config, CONFIG_DIR
from mops_voice.audio import record_until_release, play_audio, audio_to_wav_bytes
from mops_voice.transcribe import Transcriber, is_gibberish
from mops_voice.tts import create_synthesizer
from mops_voice.llm import MopsLLM
from mops_voice.logging_setup import setup_logging, redact

import logging

console = Console()
log = logging.getLogger("mops_voice.main")

EXIT_PHRASES = {"goodbye", "exit", "quit", "bye", "quit mops", "goodbye mops"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MOPS voice assistant")
    parser.add_argument(
        "--mods-url", type=str, default=None, help="Mods CE URL (forwarded to MOPS server)"
    )
    parser.add_argument(
        "--headless", action="store_true", default=False, help="Hide browser window (default: visible)"
    )
    parser.add_argument(
        "--whisper-model", type=str, default=None, help="Whisper model name"
    )
    parser.add_argument(
        "--tts-engine", type=str, default=None,
        choices=["f5", "voxtral"],
        help="TTS engine: f5 (local MLX) or voxtral (Mistral API)",
    )
    parser.add_argument(
        "--llm-engine", type=str, default=None,
        choices=["cli", "api"],
        help="LLM engine: cli (Claude Code CLI) or api (direct Anthropic API)",
    )
    parser.add_argument(
        "--user", type=str, default=None,
        help="User name (e.g. --user Fran)",
    )
    return parser.parse_args(argv)


def _resolve_mcp_command(config: dict) -> tuple[str | None, list[str]]:
    """Decide how to launch the mops MCP server.

    Prefers `mops_server_command` (a shell-style command string, e.g.
    "npx -y @thebeachlab/mops"). Falls back to legacy `mops_server_path`
    (a path to a JS file, run with `node`).
    Returns (command, args) or (None, []) if neither is set / valid.
    """
    cmd_str = config.get("mops_server_command")
    if cmd_str:
        parts = shlex.split(cmd_str)
        if parts:
            return parts[0], parts[1:]
    server_path = config.get("mops_server_path")
    if server_path:
        if not Path(server_path).is_absolute():
            server_path = str((Path(__file__).parent.parent / server_path).resolve())
        if Path(server_path).exists():
            return "node", [server_path]
    return None, []


async def run(argv: list[str] | None = None):
    log_file = setup_logging()
    args = parse_args(argv)
    config = load_config()
    config_path = CONFIG_DIR / "config.json"

    # Override config from CLI args
    if args.whisper_model:
        config["whisper_model"] = args.whisper_model
    if args.tts_engine:
        config["tts_engine"] = args.tts_engine
    if args.llm_engine:
        config["llm_engine"] = args.llm_engine
    if args.user:
        config["user_name"] = args.user

    log.info(
        "startup: engine=%s model=%s tts=%s whisper=%s user=%s headless=%s mods_url=%s",
        config.get("llm_engine"),
        config.get("claude_model"),
        config.get("tts_engine"),
        config.get("whisper_model"),
        config.get("user_name"),
        bool(args.headless),
        args.mods_url or "<default>",
    )
    log.debug(
        "secrets: anthropic=%s voxtral=%s",
        redact(config.get("anthropic", {}).get("api_key", "")),
        redact(config.get("voxtral", {}).get("api_key", "")),
    )

    # --- Startup checks ---
    import sounddevice as sd

    # Check mic
    try:
        sd.query_devices(kind="input")
    except sd.PortAudioError:
        console.print("[red]No microphone detected. Cannot start.[/red]")
        return

    # Check claude CLI (only needed for CLI engine)
    llm_engine = config.get("llm_engine", "cli")
    if llm_engine == "cli" and not MopsLLM.check_cli():
        console.print("[red]Claude CLI not found.[/red]")
        console.print("Install it: https://docs.anthropic.com/en/docs/claude-code")
        return

    console.print("[bold cyan]MOPS Voice Assistant[/bold cyan]")
    console.print(f"Assistant: {config['assistant_name']}  |  User: {config['user_name']}")
    llm_label = "Anthropic API" if llm_engine == "api" else "Claude CLI"
    console.print(f"LLM: {config['claude_model']} via {llm_label}")
    p = config["personality"]
    console.print(
        f"Personality: humor={p['humor']}% sarcasm={p['sarcasm']}% honesty={p['honesty']}%"
    )
    console.print()

    # --- Initialize components ---

    # Transcriber
    console.print("📝 Loading whisper model...", end=" ")
    try:
        transcriber = Transcriber(config["whisper_model"])
        console.print("[green]OK[/green]")
    except Exception as e:
        console.print(f"[red]FAILED: {e}[/red]")
        return

    # TTS engine
    synthesizer = None
    engine_name = config.get("tts_engine", "f5")
    console.print(f"🔊 Loading TTS ({engine_name})...", end=" ")
    try:
        synthesizer = create_synthesizer(config)
        engine_labels = {
            "f5": "F5-TTS local, MLX",
            "voxtral": "Voxtral, Mistral API",
        }
        console.print(f"[green]OK ({engine_labels.get(engine_name, engine_name)})[/green]")
    except FileNotFoundError as e:
        console.print(f"[yellow]WARN: {e}[/yellow]")
        console.print("[yellow]   TTS disabled — text-only mode[/yellow]")
    except Exception as e:
        console.print(f"[yellow]WARN: TTS failed: {e}[/yellow]")
        console.print("[yellow]   TTS disabled — text-only mode[/yellow]")

    # LLM + persistent MCP
    llm = MopsLLM(config, config_path)

    mcp_args = []
    if args.headless:
        mcp_args.append("--headless")
    if args.mods_url:
        mcp_args.extend(["--mods-url", args.mods_url])

    mcp_command, mcp_base_args = _resolve_mcp_command(config)

    console.print("🤖 Connecting to MOPS MCP server...", end=" ")
    if mcp_command is None:
        console.print("[yellow]WARN: no mops_server_command or mops_server_path configured[/yellow]")
        console.print("[yellow]   Conversation-only mode (no machine control)[/yellow]")
    else:
        mcp_ok = await llm.connect_mcp(mcp_command, mcp_base_args + mcp_args)
        if mcp_ok:
            console.print(f"[green]OK ({len(llm._mcp_tools)} tools)[/green]")
        else:
            console.print("[yellow]WARN: Could not connect[/yellow]")
            console.print("[yellow]   Conversation-only mode (no machine control)[/yellow]")

    console.print()
    console.print("[bold]SPACEBAR = talk  |  ESC = cancel  |  Q = quit[/bold]")
    console.print()

    # --- Greeting ---
    loop = asyncio.get_running_loop()
    greeting = await llm.chat(
        f"(System: {config['user_name']} just started a session. "
        f"Greet them with a short, personality-driven one-liner. "
        f"Be creative — reference fabrication machines, past disasters, "
        f"or tease them about what they might be up to today.)"
    )
    console.print(f"🤖 [green]{greeting}[/green]")
    if synthesizer:
        try:
            audio, sr = await loop.run_in_executor(
                None, synthesizer.synthesize, greeting
            )
            await loop.run_in_executor(None, play_audio, audio, sr)
        except Exception:
            pass
    console.print()

    # --- Keyboard listener ---
    from pynput import keyboard
    import subprocess as _sp

    recording = False
    stop_event = threading.Event()
    quit_event = threading.Event()
    cancel_event = threading.Event()
    space_held = False

    # Get our terminal's PID to check focus
    _our_pid = str(os.getpid())

    def _terminal_is_focused() -> bool:
        """Check if our terminal app is the frontmost window. macOS-only;
        elsewhere assume yes (the global keyboard listener still fires)."""
        if sys.platform != "darwin":
            return True
        try:
            result = _sp.run(
                ["osascript", "-e", 'tell application "System Events" to get name of first process whose frontmost is true'],
                capture_output=True, text=True, timeout=1
            )
            app = result.stdout.strip().lower()
            return app in ("terminal", "iterm2", "iterm", "warp", "alacritty", "kitty", "ghostty")
        except Exception:
            return True  # if check fails, allow recording

    def on_press(key):
        nonlocal recording, space_held
        # 'q' to quit cleanly
        if hasattr(key, 'char') and key.char == 'q' and not recording:
            if _terminal_is_focused():
                quit_event.set()
                return
        # ESC to cancel current operation
        if key == keyboard.Key.esc:
            if _terminal_is_focused():
                cancel_event.set()
                return
        if key == keyboard.Key.space and not space_held:
            if not _terminal_is_focused():
                return
            space_held = True
            if not recording:
                recording = True
                stop_event.clear()

    def on_release(key):
        nonlocal recording, space_held
        if key == keyboard.Key.space:
            space_held = False
            if recording:
                recording = False
                stop_event.set()

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()

    # Suppress spacebar and 'q' from reaching terminal: cbreak for
    # immediate delivery, then clear ECHO so the captured keys don't
    # print. Without ECHO-off, held spacebars accumulate visible whitespace
    # and the quit 'q' appears after exit.
    import tty
    import termios
    _old_term = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    _attrs = termios.tcgetattr(sys.stdin)
    _attrs[3] &= ~(termios.ECHO | termios.ECHONL)  # lflag: kill echo
    termios.tcsetattr(sys.stdin, termios.TCSANOW, _attrs)

    # --- Main loop ---
    try:
        while True:
            # Wait for spacebar press or 'q' to quit
            while not recording:
                if quit_event.is_set():
                    break
                await asyncio.sleep(0.05)
            if quit_event.is_set():
                break

            console.print("🎤 [bold red]Recording...[/bold red]", end=" ")
            log.info("--- new turn ---")

            # Record in thread
            audio_data = await loop.run_in_executor(
                None, record_until_release, stop_event
            )

            if audio_data is None or audio_data.size == 0:
                console.print("[yellow]No audio captured[/yellow]")
                log.info("turn skipped: no audio captured")
                continue

            duration = len(audio_data) / 16000
            console.print(f"[green]{duration:.1f}s[/green]")
            log.info("recorded %.2fs audio", duration)

            # Transcribe
            wav_bytes = audio_to_wav_bytes(audio_data)
            if wav_bytes is None:
                console.print("[yellow]No audio captured[/yellow]")
                continue

            console.print("📝 Transcribing...", end=" ")
            text = await loop.run_in_executor(
                None, transcriber.transcribe, wav_bytes
            )
            console.print(f'[cyan]"{text}"[/cyan]')
            log.info("transcribed: %r", text)

            if is_gibberish(text):
                console.print("[yellow]Didn't catch that, Fran.[/yellow]")
                log.info("rejected as gibberish")
                continue

            # Check for exit
            clean = text.strip().lower().rstrip(".!,")
            if clean in EXIT_PHRASES or any(p in clean for p in EXIT_PHRASES):
                console.print("", end="")
                farewell = await llm.chat("Fran is leaving. Say a brief goodbye.")
                console.print(f"[green]{farewell}[/green]")
                if synthesizer:
                    try:
                        audio, sr = await loop.run_in_executor(
                            None, synthesizer.synthesize, farewell
                        )
                        await loop.run_in_executor(None, play_audio, audio, sr)
                    except Exception:
                        pass
                break

            # Claude + tools
            cancel_event.clear()
            t0 = time.monotonic()
            console.print(f"🤖 Calling Claude ({config['claude_model']})...")

            # Background speech queue for tool progress.
            # task_done() is called on every popped item so `unfinished_tasks`
            # reflects both "queued" and "currently playing". That's what the
            # pacing gate (wait_for_speech) polls to sync voice and actions.
            speech_queue: queue.Queue[str | None] = queue.Queue()
            speech_done = threading.Event()

            def _speech_worker():
                """Drain the speech queue, synthesizing and playing each phrase."""
                while True:
                    phrase = speech_queue.get()
                    try:
                        if phrase is None:  # poison pill
                            break
                        if cancel_event.is_set():
                            continue
                        try:
                            audio, sr = synthesizer.synthesize(phrase)
                            if not cancel_event.is_set():
                                play_audio(audio, sr)
                        except Exception:
                            pass
                    finally:
                        speech_queue.task_done()
                speech_done.set()

            async def wait_for_speech(timeout: float = 30.0) -> bool:
                """Block until the speech queue has fully drained (synth + playback).

                Used as the voice-action pacing gate: _chat_api awaits this
                after a response with tool_use blocks and before executing
                those tools, so the user always hears the narration that
                describes an action *before* the action fires.
                """
                if not synthesizer:
                    return True
                start = time.monotonic()
                while time.monotonic() - start < timeout:
                    if cancel_event.is_set():
                        return False
                    if speech_queue.unfinished_tasks == 0:
                        return True
                    await asyncio.sleep(0.05)
                return False

            sentence_buffer = ""
            first_text_chunk_seen = False
            first_text_t: float | None = None

            if synthesizer:
                speech_thread = threading.Thread(target=_speech_worker, daemon=True)
                speech_thread.start()

            def on_tool_call(name, summary):
                console.print(f"  🔧 Tool call: {name} → {summary}")

            def on_text_chunk(delta: str):
                nonlocal sentence_buffer, first_text_chunk_seen, first_text_t
                if not first_text_chunk_seen:
                    first_text_chunk_seen = True
                    first_text_t = time.monotonic()
                    log.info("first text delta at %.2fs", first_text_t - t0)
                if cancel_event.is_set() or not synthesizer:
                    return
                sentence_buffer += delta
                while True:
                    m = _SENTENCE_BOUNDARY.search(sentence_buffer)
                    if not m:
                        break
                    end = m.end()
                    sentence = sentence_buffer[:end].strip()
                    sentence_buffer = sentence_buffer[end:]
                    if sentence:
                        log.debug("enqueue sentence: %r", sentence)
                        speech_queue.put(sentence)

            try:
                response_text = await llm.chat(
                    text,
                    on_tool_call=on_tool_call,
                    on_text_chunk=on_text_chunk,
                    wait_for_speech=wait_for_speech,
                )
            except Exception:
                log.exception("llm.chat failed")
                raise

            if synthesizer:
                # Push any trailing sentence that didn't end in punctuation.
                tail = sentence_buffer.strip()
                if tail:
                    speech_queue.put(tail)
                speech_queue.put(None)  # poison pill

            if cancel_event.is_set():
                console.print("[yellow]Cancelled.[/yellow]")
                cancel_event.clear()
                continue

            console.print(f"🤖 Response: [green]{response_text}[/green]")

            # Wait for the speech worker to drain the queue (tool phrases +
            # streamed sentences). Generation and synthesis already overlapped
            # while the tokens were arriving — this just blocks for playback.
            if synthesizer:
                speech_done.wait(timeout=60)

            elapsed = time.monotonic() - t0
            console.print(f"⏱️  Total: {elapsed:.1f}s")
            console.print()
            log.info("turn complete: %.2fs total, response=%r", elapsed, response_text)

    except KeyboardInterrupt:
        console.print("\n[bold]Shutting down...[/bold]")
        log.info("shutting down via KeyboardInterrupt")
    finally:
        # TCSAFLUSH (not TCSADRAIN) discards any stdin keystrokes that
        # accumulated during the session — otherwise the final 'q' or
        # queued spaces would dump into the shell prompt after exit.
        termios.tcsetattr(sys.stdin, termios.TCSAFLUSH, _old_term)
        listener.stop()
        await llm.disconnect_mcp()
        log.info("session end — log at %s", log_file)
        console.print("[bold cyan]MOPS out.[/bold cyan]")


def main():
    """Console-script entry point. Wraps the async run() in asyncio.run."""
    asyncio.run(run())
