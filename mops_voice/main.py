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

import numpy as np

_SENTENCE_BOUNDARY = re.compile(r"[.!?](\s|$)")

from rich.console import Console

from mops_voice.config import load_config, CONFIG_DIR
from mops_voice.audio import record_until_release, play_audio, audio_to_wav_bytes, close_output_stream
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
        choices=["cli", "api", "openai"],
        help="LLM engine: cli (Claude Code CLI), api (direct Anthropic API), or openai (direct OpenAI API)",
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

    # Override config from CLI args. Persistent flags (engine choices,
    # whisper model, user name) also get saved so the next run defaults
    # to whatever you last used — no need to retype --llm-engine api
    # every session. Session-only flags (--mods-url, --headless) are
    # deliberately not persisted.
    persist = False
    if args.whisper_model:
        config["whisper_model"] = args.whisper_model; persist = True
    if args.tts_engine:
        config["tts_engine"] = args.tts_engine; persist = True
    if args.llm_engine:
        config["llm_engine"] = args.llm_engine; persist = True
    if args.user:
        config["user_name"] = args.user; persist = True
    if persist:
        from mops_voice.config import save_config
        save_config(config_path, config)
        log.info("saved CLI overrides to %s", config_path)

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
    if llm_engine == "openai":
        llm_label = "OpenAI API"
        model_display = config.get("openai", {}).get("model", "gpt-5-mini")
    elif llm_engine == "api":
        llm_label = "Anthropic API"
        model_display = config["claude_model"]
    else:
        llm_label = "Claude CLI"
        model_display = config["claude_model"]
    console.print(f"LLM: {model_display} via {llm_label}")
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
        f"Greet them with a HIGH-ENERGY, excited one-liner — think "
        f"punchy, exclamatory, hyped-up. MUST open with a salutation "
        f"addressing them by name (e.g. 'Hey {config['user_name']}!', "
        f"'Hello {config['user_name']}!', 'Welcome back, {config['user_name']}!'). "
        f"Lead with the energy, not the sarcasm. One sentence, under "
        f"15 words, ending with an exclamation mark. Reference "
        f"fabrication machines, past disasters, or tease them about "
        f"what they might be up to today.)"
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

    def _resolve_key(name: str):
        """Map a config key name to a pynput Key/KeyCode.

        Single-char names (e.g. "b") become a `KeyCode`; multi-char names
        (e.g. "page_down", "f5", "esc") resolve via `keyboard.Key.<name>`.
        Returns None if the name doesn't match either form — caller should
        treat that as "feature disabled" rather than crashing.
        """
        if not name:
            return None
        name = name.lower().strip()
        if len(name) == 1:
            return keyboard.KeyCode.from_char(name)
        return getattr(keyboard.Key, name, None)

    clicker_cfg = config.get("clicker") or {}
    clicker_enabled = bool(clicker_cfg.get("enabled"))
    clicker_trigger = _resolve_key(clicker_cfg.get("trigger_key", "")) if clicker_enabled else None
    clicker_cancel = _resolve_key(clicker_cfg.get("cancel_key", "")) if clicker_enabled else None
    clicker_mode = clicker_cfg.get("mode", "toggle")

    if clicker_enabled:
        log.info(
            "clicker enabled: trigger=%s cancel=%s mode=%s",
            clicker_cfg.get("trigger_key"), clicker_cfg.get("cancel_key"), clicker_mode,
        )
        controls = [
            f"{clicker_cfg.get('trigger_key', '?').upper().replace('_', ' ')} = "
            f"{'toggle' if clicker_mode == 'toggle' else 'hold to talk'}",
        ]
        if clicker_cancel is not None:
            controls.append(f"{clicker_cfg.get('cancel_key', '?').upper()} = cancel")
        console.print(f"📡 [bold]Clicker:[/bold] " + "  |  ".join(controls))

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
        # Clicker keys: NO focus check — designed for when you're across
        # the room. Trade-off: a stray Page Down (or whatever the trigger
        # is) anywhere on the system will toggle a recording.
        if clicker_enabled:
            if clicker_trigger is not None and key == clicker_trigger:
                if clicker_mode == "toggle":
                    if not recording:
                        recording = True
                        stop_event.clear()
                    else:
                        recording = False
                        stop_event.set()
                else:  # "hold" — start on press, on_release will stop
                    if not recording:
                        recording = True
                        stop_event.clear()
                return
            if clicker_cancel is not None and key == clicker_cancel:
                cancel_event.set()
                return
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
        # Clicker hold-mode release stops the recording. Toggle mode
        # ignores releases (the second click handles stop in on_press).
        if (
            clicker_enabled
            and clicker_mode == "hold"
            and clicker_trigger is not None
            and key == clicker_trigger
        ):
            if recording:
                recording = False
                stop_event.set()
            return
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
            if llm_engine == "openai":
                _model_label = config.get("openai", {}).get("model", "gpt-5-mini")
                console.print(f"🤖 Calling OpenAI ({_model_label})...")
            else:
                console.print(f"🤖 Calling Claude ({config['claude_model']})...")

            # Pipelined speech: a synth thread pulls sentences and produces
            # audio, a play thread pulls audio and plays it. With sequential
            # synth+play in one thread, sentence N+1's synth was blocked
            # during sentence N's playback — leaving a ~1.5s silent gap per
            # sentence on Voxtral. Splitting them lets synth(N+1) overlap
            # with play(N), shrinking inter-sentence gaps to ~the difference
            # between synth latency and playback duration.
            sentence_queue: queue.Queue[str | None] = queue.Queue()
            audio_queue: queue.Queue[tuple[np.ndarray, int] | None] = queue.Queue()
            speech_done = threading.Event()

            def _synth_worker():
                """Pull sentences, synthesize, push audio chunks downstream."""
                while True:
                    phrase = sentence_queue.get()
                    try:
                        if phrase is None:  # poison pill
                            audio_queue.put(None)
                            break
                        if cancel_event.is_set():
                            continue
                        try:
                            audio, sr = synthesizer.synthesize(phrase)
                            if not cancel_event.is_set():
                                audio_queue.put((audio, sr))
                        except Exception:
                            pass
                    finally:
                        sentence_queue.task_done()

            def _play_worker():
                """Pull audio chunks and play them, one after the next."""
                while True:
                    item = audio_queue.get()
                    try:
                        if item is None:  # poison pill from synth
                            break
                        if cancel_event.is_set():
                            continue
                        audio, sr = item
                        try:
                            play_audio(audio, sr)
                        except Exception:
                            pass
                    finally:
                        audio_queue.task_done()
                speech_done.set()

            async def wait_for_speech(timeout: float = 30.0) -> bool:
                """Block until both pipelines have fully drained.

                Used as the voice-action pacing gate: _chat_api awaits this
                after a response with tool_use blocks and before executing
                those tools, so the user always hears the narration that
                describes an action *before* the action fires. With the
                synth/play split we must wait on BOTH stages — sentence
                still in the synth queue, or audio still queued/playing.
                """
                nonlocal sentence_buffer
                if not synthesizer:
                    return True
                # Flush any partial sentence the streamer didn't enqueue
                # because it never saw a terminal `.!?`. Without this,
                # narration like "opening browser" (no period) sits in the
                # buffer while tools fire — voice-leads-action breaks and
                # the user hears the line merged with the next iteration's.
                tail = sentence_buffer.strip()
                if tail:
                    sentence_buffer = ""
                    sentence_queue.put(tail)
                start = time.monotonic()
                while time.monotonic() - start < timeout:
                    if cancel_event.is_set():
                        return False
                    if (
                        sentence_queue.unfinished_tasks == 0
                        and audio_queue.unfinished_tasks == 0
                    ):
                        return True
                    await asyncio.sleep(0.05)
                return False

            sentence_buffer = ""
            first_text_chunk_seen = False
            first_text_t: float | None = None

            if synthesizer:
                synth_thread = threading.Thread(target=_synth_worker, daemon=True)
                play_thread = threading.Thread(target=_play_worker, daemon=True)
                synth_thread.start()
                play_thread.start()

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
                        sentence_queue.put(sentence)

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
                # Push any trailing sentence that didn't end in punctuation,
                # then a poison pill. The synth worker forwards the pill to
                # audio_queue once it's drained, so play_worker exits cleanly
                # only after every queued sentence has been synth'd + played.
                tail = sentence_buffer.strip()
                if tail:
                    sentence_queue.put(tail)
                sentence_queue.put(None)  # poison pill

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
        close_output_stream()
        await llm.disconnect_mcp()
        log.info("session end — log at %s", log_file)
        console.print("[bold cyan]MOPS out.[/bold cyan]")


def main():
    """Console-script entry point. Wraps the async run() in asyncio.run."""
    asyncio.run(run())
