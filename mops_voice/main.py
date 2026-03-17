"""Main loop: push-to-talk -> transcribe -> Claude -> TTS -> playback."""

import argparse
import asyncio
import os
import threading
import time
from pathlib import Path

from rich.console import Console

from mops_voice.config import load_config, CONFIG_DIR
from mops_voice.audio import record_until_release, play_audio, audio_to_wav_bytes
from mops_voice.transcribe import Transcriber, is_gibberish
from mops_voice.tts import Synthesizer
from mops_voice.llm import MopsLLM

console = Console()

EXIT_PHRASES = {"goodbye", "exit", "quit", "bye"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MOPS voice assistant")
    parser.add_argument(
        "--mods-url", type=str, default=None, help="Mods CE URL (forwarded to MOPS server)"
    )
    parser.add_argument(
        "--headed", action="store_true", default=False, help="Show browser window (default: headless)"
    )
    parser.add_argument(
        "--whisper-model", type=str, default=None, help="Whisper model name"
    )
    return parser.parse_args(argv)


async def run(argv: list[str] | None = None):
    args = parse_args(argv)
    config = load_config()
    config_path = CONFIG_DIR / "config.json"

    # Override config from CLI args
    if args.whisper_model:
        config["whisper_model"] = args.whisper_model

    # --- Startup checks ---
    import sounddevice as sd

    # Check mic
    try:
        sd.query_devices(kind="input")
    except sd.PortAudioError:
        console.print("[red]No microphone detected. Cannot start.[/red]")
        return

    # Check claude CLI
    if not MopsLLM.check_cli():
        console.print("[red]Claude CLI not found.[/red]")
        console.print("Install it: https://docs.anthropic.com/en/docs/claude-code")
        return

    console.print("[bold cyan]MOPS Voice Assistant[/bold cyan]")
    console.print(f"Assistant: {config['assistant_name']}  |  User: {config['user_name']}")
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

    # TTS
    ref_audio = CONFIG_DIR / "tars_reference.wav"
    ref_text = CONFIG_DIR / "tars_reference.txt"
    synthesizer = None
    console.print("🔊 Loading TTS model...", end=" ")
    try:
        synthesizer = Synthesizer(ref_audio, ref_text)
        console.print("[green]OK[/green]")
    except FileNotFoundError as e:
        console.print(f"[yellow]WARN: {e}[/yellow]")
        console.print("[yellow]   TTS disabled -- text-only mode[/yellow]")
    except Exception as e:
        console.print(f"[yellow]WARN: TTS failed: {e}[/yellow]")
        console.print("[yellow]   TTS disabled -- text-only mode[/yellow]")

    # LLM + MCP
    llm = MopsLLM(config, config_path)

    # Configure MOPS MCP server for claude CLI
    mcp_args = []
    if args.headed:
        pass  # user wants to see the browser; don't pass --headless
    else:
        mcp_args.append("--headless")
    if args.mods_url:
        mcp_args.extend(["--mods-url", args.mods_url])

    server_path = config["mops_server_path"]
    # Resolve relative path from mops-voice repo root
    if not Path(server_path).is_absolute():
        server_path = str((Path(__file__).parent.parent / server_path).resolve())

    console.print("🤖 Configuring MOPS MCP server...", end=" ")
    if Path(server_path).exists():
        llm.setup_mcp(server_path, mcp_args)
        console.print("[green]OK[/green]")
    else:
        console.print(f"[yellow]WARN: Server not found at {server_path}[/yellow]")
        console.print("[yellow]   Conversation-only mode (no machine control)[/yellow]")

    console.print()
    console.print("[bold]Hold SPACEBAR to talk. Ctrl+C to exit.[/bold]")
    console.print()

    # --- Keyboard listener ---
    from pynput import keyboard
    import subprocess as _sp

    recording = False
    stop_event = threading.Event()
    space_held = False

    # Get our terminal's PID to check focus
    _our_pid = str(os.getpid())

    def _terminal_is_focused() -> bool:
        """Check if our terminal app is the frontmost window on macOS."""
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

    # --- Main loop ---
    loop = asyncio.get_running_loop()

    try:
        while True:
            # Wait for spacebar press
            while not recording:
                await asyncio.sleep(0.05)

            console.print("🎤 [bold red]Recording...[/bold red]", end=" ")

            # Record in thread
            audio_data = await loop.run_in_executor(
                None, record_until_release, stop_event
            )

            if audio_data is None or audio_data.size == 0:
                console.print("[yellow]No audio captured[/yellow]")
                continue

            duration = len(audio_data) / 16000
            console.print(f"[green]{duration:.1f}s[/green]")

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

            if is_gibberish(text):
                console.print("[yellow]Didn't catch that, Fran.[/yellow]")
                continue

            # Check for exit
            if text.strip().lower().rstrip(".!") in EXIT_PHRASES:
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

            # Claude API with tool loop
            t0 = time.monotonic()
            console.print(f"🤖 Calling Claude ({config['claude_model'].split('-')[1]})...")

            def on_tool_call(name, summary):
                console.print(f"  🔧 Tool call: {name} → {summary}")

            response_text = await llm.chat(text, on_tool_call=on_tool_call)
            console.print(f"🤖 Response: [green]{response_text}[/green]")

            # TTS
            if synthesizer:
                console.print("🔊 Synthesizing speech...", end=" ")
                try:
                    audio, sr = await loop.run_in_executor(
                        None, synthesizer.synthesize, response_text
                    )
                    audio_duration = len(audio) / sr
                    console.print(f"[green]{audio_duration:.1f}s[/green]")
                    console.print("🔊 Playing...", end=" ")
                    await loop.run_in_executor(None, play_audio, audio, sr)
                    console.print("[green]done[/green]")
                except Exception as e:
                    console.print(f"[red]TTS error: {e}[/red]")

            elapsed = time.monotonic() - t0
            console.print(f"⏱️  Total: {elapsed:.1f}s")
            console.print()

    except KeyboardInterrupt:
        console.print("\n[bold]Shutting down...[/bold]")
    finally:
        listener.stop()
        console.print("[bold cyan]MOPS out.[/bold cyan]")
