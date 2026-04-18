"""File-based debug logging for mops-voice and the mops MCP server.

Writes to `logs/session_<timestamp>.log` at the repo root (gitignored).
Nothing goes to the terminal — the existing Rich console output is untouched.

`mcp_stderr_file()` returns an append handle to the same session log, so the
MCP subprocess's stderr (the Node-side `[mops] ...` lines from server.js and
browser.js) is captured interleaved with the Python-formatted log lines.
Using a real file descriptor is required because the MCP SDK passes this
handle straight to `subprocess.stderr`.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TextIO

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)-5s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False
_log_file: Path | None = None


def setup_logging(level: int = logging.DEBUG) -> Path:
    """Configure the `mops_voice` logger tree to write to a session file.

    Safe to call repeatedly — subsequent calls return the existing log path.
    """
    global _configured, _log_file
    if _configured and _log_file is not None:
        return _log_file

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"session_{timestamp}.log"

    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    root = logging.getLogger("mops_voice")
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.propagate = False  # keep log output out of the terminal

    latest = LOG_DIR / "latest.log"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(log_file.name)
    except OSError:
        pass

    _log_file = log_file
    _configured = True

    root.info("=" * 72)
    root.info("mops-voice session start — %s", log_file)
    root.info("=" * 72)
    return log_file


def get_log_file() -> Path | None:
    return _log_file


def mcp_stderr_file() -> TextIO:
    """Open an append handle to the session log for the MCP subprocess's stderr.

    The returned file has a real `fileno()`, which is required by the Python
    MCP SDK (it passes this straight to `subprocess.stderr`). Caller is
    responsible for closing it during shutdown.
    """
    if _log_file is None:
        raise RuntimeError("setup_logging() must be called before mcp_stderr_file()")
    return open(_log_file, "a", buffering=1, encoding="utf-8", errors="replace")


def redact(value: str, keep: int = 4) -> str:
    """Mask a secret, keeping only the last `keep` characters."""
    if not value:
        return "<empty>"
    if len(value) <= keep:
        return "***"
    return "***" + value[-keep:]
