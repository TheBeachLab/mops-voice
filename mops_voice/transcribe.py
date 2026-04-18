"""Speech-to-text via whisper.cpp with CoreML acceleration."""

import logging
import time

from pywhispercpp.model import Model

log = logging.getLogger("mops_voice.transcribe")

HALLUCINATION_BLOCKLIST = [
    "thank you for watching",
    "thanks for watching",
    "subscribe to my channel",
    "please subscribe",
    "like and subscribe",
]


def is_gibberish(text: str) -> bool:
    """Return True if transcription should be skipped."""
    text = text.strip()
    if len(text) < 3:
        return True
    if text.lower() in HALLUCINATION_BLOCKLIST:
        return True
    return False


class Transcriber:
    """Whisper.cpp wrapper. Load once, transcribe many."""

    def __init__(self, model_name: str = "base.en"):
        log.info("loading whisper model: %s", model_name)
        self.model = Model(model_name)

    def transcribe(self, wav_bytes: bytes) -> str:
        """Transcribe WAV audio bytes to text."""
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name
        t0 = time.monotonic()
        try:
            segments = self.model.transcribe(tmp_path)
            text = " ".join(seg.text.strip() for seg in segments).strip()
            log.debug(
                "transcribed %d bytes → %r in %.2fs",
                len(wav_bytes), text, time.monotonic() - t0,
            )
            return text
        except Exception:
            log.exception("transcription failed after %.2fs", time.monotonic() - t0)
            raise
        finally:
            os.unlink(tmp_path)
