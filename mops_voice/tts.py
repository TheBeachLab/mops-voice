"""Text-to-speech via F5-TTS with MLX acceleration and voice cloning."""

import tempfile
import os
from pathlib import Path

import numpy as np
import soundfile as sf


class Synthesizer:
    """F5-TTS wrapper. Load once, synthesize many."""

    SAMPLE_RATE = 24000  # F5-TTS default output rate

    def __init__(self, ref_audio_path: Path, ref_text_path: Path):
        from f5_tts_mlx.generate import generate

        self._generate = generate
        self.ref_audio_path = str(ref_audio_path)

        if not Path(self.ref_audio_path).exists():
            raise FileNotFoundError(
                f"Reference audio not found: {ref_audio_path}\n"
                "Place a 5-15s WAV clip (24kHz mono 16-bit) of the target voice there."
            )

        if not ref_text_path.exists():
            raise FileNotFoundError(
                f"Reference transcript not found: {ref_text_path}\n"
                "Create it with the exact text spoken in the reference audio."
            )
        self.ref_text = ref_text_path.read_text().strip()

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize text to audio. Returns (audio_array, sample_rate).

        f5_tts_mlx.generate() writes to a file (side-effect only).
        We write to a temp file, then read it back as a numpy array.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            self._generate(
                generation_text=text,
                ref_audio_path=self.ref_audio_path,
                ref_audio_text=self.ref_text,
                output_path=tmp_path,
            )
            audio_data, sample_rate = sf.read(tmp_path, dtype="float32")
            return audio_data, sample_rate
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
