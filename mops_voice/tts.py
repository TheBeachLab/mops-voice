"""Text-to-speech via F5-TTS with MLX acceleration and voice cloning."""

from pathlib import Path

import numpy as np

SAMPLE_RATE = 24000
TARGET_RMS = 0.1


class Synthesizer:
    """F5-TTS wrapper. Pre-loads model once, synthesizes many."""

    def __init__(self, ref_audio_path: Path, ref_text_path: Path):
        import mlx.core as mx
        import soundfile as sf
        from f5_tts_mlx import F5TTS

        self._mx = mx
        self._sf = sf

        if not ref_audio_path.exists():
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

        # Load reference audio
        audio, sr = sf.read(str(ref_audio_path))
        if sr != SAMPLE_RATE:
            raise ValueError(f"Reference audio must be {SAMPLE_RATE}Hz, got {sr}Hz")
        self._ref_audio = mx.array(audio)

        # Normalize RMS
        rms = mx.sqrt(mx.mean(mx.square(self._ref_audio)))
        if rms < TARGET_RMS:
            self._ref_audio = self._ref_audio * TARGET_RMS / rms

        # Pre-load the model (downloads once, then cached)
        self._model = F5TTS.from_pretrained("lucasnewman/f5-tts-mlx")

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize text to audio. Returns (audio_array, sample_rate)."""
        mx = self._mx
        from f5_tts_mlx.utils import convert_char_to_pinyin

        generation_text = convert_char_to_pinyin([self.ref_text + " " + text])

        wave, _ = self._model.sample(
            mx.expand_dims(self._ref_audio, axis=0),
            text=generation_text,
            steps=8,
            method="rk4",
            speed=1.2,
            cfg_strength=2.0,
            sway_sampling_coef=-1.0,
        )

        # Trim the reference audio portion from output
        wave = wave[self._ref_audio.shape[0]:]
        # Force MLX computation to complete (mx.eval is MLX's
        # graph evaluation, not Python's eval builtin)
        mx.eval(wave)

        return np.array(wave), SAMPLE_RATE
