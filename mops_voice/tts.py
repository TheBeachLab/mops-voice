"""Text-to-speech via XTTS server (pre-trained TARS voice) or F5-TTS fallback."""

import io
import urllib.request
import json
from pathlib import Path

import numpy as np
import soundfile as sf

XTTS_URL = "http://127.0.0.1:8090"


class Synthesizer:
    """TTS wrapper. Tries XTTS server first, falls back to F5-TTS."""

    def __init__(self, ref_audio_path: Path = None, ref_text_path: Path = None):
        self._use_xtts = self._check_xtts()

        if self._use_xtts:
            return  # no local model needed

        # Fall back to F5-TTS with reference audio
        if ref_audio_path is None or ref_text_path is None:
            raise FileNotFoundError(
                "XTTS server not running and no reference audio provided.\n"
                "Either start the XTTS server: cd xtts_server && .venv/bin/python server.py\n"
                "Or provide reference audio for F5-TTS."
            )

        import mlx.core as mx

        self._mx = mx
        self._ref_text = ref_text_path.read_text().strip()

        audio, sr = sf.read(str(ref_audio_path))
        if sr != 24000:
            raise ValueError(f"Reference audio must be 24000Hz, got {sr}Hz")
        self._ref_audio = mx.array(audio)

        rms = mx.sqrt(mx.mean(mx.square(self._ref_audio)))
        if rms < 0.1:
            self._ref_audio = self._ref_audio * 0.1 / rms

        from f5_tts_mlx import F5TTS
        self._model = F5TTS.from_pretrained("lucasnewman/f5-tts-mlx")

    @staticmethod
    def _check_xtts() -> bool:
        """Check if XTTS server is running."""
        try:
            req = urllib.request.urlopen(f"{XTTS_URL}/health", timeout=2)
            return req.status == 200
        except Exception:
            return False

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize text to audio. Returns (audio_array, sample_rate)."""
        if self._use_xtts:
            return self._synthesize_xtts(text)
        return self._synthesize_f5(text)

    def _synthesize_xtts(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize via XTTS HTTP server."""
        payload = json.dumps({"text": text}).encode("utf-8")
        req = urllib.request.Request(
            f"{XTTS_URL}/synthesize",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        response = urllib.request.urlopen(req, timeout=60)
        wav_bytes = response.read()

        audio_data, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        return audio_data, sample_rate

    def _synthesize_f5(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize via local F5-TTS with reference audio."""
        mx = self._mx
        from f5_tts_mlx.utils import convert_char_to_pinyin

        generation_text = convert_char_to_pinyin([self._ref_text + " " + text])

        wave, _ = self._model.sample(
            mx.expand_dims(self._ref_audio, axis=0),
            text=generation_text,
            steps=8,
            method="rk4",
            speed=1.2,
            cfg_strength=2.0,
            sway_sampling_coef=-1.0,
        )

        wave = wave[self._ref_audio.shape[0]:]
        mx.eval(wave)

        return np.array(wave), 24000
