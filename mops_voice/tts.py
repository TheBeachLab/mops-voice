"""Text-to-speech with pluggable engines: F5-TTS (local) or Voxtral (API)."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
import urllib.request
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24000
TARGET_RMS = 0.1

log = logging.getLogger("mops_voice.tts")


class F5Synthesizer:
    """F5-TTS via MLX — local voice cloning from a reference clip."""

    engine = "f5"

    def __init__(self, ref_audio_path: Path, ref_text_path: Path):
        import mlx.core as mx
        import soundfile as sf
        from f5_tts_mlx import F5TTS

        self._mx = mx

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

        audio, sr = sf.read(str(ref_audio_path))
        if sr != SAMPLE_RATE:
            raise ValueError(f"Reference audio must be {SAMPLE_RATE}Hz, got {sr}Hz")
        self._ref_audio = mx.array(audio)

        rms = mx.sqrt(mx.mean(mx.square(self._ref_audio)))
        if rms < TARGET_RMS:
            self._ref_audio = self._ref_audio * TARGET_RMS / rms

        self._model = F5TTS.from_pretrained("lucasnewman/f5-tts-mlx")

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize text to audio. Returns (audio_array, sample_rate)."""
        mx = self._mx
        from f5_tts_mlx.utils import convert_char_to_pinyin

        log.debug("f5 synth → %d chars: %r", len(text), text[:120])
        t0 = time.monotonic()
        generation_text = convert_char_to_pinyin([self.ref_text + " " + text])

        try:
            wave, _ = self._model.sample(
                mx.expand_dims(self._ref_audio, axis=0),
                text=generation_text,
                steps=8,
                method="rk4",
                speed=1.2,
                cfg_strength=2.0,
                sway_sampling_coef=-1.0,
            )
        except Exception:
            log.exception("f5 synth failed after %.2fs", time.monotonic() - t0)
            raise

        wave = wave[self._ref_audio.shape[0]:]
        mx.eval(wave)  # materialize the MLX lazy array

        audio = np.array(wave)
        log.info(
            "f5 synth done %.2fs → %.2fs audio",
            time.monotonic() - t0, len(audio) / SAMPLE_RATE,
        )
        return audio, SAMPLE_RATE


class VoxtralSynthesizer:
    """Voxtral TTS via Mistral API — cloud-based, no local model needed."""

    engine = "voxtral"

    # Preset voices (slug -> display name)
    VOICES = {
        "en_paul_neutral": "Paul - Neutral",
        "en_paul_confident": "Paul - Confident",
        "en_paul_happy": "Paul - Happy",
        "en_paul_excited": "Paul - Excited",
        "en_paul_cheerful": "Paul - Cheerful",
        "en_paul_angry": "Paul - Angry",
        "en_paul_frustrated": "Paul - Frustrated",
        "en_paul_sad": "Paul - Sad",
        "gb_oliver_neutral": "Oliver - Neutral",
        "gb_jane_sarcasm": "Jane - Sarcasm",
    }

    def __init__(self, api_key: str, voice: str = "en_paul_confident",
                 ref_audio_path: Path | None = None):
        import soundfile  # noqa: F401 — verify available at init time

        if not api_key:
            raise ValueError(
                "Voxtral API key required. Set it in ~/.mops-voice/config.json "
                'under voxtral.api_key, or export VOXTRAL_API_KEY.'
            )
        self._api_key = api_key
        self._voice = voice
        self._url = "https://api.mistral.ai/v1/audio/speech"

        # Load reference audio for voice cloning (if voice is "tars" or a .wav path)
        self._ref_audio_b64 = None
        if ref_audio_path and ref_audio_path.exists():
            self._ref_audio_b64 = base64.b64encode(
                ref_audio_path.read_bytes()
            ).decode()

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize text via Voxtral API. Returns (audio_array, sample_rate)."""
        import soundfile as sf

        body: dict = {
            "model": "voxtral-mini-tts-2603",
            "input": text,
            "response_format": "wav",
        }

        if self._ref_audio_b64:
            body["ref_audio"] = self._ref_audio_b64
        else:
            body["voice_id"] = self._voice

        payload = json.dumps(body).encode()

        req = urllib.request.Request(
            self._url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

        log.debug(
            "voxtral synth → voice=%s chars=%d ref_clone=%s",
            self._voice, len(text), bool(self._ref_audio_b64),
        )
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read())
        except Exception:
            log.exception("voxtral synth failed after %.2fs", time.monotonic() - t0)
            raise

        audio_bytes = base64.b64decode(body["audio_data"])
        audio, sr = sf.read(io.BytesIO(audio_bytes))
        audio = audio.astype(np.float32)
        log.info(
            "voxtral synth done %.2fs → %.2fs audio @ %dHz",
            time.monotonic() - t0, len(audio) / sr, sr,
        )
        return audio, sr


def create_synthesizer(config: dict) -> F5Synthesizer | VoxtralSynthesizer:
    """Factory: build the right synthesizer from config."""
    engine = config.get("tts_engine", "f5")

    if engine == "voxtral":
        from mops_voice.config import CONFIG_DIR

        voxtral_cfg = config.get("voxtral", {})
        api_key = voxtral_cfg.get("api_key") or os.environ.get("VOXTRAL_API_KEY", "")
        voice = voxtral_cfg.get("voice", "en_paul_confident")

        # "tars" voice uses inline cloning from reference audio
        ref_audio_path = None
        if voice == "tars":
            ref_audio_path = CONFIG_DIR / "tars_reference.wav"
            if not ref_audio_path.exists():
                raise FileNotFoundError(
                    f"TARS reference audio not found: {ref_audio_path}"
                )

        return VoxtralSynthesizer(
            api_key=api_key, voice=voice, ref_audio_path=ref_audio_path,
        )

    # Default: F5-TTS
    from mops_voice.config import CONFIG_DIR

    ref_audio = CONFIG_DIR / "tars_reference.wav"
    ref_text = CONFIG_DIR / "tars_reference.txt"
    return F5Synthesizer(ref_audio, ref_text)
