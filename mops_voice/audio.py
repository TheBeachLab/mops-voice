"""Audio recording and playback via sounddevice."""

import io
import wave

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"


def audio_to_wav_bytes(audio_data: np.ndarray) -> bytes | None:
    """Convert numpy audio array to WAV bytes. Returns None if empty."""
    if audio_data.size == 0:
        return None
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())
    return buf.getvalue()


def record_until_release(stop_event) -> np.ndarray | None:
    """Record audio until stop_event is set. Returns numpy array or None."""
    frames = []

    def callback(indata, frame_count, time_info, status):
        if status:
            pass  # ignore underflow warnings during recording
        frames.append(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=callback,
    ):
        stop_event.wait()

    if not frames:
        return None
    return np.concatenate(frames, axis=0)


def play_audio(audio_data: np.ndarray, sample_rate: int = 24000) -> None:
    """Play audio through speakers. Blocks until complete.

    Applies a short linear fade-out so the waveform doesn't end on a
    non-zero sample — otherwise the speaker snaps back to zero and you
    hear a click/pop at the end of every utterance.
    """
    fade_samples = min(int(sample_rate * 0.015), len(audio_data))  # 15ms
    if fade_samples > 1 and np.issubdtype(audio_data.dtype, np.floating):
        audio_data = audio_data.copy()
        fade = np.linspace(1.0, 0.0, fade_samples, dtype=audio_data.dtype)
        audio_data[-fade_samples:] *= fade
    sd.play(audio_data, samplerate=sample_rate)
    sd.wait()
