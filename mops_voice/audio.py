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
    """Play audio through speakers. Blocks until complete."""
    sd.play(audio_data, samplerate=sample_rate)
    sd.wait()
