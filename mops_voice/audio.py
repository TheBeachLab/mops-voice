"""Audio recording and playback via sounddevice."""

import io
import time
import wave

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

# Persistent output stream. sd.play() opens and closes a CoreAudio
# stream per call, which on macOS reliably produces a click on close.
# Keep one stream open for the whole session instead.
_output_stream: sd.OutputStream | None = None
_output_sr: int | None = None


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


def _get_output_stream(sample_rate: int) -> sd.OutputStream:
    """Lazily open (or reopen on sample-rate change) the persistent stream."""
    global _output_stream, _output_sr
    if _output_stream is None or _output_sr != sample_rate:
        if _output_stream is not None:
            _output_stream.stop()
            _output_stream.close()
        _output_stream = sd.OutputStream(
            samplerate=sample_rate, channels=1, dtype="float32",
        )
        _output_stream.start()
        _output_sr = sample_rate
    return _output_stream


def close_output_stream() -> None:
    """Stop and close the persistent output stream. Call at shutdown."""
    global _output_stream, _output_sr
    if _output_stream is not None:
        _output_stream.stop()
        _output_stream.close()
        _output_stream = None
        _output_sr = None


def play_audio(audio_data: np.ndarray, sample_rate: int = 24000) -> None:
    """Play audio through the persistent output stream. Blocks until
    playback is complete (not just until the buffer is queued), so the
    caller's next step happens *after* the sound has actually finished."""
    stream = _get_output_stream(sample_rate)
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32, copy=False)
    if audio_data.ndim == 1:
        audio_data = audio_data.reshape(-1, 1)
    duration = len(audio_data) / sample_rate
    t0 = time.monotonic()
    stream.write(audio_data)
    # write() returns once the data is buffered, not when it's played.
    # Sleep the remainder so the caller sees true playback completion.
    remaining = (t0 + duration) - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)
