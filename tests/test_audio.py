import numpy as np

from mops_voice.audio import SAMPLE_RATE, CHANNELS, audio_to_wav_bytes


def test_sample_rate_is_16k():
    assert SAMPLE_RATE == 16000


def test_channels_is_mono():
    assert CHANNELS == 1


def test_audio_to_wav_bytes_returns_bytes():
    # 1 second of silence
    audio_data = np.zeros((16000, 1), dtype=np.int16)
    wav_bytes = audio_to_wav_bytes(audio_data)
    assert isinstance(wav_bytes, bytes)
    assert len(wav_bytes) > 44  # WAV header is 44 bytes
    assert wav_bytes[:4] == b"RIFF"


def test_audio_to_wav_bytes_empty_audio():
    audio_data = np.zeros((0, 1), dtype=np.int16)
    wav_bytes = audio_to_wav_bytes(audio_data)
    assert wav_bytes is None
