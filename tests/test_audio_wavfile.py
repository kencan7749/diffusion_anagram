"""WAV writing must be lossless enough that what is scored is what is heard."""

import wave
from pathlib import Path

import numpy as np
import pytest

from ava.audio.wavfile import write_wav


def _read(path: Path) -> tuple[np.ndarray, int, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        assert handle.getsampwidth() == 2
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    decoded = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32767.0
    return decoded, channels, rate


def test_mono_round_trips_within_quantisation_error(tmp_path: Path) -> None:
    signal = np.sin(np.linspace(0, 40 * np.pi, 8000)) * 0.9
    decoded, channels, rate = _read(write_wav(tmp_path / "probe.wav", signal, 48_000))

    assert channels == 1
    assert rate == 48_000
    assert decoded.size == signal.size
    assert np.max(np.abs(decoded - signal)) < 1.0 / 32767.0


def test_stereo_channels_are_interleaved_not_concatenated(tmp_path: Path) -> None:
    """A concatenating writer passes a length check and plays back as nonsense."""
    left = np.full(64, 0.5)
    right = np.full(64, -0.5)
    decoded, channels, _ = _read(
        write_wav(tmp_path / "stereo.wav", np.stack([left, right]), 44_100)
    )

    assert channels == 2
    assert decoded.size == 128
    assert np.allclose(decoded[0::2], 0.5, atol=1e-4)
    assert np.allclose(decoded[1::2], -0.5, atol=1e-4)


def test_rejects_more_than_two_dimensions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"channels"):
        write_wav(tmp_path / "bad.wav", np.zeros((1, 2, 100)), 48_000)
