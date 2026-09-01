"""WAV writing must be lossless enough that what is scored is what is heard."""

import wave
from pathlib import Path

import numpy as np
import pytest

from ava.audio.wavfile import write_wav


def test_round_trips_within_quantisation_error(tmp_path: Path) -> None:
    signal = np.sin(np.linspace(0, 40 * np.pi, 8000)) * 0.9
    path = write_wav(tmp_path / "probe.wav", signal, 48_000)

    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 48_000
        raw = handle.readframes(handle.getnframes())

    decoded = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32767.0
    assert decoded.size == signal.size
    assert np.max(np.abs(decoded - signal)) < 1.0 / 32767.0


def test_rejects_multichannel_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mono"):
        write_wav(tmp_path / "bad.wav", np.zeros((2, 100)), 48_000)
