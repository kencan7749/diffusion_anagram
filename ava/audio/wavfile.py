"""16-bit PCM WAV writing, on the standard library.

The experiment is about what a person hears, so every signal that gets scored
is also written to disk in a form that can be played. Using `wave` keeps this
free of a `soundfile` / `torchaudio` dependency the project does not otherwise
need.
"""

import wave
from pathlib import Path

import numpy as np


def write_wav(path: Path, wave_data: np.ndarray, sample_rate: int) -> Path:
    """Write a mono float array in [-1, 1] as 16-bit PCM."""
    if wave_data.ndim != 1:
        raise ValueError(f"expected a mono signal, got shape {wave_data.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(wave_data, -1.0, 1.0)
    pcm = (clipped * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return path
