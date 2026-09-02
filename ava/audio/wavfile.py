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
    """Write float audio in [-1, 1] as 16-bit PCM.

    Accepts mono as (n,) or multi-channel as (channels, n) -- the layout the
    generators hand back. Channels are interleaved, which is what the WAV
    container expects.
    """
    if wave_data.ndim == 1:
        channels, frames = 1, wave_data.reshape(1, -1)
    elif wave_data.ndim == 2:
        channels, frames = wave_data.shape[0], wave_data
    else:
        raise ValueError(f"expected (n,) or (channels, n), got {wave_data.shape}")

    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(frames, -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.T.tobytes())  # interleave
    return path
