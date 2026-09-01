"""Cheap descriptors that say what a probe signal actually contains.

These are not part of the judge. They exist so a result can be read without
trusting the docstrings: if `rising_chirp` is claimed to move its spectrum and
hold its level, that claim should be a number in the persisted result, next to
whatever CLAP said about it. They are also what lets the figures label a panel
without re-deriving anything.
"""

import numpy as np


def rms_profile(wave: np.ndarray, windows: int = 50) -> np.ndarray:
    """Level over time, as `windows` equal-length RMS values."""
    return np.array(
        [float(np.sqrt((chunk**2).mean())) for chunk in np.array_split(wave, windows)]
    )


def envelope_symmetry(wave: np.ndarray, windows: int = 50) -> float:
    """Correlation between the level profile and its reverse, in [-1, 1].

    Near 1 means reversal leaves the loudness contour alone, so any direction
    the judge reports cannot have come from the envelope.
    """
    profile = rms_profile(wave, windows)
    if float(profile.std()) == 0.0:
        return 1.0
    return float(np.corrcoef(profile, profile[::-1])[0, 1])


def spectral_centroid_halves(wave: np.ndarray, sample_rate: int) -> tuple[float, float]:
    """Spectral centre of mass of the first and second half, in Hz.

    Two numbers rather than a full trajectory because that is all the question
    needs: whether the spectrum goes somewhere, and which way.
    """
    centroids: list[float] = []
    for chunk in np.array_split(wave, 2):
        spectrum = np.abs(np.fft.rfft(chunk * np.hanning(chunk.size)))
        freqs = np.fft.rfftfreq(chunk.size, 1.0 / sample_rate)
        total = float(spectrum.sum())
        centroids.append(
            0.0 if total == 0.0 else float((spectrum * freqs).sum() / total)
        )
    return centroids[0], centroids[1]


def describe(wave: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Everything a figure or a reader needs about one signal."""
    first, second = spectral_centroid_halves(wave, sample_rate)
    return {
        "envelope_symmetry": envelope_symmetry(wave),
        "centroid_first_half_hz": first,
        "centroid_second_half_hz": second,
        "rms": float(np.sqrt((wave**2).mean())),
        "peak": float(np.max(np.abs(wave))),
    }
