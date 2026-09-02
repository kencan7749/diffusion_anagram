"""Descriptors, and a distance between two signals.

None of this is part of a judge. The descriptors exist so a result can be read
without trusting the docstrings: if `rising_chirp` is claimed to move its
spectrum and hold its level, that claim should be a number in the persisted
result, next to whatever CLAP said about it.

`log_mel_distance` answers a different question -- are these two signals the
same sound -- which is what Step 0b needs when it asks whether flipping a
latent is the same operation as reversing a waveform.
"""

import numpy as np
from transformers.audio_utils import mel_filter_bank, spectrogram, window_function

# Analysis window for the spectral distance. 1024 samples is ~23 ms at 44.1 kHz
# -- long enough to resolve pitch, short enough to keep an attack from smearing
# across frames, which matters when the thing being measured is time direction.
FRAME_LENGTH = 1024
HOP_LENGTH = 256
NUM_MEL_FILTERS = 64


def log_mel(wave: np.ndarray, sample_rate: int) -> np.ndarray:
    """Log-mel magnitude spectrogram, shape (frames, mels).

    Used as the yardstick for "are these two signals the same sound". Log
    because loudness is logarithmic and a linear error would be dominated by
    the loudest frames; mel because a linear frequency axis would let inaudible
    high-frequency detail dominate a comparison meant to be perceptual.
    """
    mono = to_mono(wave).astype(np.float64)
    filters = mel_filter_bank(
        num_frequency_bins=FRAME_LENGTH // 2 + 1,
        num_mel_filters=NUM_MEL_FILTERS,
        min_frequency=20.0,
        max_frequency=sample_rate / 2.0,
        sampling_rate=sample_rate,
        norm="slaney",
        mel_scale="slaney",
    )
    power = spectrogram(
        mono,
        window=window_function(FRAME_LENGTH, "hann"),
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        power=2.0,
        mel_filters=filters,
    )
    # (mels, frames) -> (frames, mels); floor at -80 dB so silence does not
    # dominate the distance through log(0).
    return np.maximum(10.0 * np.log10(np.maximum(power, 1e-10)), -80.0).T


def log_mel_distance(a: np.ndarray, b: np.ndarray, sample_rate: int) -> float:
    """RMS difference of two log-mel spectrograms, in dB.

    Lengths are trimmed to the shorter of the two: a VAE round trip does not
    always return exactly as many samples as it was given, and a length
    mismatch is not the thing being measured.
    """
    x, y = log_mel(a, sample_rate), log_mel(b, sample_rate)
    frames = min(x.shape[0], y.shape[0])
    return float(np.sqrt(((x[:frames] - y[:frames]) ** 2).mean()))


def to_mono(wave: np.ndarray) -> np.ndarray:
    """Average channels. Generators return (channels, n); probes return (n,)."""
    return wave if wave.ndim == 1 else wave.mean(axis=0)


def rms_profile(wave: np.ndarray, windows: int = 50) -> np.ndarray:
    """Level over time, as `windows` equal-length RMS values.

    Downmixes first. Without that, a (channels, n) array from a generator gets
    split along its channel axis instead of time, which yields two values and
    then NaN -- a wrong answer rather than an error.
    """
    return np.array(
        [
            float(np.sqrt((chunk**2).mean()))
            for chunk in np.array_split(to_mono(wave), windows)
        ]
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
