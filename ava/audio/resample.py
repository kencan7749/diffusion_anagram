"""Rate conversion, so a generator and a judge need not agree on a sample rate.

Stable Audio Open produces 44.1 kHz; CLAP refuses anything but the 48 kHz it
was trained on. Rather than let that pin the choice of either, the conversion
happens where the two meet.
"""

import numpy as np
from scipy.signal import resample_poly


def resample(wave: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample along the last axis. Accepts (n,) or (channels, n).

    Polyphase rather than Fourier resampling: `resample_poly` filters with a
    windowed sinc and does not assume the signal is periodic, so it does not
    wrap the end of a clip into its beginning. For an anagram, whose whole
    point is what happens at the two ends of a sound, that assumption would be
    exactly the wrong one.
    """
    if from_rate == to_rate:
        return wave
    common = np.gcd(from_rate, to_rate)
    return np.ascontiguousarray(
        resample_poly(wave, to_rate // common, from_rate // common, axis=-1)
    )
