"""The descriptors, on the layouts the generators actually return.

`spectral_centroid_halves` once split its input along the first axis, which
on (channels, n) audio is the channel axis: on stereo it quietly reported the
left and right channels as "first half" and "second half", and on mono (1, n)
it produced an empty second half and crashed. The halves must be halves of
time whatever the layout.
"""

import numpy as np

from ava.audio.features import describe, spectral_centroid_halves
from ava.audio.signals import rising_chirp

RATE = 16_000


def test_centroid_halves_split_time_for_every_layout() -> None:
    mono = rising_chirp(sample_rate=RATE, duration=2.0)
    first, second = spectral_centroid_halves(mono, RATE)
    assert second > first * 1.5

    one_channel = mono[None, :]
    assert spectral_centroid_halves(one_channel, RATE) == (first, second)

    stereo = np.stack([mono, mono])
    assert spectral_centroid_halves(stereo, RATE) == (first, second)


def test_describe_accepts_mono_with_a_channel_axis() -> None:
    wave = rising_chirp(sample_rate=RATE, duration=1.0)[None, :]
    out = describe(wave, RATE)
    assert out["centroid_second_half_hz"] > out["centroid_first_half_hz"]
    assert out["peak"] > 0.0
