"""The reversal view is an involution, and must stay one.

The anagram sampler will apply `view` to a latent and `view` again to undo it.
If reversal ever stopped being its own inverse, that would silently produce
noise rather than an error, so it is pinned here.
"""

import numpy as np

from ava.audio.perceive import VIEWS, forward_view, reverse_view
from ava.audio.resample import resample


def test_forward_view_is_the_identity() -> None:
    wave = np.linspace(-1.0, 1.0, 101)
    assert np.array_equal(forward_view(wave), wave)


def test_reverse_view_is_its_own_inverse() -> None:
    wave = np.linspace(-1.0, 1.0, 101)
    assert np.array_equal(reverse_view(reverse_view(wave)), wave)


def test_reverse_view_actually_reverses() -> None:
    assert np.array_equal(reverse_view(np.array([1.0, 2.0, 3.0])), [3.0, 2.0, 1.0])


def test_reverse_view_returns_a_contiguous_copy() -> None:
    """Negative strides break downstream buffers without raising."""
    wave = np.arange(10, dtype=np.float64)
    out = reverse_view(wave)
    assert out.flags["C_CONTIGUOUS"]
    out[0] = 99.0
    assert wave[-1] == 9.0


def test_both_views_are_registered() -> None:
    assert set(VIEWS) == {"forward", "reverse"}


def test_reverse_view_reverses_time_not_channels() -> None:
    """(channels, n) must be reversed along n.

    Indexing the first axis swaps left and right and leaves time untouched.
    Nothing raises, and on a near-symmetric stereo image it even sounds close
    to right, so this is pinned rather than trusted.
    """
    left = np.array([1.0, 2.0, 3.0])
    right = np.array([10.0, 20.0, 30.0])
    out = reverse_view(np.stack([left, right]))

    assert out.shape == (2, 3)
    assert out[0].tolist() == [3.0, 2.0, 1.0]
    assert out[1].tolist() == [30.0, 20.0, 10.0]


def test_reverse_view_is_its_own_inverse_for_stereo() -> None:
    stereo = np.arange(12, dtype=np.float64).reshape(2, 6)
    assert np.array_equal(reverse_view(reverse_view(stereo)), stereo)


def test_resample_is_a_no_op_at_the_same_rate() -> None:
    wave = np.linspace(-1.0, 1.0, 100)
    assert np.array_equal(resample(wave, 48_000, 48_000), wave)


def test_resample_changes_length_by_the_rate_ratio() -> None:
    wave = np.sin(np.linspace(0, 20 * np.pi, 44_100))
    out = resample(wave, 44_100, 48_000)
    assert abs(out.size - 48_000) <= 1


def test_resample_keeps_channels_separate() -> None:
    """Resampling the wrong axis would mix the channels into each other."""
    n = 4410
    left = np.sin(np.linspace(0, 20 * np.pi, n))
    stereo = np.stack([left, np.zeros(n)])
    out = resample(stereo, 44_100, 48_000)

    assert out.shape[0] == 2
    assert np.abs(out[1]).max() < 1e-6
    assert np.abs(out[0]).max() > 0.5


def test_resampling_does_not_wrap_the_end_into_the_beginning() -> None:
    """A Fourier resampler assumes periodicity; an anagram lives at the ends."""
    n = 4410
    wave = np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)])
    out = resample(wave, 44_100, 48_000)
    assert np.abs(out[: out.size // 4]).max() < 0.05
