"""The reversal view is an involution, and must stay one.

The anagram sampler will apply `view` to a latent and `view` again to undo it.
If reversal ever stopped being its own inverse, that would silently produce
noise rather than an error, so it is pinned here.
"""

import numpy as np

from ava.audio.perceive import VIEWS, forward_view, reverse_view


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
