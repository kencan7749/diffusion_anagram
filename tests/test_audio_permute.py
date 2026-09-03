"""The time jigsaw's permutation, on the latent and on the waveform.

The one property that matters is that the two sides cut at the same instants
and move the same blocks: a latent permuted by `BlockPermute` must decode to
what `permute_view` does to the waveform. That needs a model; what can be
fixed here is that both sides implement the same integer rule, that the
inverse undoes the view, and that the permutation is a fixed derangement.
"""

import numpy as np
import torch

from ava.audio.engine import (
    JIGSAW_VIEW,
    LATENT_VIEWS,
    TIME_REVERSE_VIEW,
    BlockPermute,
    anagram_epsilon,
    permute_blocks,
)
from ava.audio.perceive import VIEW_PARAMS, VIEWS, permute_view
from ava.audio.permute import (
    JIGSAW_BLOCKS,
    JIGSAW_PERM,
    JIGSAW_SEED,
    block_slices,
    derangement,
    inverse_permutation,
)


def test_the_jigsaw_permutation_is_a_fixed_derangement() -> None:
    assert JIGSAW_PERM == derangement(JIGSAW_BLOCKS, JIGSAW_SEED)
    assert sorted(JIGSAW_PERM) == list(range(JIGSAW_BLOCKS))
    assert all(source != i for i, source in enumerate(JIGSAW_PERM))
    assert derangement(4, 0) == derangement(4, 0)
    assert VIEW_PARAMS["jigsaw_4"] == {
        "blocks": 4,
        "seed": 0,
        "perm": list(JIGSAW_PERM),
    }


def test_inverse_permutation_undoes() -> None:
    perm = (2, 0, 3, 1)
    inverse = inverse_permutation(perm)
    assert [perm[inverse[i]] for i in range(4)] == [0, 1, 2, 3]


def test_block_slices_leave_the_remainder_alone() -> None:
    assert block_slices(10, 4) == [slice(0, 2), slice(2, 4), slice(4, 6), slice(6, 8)]


def test_permute_blocks_moves_whole_blocks_on_the_given_axis() -> None:
    latent = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    out = permute_blocks(latent, frames=8, perm=(1, 0, 3, 2))
    assert out.flatten().tolist() == [2, 3, 0, 1, 6, 7, 4, 5]
    # Past `frames` and the remainder within it are untouched.
    out = permute_blocks(latent, frames=6, perm=(1, 0))
    assert out.flatten().tolist() == [3, 4, 5, 0, 1, 2, 6, 7]
    mel = torch.arange(8, dtype=torch.float32).reshape(1, 1, 4, 2)
    out = permute_blocks(mel, frames=4, perm=(1, 0), dim=-2)
    assert out[0, 0].tolist() == [[4, 5], [6, 7], [0, 1], [2, 3]]


def test_block_permute_inverts_itself_and_agrees_with_the_waveform_view() -> None:
    view = BlockPermute(perm=(2, 0, 3, 1))
    latent = torch.randn(1, 3, 12)
    assert torch.equal(view.invert(view.apply(latent, 12), 12), latent)
    # 3 samples per frame: the waveform view must move the same stretches.
    wave = np.repeat(np.arange(12, dtype=np.float64), 3)[None, :]
    frames_out = view.apply(torch.arange(12, dtype=torch.float32).reshape(1, 1, 12), 12)
    expected = np.repeat(frames_out.flatten().numpy().astype(np.float64), 3)[None, :]
    assert np.array_equal(permute_view(view.perm)(wave, 8000), expected)


def test_the_registries_agree_on_names() -> None:
    assert set(LATENT_VIEWS) == {"time_reverse", "jigsaw_4"}
    assert JIGSAW_VIEW.perm == JIGSAW_PERM
    assert "jigsaw_4" in VIEWS and "whole" in VIEWS


def test_anagram_epsilon_inverts_the_view_before_averaging() -> None:
    """A branch that predicts the permuted forward estimate returns it unchanged."""
    view = BlockPermute(perm=(1, 0))
    forward = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4)
    zeros = torch.zeros_like(forward)
    prediction = torch.cat([zeros, forward, zeros, view.apply(forward, 4)])
    out = anagram_epsilon(prediction, 1.0, 4, view=view)
    assert torch.equal(out, forward)
    assert TIME_REVERSE_VIEW.invert(TIME_REVERSE_VIEW.apply(forward, 4), 4).equal(
        forward
    )
