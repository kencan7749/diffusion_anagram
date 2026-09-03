"""The anagram arithmetic, checked without the model.

Everything expensive in the sampler is the transformer; everything that can be
wrong in a way that still produces plausible audio is the bookkeeping around
it. A branch added in the wrong time frame, or a view that is not its own
inverse, yields noise that converges and sounds like something -- so the parts
that can be checked on hand-written tensors are checked here.
"""

import pytest
import torch

from ava.audio.engine import anagram_epsilon, flip_window
from ava.audio.spec import AudioCandidateSpec


def _latent(values: list[float]) -> torch.Tensor:
    """(1, 1, T) latent from a list, so positions are readable in a failure."""
    return torch.tensor(values).reshape(1, 1, -1)


def test_flip_window_reverses_only_the_occupied_prefix() -> None:
    """The audio is a prefix of a much longer latent; the tail is not ours."""
    out = flip_window(_latent([1, 2, 3, 9, 9, 9]), frames=3)
    assert out.flatten().tolist() == [3, 2, 1, 9, 9, 9]


def test_flip_window_is_its_own_inverse() -> None:
    latent = _latent([1, 2, 3, 4, 5, 6, 7])
    assert torch.equal(flip_window(flip_window(latent, 4), 4), latent)


def test_flip_window_flips_everything_when_the_window_covers_the_latent() -> None:
    out = flip_window(_latent([1, 2, 3]), frames=3)
    assert out.flatten().tolist() == [3, 2, 1]
    out = flip_window(_latent([1, 2, 3]), frames=99)
    assert out.flatten().tolist() == [3, 2, 1]


def test_flip_window_does_not_mutate_its_input() -> None:
    latent = _latent([1, 2, 3, 4])
    flip_window(latent, 2)
    assert latent.flatten().tolist() == [1, 2, 3, 4]


def test_guidance_is_applied_within_each_branch() -> None:
    """Rows are [uncond, forward, uncond, reverse]; scale acts on each pair."""
    prediction = torch.cat(
        [
            _latent([0.0, 0.0]),  # uncond, forward branch
            _latent([1.0, 1.0]),  # cond, forward branch
            _latent([0.0, 0.0]),  # uncond, reverse branch
            _latent([2.0, 2.0]),  # cond, reverse branch
        ]
    )
    # forward -> 0 + 3*(1-0) = 3; reverse -> 0 + 3*(2-0) = 6; mean 4.5.
    out = anagram_epsilon(prediction, guidance_scale=3.0, frames=2)
    assert out.flatten().tolist() == pytest.approx([4.5, 4.5])


def test_reverse_branch_is_flipped_back_before_averaging() -> None:
    """Two branches that agree must average to themselves, not to a smear.

    If the reverse prediction were added without being flipped back, agreeing
    branches would cancel into something neither of them said. Here the reverse
    branch predicts the mirror of the forward one, so a correct implementation
    returns the forward prediction unchanged.
    """
    forward = [1.0, 2.0, 3.0]
    prediction = torch.cat(
        [
            _latent([0.0, 0.0, 0.0]),
            _latent(forward),
            _latent([0.0, 0.0, 0.0]),
            _latent(forward[::-1]),
        ]
    )
    out = anagram_epsilon(prediction, guidance_scale=1.0, frames=3)
    assert out.flatten().tolist() == pytest.approx(forward)


def test_guidance_scale_of_one_passes_the_conditional_through() -> None:
    prediction = torch.cat(
        [_latent([5.0]), _latent([7.0]), _latent([5.0]), _latent([7.0])]
    )
    out = anagram_epsilon(prediction, guidance_scale=1.0, frames=1)
    assert out.flatten().tolist() == pytest.approx([7.0])


def test_spec_uid_distinguishes_the_two_prompt_orders() -> None:
    """Swapping the prompts is a different candidate, not the same one."""
    a = AudioCandidateSpec("a hit that decays", "a swell that stops")
    b = AudioCandidateSpec("a swell that stops", "a hit that decays")
    assert a.uid() != b.uid()
    assert a.prompts == ["a hit that decays", "a swell that stops"]


# ---- the time axis need not be the last one (AudioLDM 2: (B, C, T, mel)) ----


def _mel_latent(rows: list[list[float]]) -> torch.Tensor:
    """(1, 1, T, mel) latent from rows, one per time frame."""
    return torch.tensor(rows).reshape(1, 1, len(rows), -1)


def test_flip_window_on_the_second_to_last_axis_leaves_mel_bins_alone() -> None:
    latent = _mel_latent([[1, 10], [2, 20], [3, 30], [9, 90]])
    out = flip_window(latent, frames=3, dim=-2)
    assert out[0, 0].tolist() == [[3, 30], [2, 20], [1, 10], [9, 90]]
    out = flip_window(latent, frames=4, dim=-2)
    assert out[0, 0].tolist() == [[9, 90], [3, 30], [2, 20], [1, 10]]


def test_anagram_epsilon_flips_the_reverse_branch_along_the_given_axis() -> None:
    forward = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
    zeros = [[0.0, 0.0]] * 3
    prediction = torch.cat(
        [
            _mel_latent(zeros),
            _mel_latent(forward),
            _mel_latent(zeros),
            _mel_latent(forward[::-1]),
        ]
    )
    out = anagram_epsilon(prediction, guidance_scale=1.0, frames=3, dim=-2)
    assert torch.allclose(out[0, 0], torch.tensor(forward))
