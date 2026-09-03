"""The AudioLDM 2 engine's bookkeeping, checked without the model.

The sampler's arithmetic is `ava.audio.engine`'s and is tested there with
`dim=-2`. What is specific to this engine is how a duration becomes a latent
size, and that the time axis it flips is the mel latent's.
"""

import numpy as np
import pytest
import torch

from ava.audio.bands import LowpassProjector
from ava.audio.engine_audioldm2 import (
    TIME_DIM,
    AudioLDM2Engine,
    AudioLDM2HybridEngine,
    hybrid_epsilon,
    stack_padded,
)


def test_the_time_axis_is_the_second_to_last() -> None:
    """(B, C, T, mel): flipping the last axis would reverse pitch, not time."""
    assert TIME_DIM == -2


@pytest.mark.parametrize(
    ("duration_s", "frames"),
    [(5.0, 125), (10.0, 250), (10.24, 256), (5.01, 126), (0.01, 1)],
)
def test_latent_frames_round_up_to_the_forty_millisecond_grid(
    duration_s: float, frames: int
) -> None:
    engine = AudioLDM2Engine(device="cpu")
    assert engine.latent_frames(duration_s) == frames


def test_effective_duration_is_never_shorter_than_requested() -> None:
    engine = AudioLDM2Engine(device="cpu")
    assert engine.effective_duration_s(5.0) == pytest.approx(5.0)
    assert engine.effective_duration_s(5.01) == pytest.approx(5.04)
    assert engine.effective_duration_s(10.24) == pytest.approx(10.24)


def test_construction_does_not_load_anything() -> None:
    engine = AudioLDM2Engine(device="cpu")
    assert engine._pipe is None


def test_stack_padded_pads_sequences_and_masks_to_the_longest() -> None:
    """Prompts tokenise to different lengths; rows are padded with zeros."""
    short = torch.ones(1, 2, 3)
    long = torch.full((1, 4, 3), 2.0)
    out = stack_padded([short, long, short, long])
    assert out.shape == (4, 4, 3)
    assert out[0, :2].tolist() == short[0].tolist()
    assert out[0, 2:].abs().sum() == 0
    assert torch.equal(out[1], long[0])

    mask = stack_padded(
        [torch.ones(1, 2, dtype=torch.long), torch.ones(1, 4, dtype=torch.long)]
    )
    assert mask.tolist() == [[1, 1, 0, 0], [1, 1, 1, 1]]


def test_stack_padded_leaves_equal_lengths_alone() -> None:
    rows = [torch.rand(1, 8, 5) for _ in range(4)]
    assert torch.equal(stack_padded(rows), torch.cat(rows))


# ---- the frequency hybrid's arithmetic -----------------------------------------

_C, _T, _F = 2, 3, 4
_D = _C * _F


def _split(weight: np.ndarray) -> LowpassProjector:
    return LowpassProjector(weight=weight, bias=np.zeros(_D), cutoff_hz=750.0)


def _rows(near: float, far: float) -> torch.Tensor:
    """[uncond, near, uncond, far] with constant estimates, uncond = 0."""
    zeros = torch.zeros(1, _C, _T, _F, dtype=torch.float64)
    return torch.cat(
        [zeros, torch.full_like(zeros, near), zeros, torch.full_like(zeros, far)]
    )


def test_hybrid_takes_the_low_band_from_far_and_the_rest_from_near() -> None:
    everything_is_low = _split(np.eye(_D))
    nothing_is_low = _split(np.zeros((_D, _D)))
    prediction = _rows(near=1.0, far=5.0)
    assert torch.allclose(
        hybrid_epsilon(prediction, 1.0, everything_is_low),
        torch.full((1, _C, _T, _F), 5.0, dtype=torch.float64),
    )
    assert torch.allclose(
        hybrid_epsilon(prediction, 1.0, nothing_is_low),
        torch.full((1, _C, _T, _F), 1.0, dtype=torch.float64),
    )


def test_hybrid_returns_agreeing_branches_unchanged() -> None:
    """The components sum to the identity, so agreement passes straight through."""
    rng = np.random.default_rng(0)
    projector = _split(rng.normal(size=(_D, _D)))
    estimate = torch.randn(1, _C, _T, _F, dtype=torch.float64)
    zeros = torch.zeros_like(estimate)
    prediction = torch.cat([zeros, estimate, zeros, estimate])
    assert torch.allclose(hybrid_epsilon(prediction, 1.0, projector), estimate)


def test_hybrid_applies_guidance_within_each_branch() -> None:
    # uncond 0, cond 1 -> 0 + 3 * (1 - 0) = 3 on the near side; far = 0.
    prediction = _rows(near=1.0, far=0.0)
    out = hybrid_epsilon(prediction, 3.0, _split(np.zeros((_D, _D))))
    assert torch.allclose(out, torch.full_like(out, 3.0))


def test_hybrid_engine_needs_a_projector_and_loads_nothing() -> None:
    engine = AudioLDM2HybridEngine(projector=_split(np.eye(_D)), device="cpu")
    assert engine._pipe is None
    assert engine.projector.cutoff_hz == 750.0
