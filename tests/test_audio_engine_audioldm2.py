"""The AudioLDM 2 engine's bookkeeping, checked without the model.

The sampler's arithmetic is `ava.audio.engine`'s and is tested there with
`dim=-2`. What is specific to this engine is how a duration becomes a latent
size, and that the time axis it flips is the mel latent's.
"""

import pytest
import torch

from ava.audio.engine_audioldm2 import TIME_DIM, AudioLDM2Engine, stack_padded


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
