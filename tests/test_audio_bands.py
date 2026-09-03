"""The low-pass projector, checked on synthetic latents.

Whether the map is a low-pass of real audio is Step 0c's measurement. What
can be fixed here is the algebra: the fit recovers a known map, the two
components of a noise estimate sum back exactly, the bias enters samples and
not noise, and a projector survives a save/load round trip.
"""

from pathlib import Path

import numpy as np
import torch

from ava.audio.bands import (
    LowpassProjector,
    fit_lowpass_projector,
    from_frames,
    to_frames,
)

C, T, F = 8, 6, 16
D = C * F


def _projector(seed: int = 0) -> LowpassProjector:
    rng = np.random.default_rng(seed)
    return LowpassProjector(
        weight=rng.normal(size=(D, D)) * 0.05,
        bias=rng.normal(size=D),
        cutoff_hz=750.0,
        meta={"clips": ["a", "b"]},
    )


def test_frames_round_trip_preserves_the_latent() -> None:
    latent = torch.arange(C * T * F, dtype=torch.float32).reshape(1, C, T, F)
    frames = to_frames(latent)
    assert frames.shape == (T, D)
    # Row t is frame t: all channels and bins of that time step.
    assert torch.equal(frames[2], latent[0, :, 2, :].reshape(-1))
    assert torch.equal(from_frames(frames, latent), latent)


def test_fit_recovers_a_known_affine_map() -> None:
    rng = np.random.default_rng(1)
    truth = _projector(seed=2)
    latents = [rng.normal(size=(50, D)) for _ in range(6)]
    lowpassed = [z @ truth.weight + truth.bias for z in latents]
    fitted = fit_lowpass_projector(latents, lowpassed, cutoff_hz=750.0, ridge=1e-8)
    assert np.allclose(fitted.weight, truth.weight, atol=1e-6)
    assert np.allclose(fitted.bias, truth.bias, atol=1e-6)
    assert fitted.meta["train_relative_residual"] < 1e-6
    assert fitted.meta["frames"] == 300


def test_split_epsilon_sums_to_the_whole_and_ignores_the_bias() -> None:
    proj = _projector()
    eps = torch.randn(1, C, T, F, dtype=torch.float64)
    low, high = proj.split_epsilon(eps)
    assert torch.allclose(low + high, eps)
    # No bias in the noise decomposition: a zero estimate splits into zeros.
    zero_low, zero_high = proj.split_epsilon(torch.zeros_like(eps))
    assert torch.equal(zero_low, torch.zeros_like(eps))
    assert torch.equal(zero_high, torch.zeros_like(eps))


def test_low_applies_the_affine_map_frame_by_frame() -> None:
    proj = _projector()
    latent = torch.randn(1, C, T, F, dtype=torch.float64)
    low = proj.low(latent)
    expected = to_frames(latent).numpy() @ proj.weight + proj.bias
    # The map is applied in float32 on purpose (see LowpassProjector._tensors).
    assert np.allclose(to_frames(low).numpy(), expected, atol=1e-5)
    assert low.shape == latent.shape


def test_low_keeps_dtype_and_device_of_the_latent() -> None:
    proj = _projector()
    latent = torch.randn(1, C, T, F, dtype=torch.float16)
    assert proj.low(latent).dtype == torch.float16


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    proj = _projector()
    saved = proj.save(tmp_path / "projector_750hz")
    assert saved.suffix == ".npz"
    assert (tmp_path / "projector_750hz.json").exists()
    back = LowpassProjector.load(tmp_path / "projector_750hz")
    assert np.array_equal(back.weight, proj.weight)
    assert np.array_equal(back.bias, proj.bias)
    assert back.cutoff_hz == 750.0
    assert back.meta == {"clips": ["a", "b"]}


def test_shapes_are_checked() -> None:
    import pytest

    with pytest.raises(ValueError):
        LowpassProjector(np.zeros((3, 2)), np.zeros(3), 1.0)
    with pytest.raises(ValueError):
        LowpassProjector(np.zeros((3, 3)), np.zeros(2), 1.0)
    with pytest.raises(ValueError):
        to_frames(torch.zeros(2, C, T, F))
