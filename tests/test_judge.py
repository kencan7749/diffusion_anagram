"""Properties of the illusion score J.

The model-free tests pin down the arithmetic. The model-backed test (marked
slow) is Step 0-1: the known-good example must score above chance on both views.
"""

from pathlib import Path

import pytest
import torch

from ava.judge import scores_to_probs
from ava.perceive import blur_params, far_view, far_view_resize, near_view
from ava.spec import KERNEL_SIZE, SIGMA, CandidateSpec, Verdict

# CLIP ViT-L/14 ships logit_scale = ln(100), so exp() is 100.
LOGIT_SCALE = 100.0

SMOKE_IMAGE = Path("results/hybrid_smoke/0000/sample_256.png")
SMOKE_SPEC = CandidateSpec(
    prompt_low="a painting of a panda",
    prompt_high="a painting of a flower arrangement",
)


def _matrix(far_low: float, far_high: float, near_low: float, near_high: float):
    """Build S[view][prompt] with both axes ordered (low, high)."""
    return torch.tensor([[far_low, far_high], [near_low, near_high]])


def test_chance_level_is_one_half() -> None:
    """A view that cannot tell the prompts apart must score exactly 0.5."""
    p_far, p_near, j = scores_to_probs(_matrix(0.3, 0.3, 0.3, 0.3), LOGIT_SCALE)
    assert p_far == pytest.approx(0.5)
    assert p_near == pytest.approx(0.5)
    assert j == pytest.approx(0.5)


def test_j_is_min_not_mean() -> None:
    """A candidate perfect on one view and failing the other must rank low.

    This is the whole reason J is a min. Case A wins the far view outright and
    loses the near view; case B is mediocre but correct on both. Averaging
    ranks A above B, which would fill the search with images where only one
    prompt is ever visible. The min must rank B above A.
    """
    a = _matrix(far_low=0.30, far_high=0.20, near_low=0.30, near_high=0.2861)
    b = _matrix(far_low=0.301, far_high=0.299, near_low=0.299, near_high=0.301)

    a_far, a_near, a_j = scores_to_probs(a, LOGIT_SCALE)
    b_far, b_near, b_j = scores_to_probs(b, LOGIT_SCALE)

    # A really is lopsided and B really is balanced-but-correct.
    assert a_far > 0.99 and a_near < 0.25
    assert b_far > 0.5 and b_near > 0.5

    # Averaging prefers A; the min must prefer B.
    assert (a_far + a_near) / 2 > (b_far + b_near) / 2
    assert a_j < b_j


def test_softmax_saturates_at_clip_logit_scale() -> None:
    """Documents the dynamic range J actually has.

    With logit_scale = 100 a cosine gap of 0.05 already pins p at 0.99, so J is
    close to binary and is a screening signal rather than a fine-grained
    ranking. Recorded as a test so a future change to the temperature is a
    deliberate decision rather than an accident.
    """
    small, _, _ = scores_to_probs(_matrix(0.30, 0.295, 0.3, 0.3), LOGIT_SCALE)
    large, _, _ = scores_to_probs(_matrix(0.30, 0.250, 0.3, 0.3), LOGIT_SCALE)
    assert small == pytest.approx(0.6225, abs=1e-3)
    assert large > 0.99


def test_j_never_exceeds_either_view() -> None:
    p_far, p_near, j = scores_to_probs(_matrix(0.35, 0.20, 0.24, 0.26), LOGIT_SCALE)
    assert j == min(p_far, p_near)
    assert 0.0 <= j <= 1.0


def test_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        scores_to_probs(torch.zeros(3, 3), LOGIT_SCALE)


def test_verdict_diagnosis_covers_each_failure_mode() -> None:
    def verdict(p_far: float, p_near: float) -> Verdict:
        return Verdict(
            uid="x",
            s_far_low=0.0,
            s_far_high=0.0,
            s_near_low=0.0,
            s_near_high=0.0,
            p_far=p_far,
            p_near=p_near,
            j=min(p_far, p_near),
        )

    assert verdict(0.8, 0.8).diagnose() == "ok"
    assert verdict(0.2, 0.2).diagnose() == "pair_mismatch"
    assert verdict(0.2, 0.8).diagnose() == "low_loses"
    assert verdict(0.8, 0.2).diagnose() == "high_absent"


def test_blur_params_match_inverse_view_rule() -> None:
    """perceive.blur_params must reproduce view_hybrid.inverse_view's scaling."""
    for size in (64, 256, 1024):
        factor = size // 64
        k, sigma = blur_params(size)
        assert k == KERNEL_SIZE * factor + ((factor + 1) % 2)
        assert sigma == SIGMA * factor
        assert k % 2 == 1


def test_near_view_is_the_image_itself() -> None:
    """Guards against near_view drifting back to the sampler's (img - blur)."""
    torch.manual_seed(0)
    img = torch.rand(3, 64, 64)
    assert near_view(img) is img


@pytest.mark.parametrize("far_fn", [far_view, far_view_resize])
def test_far_view_removes_high_frequency_energy(far_fn) -> None:
    torch.manual_seed(0)
    img = torch.rand(3, 256, 256)
    far = far_fn(img)
    assert far.shape == img.shape
    # A far view must be smoother than the original by any sane measure.
    assert far.std() < img.std()


@pytest.mark.slow
def test_known_good_example_scores_above_chance() -> None:
    """Step 0-1. Requires CUDA and the CLIP/BLIP weights."""
    from ava.judge import ClipBlipJudge, load_image

    if not SMOKE_IMAGE.exists():
        pytest.skip(f"{SMOKE_IMAGE} not generated")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    judge = ClipBlipJudge(use_blip=False)
    verdict = judge.evaluate(load_image(SMOKE_IMAGE), SMOKE_SPEC)

    assert verdict.p_far > 0.5, f"far view failed: {verdict.to_json()}"
    assert verdict.p_near > 0.5, f"near view failed: {verdict.to_json()}"
