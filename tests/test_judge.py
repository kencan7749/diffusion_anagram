"""Properties of the illusion score J, for two views and for N.

The model-free tests pin down the arithmetic. The model-backed test (marked
slow) is Step 0-1: the known-good example must score above chance on both views.
"""

from pathlib import Path

import pytest
import torch

from ava.image.perceive import blur_params, far_view, far_view_resize, near_view
from ava.metric import alignment, concealment, multiway_probs, scores_to_probs
from ava.spec import KERNEL_SIZE, SIGMA, CandidateSpec

# CLIP ViT-L/14 ships logit_scale = ln(100), so exp() is 100.
LOGIT_SCALE = 100.0

SMOKE_IMAGE = Path("results/hybrid_smoke/0000/sample_256.png")
SMOKE_SPEC = CandidateSpec(
    task="hybrid",
    prompts=("a panda", "a flower arrangement"),
    style="a painting of",
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


def test_two_view_helper_rejects_other_shapes() -> None:
    with pytest.raises(ValueError):
        scores_to_probs(torch.zeros(3, 3), LOGIT_SCALE)


# -- N views ------------------------------------------------------------------


def test_multiway_agrees_with_the_two_view_form() -> None:
    s = _matrix(0.35, 0.20, 0.24, 0.26)
    p, j = multiway_probs(s, LOGIT_SCALE)
    assert tuple(p) + (j,) == pytest.approx(scores_to_probs(s, LOGIT_SCALE))


def test_multiway_chance_level_is_one_over_n() -> None:
    p, j = multiway_probs(torch.full((3, 3), 0.3), LOGIT_SCALE)
    assert p == pytest.approx([1 / 3] * 3)
    assert j == pytest.approx(1 / 3)


def test_multiway_j_is_the_weakest_view() -> None:
    s = torch.tensor([[0.30, 0.20, 0.20], [0.20, 0.30, 0.20], [0.25, 0.26, 0.27]])
    p, j = multiway_probs(s, LOGIT_SCALE)
    assert j == p[2] == min(p)
    # A 0.01 gap over the runner-up at logit_scale 100 is a factor of e, so the
    # third view keeps only about two thirds of its row: 1 / (1 + e^-1 + e^-2).
    assert p[2] == pytest.approx(0.665, abs=1e-3)
    assert p[0] > 0.99 and p[1] > 0.99


def test_multiway_rejects_non_square_and_degenerate_matrices() -> None:
    with pytest.raises(ValueError):
        multiway_probs(torch.zeros(2, 3), LOGIT_SCALE)
    with pytest.raises(ValueError):
        multiway_probs(torch.zeros(1, 1), LOGIT_SCALE)
    with pytest.raises(ValueError):
        multiway_probs(torch.zeros(4), LOGIT_SCALE)


def test_j_and_sep_min_can_disagree_with_three_views() -> None:
    """With N > 2 the two orderings are not the same one; the README says so.

    Candidate A has the larger runner-up margin on its weakest view but a
    strong third competitor, so its softmax mass is split three ways; B has a
    smaller margin but no third competitor. sep_min prefers A, J prefers B.
    """
    from ava.spec import Verdict

    def verdict(s: torch.Tensor) -> Verdict:
        p, j = multiway_probs(s, LOGIT_SCALE)
        return Verdict(
            uid="x",
            task="three_view",
            slots=["a", "b", "c"],
            scores=s.tolist(),
            p=p,
            j=j,
            alignment=alignment(s),
            concealment=concealment(s, LOGIT_SCALE),
        )

    a = verdict(torch.tensor([[0.30, 0.29, 0.29], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]))
    b = verdict(torch.tensor([[0.30, 0.292, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]))
    assert a.sep_min > b.sep_min
    assert a.j < b.j


def test_alignment_is_the_worst_diagonal_entry() -> None:
    s = torch.tensor([[0.30, 0.1, 0.1], [0.1, 0.25, 0.1], [0.1, 0.1, 0.28]])
    assert alignment(s) == pytest.approx(0.25)


def test_concealment_is_chance_when_nothing_is_distinguishable() -> None:
    assert concealment(torch.full((2, 2), 0.3), LOGIT_SCALE) == pytest.approx(0.5)
    assert concealment(torch.full((4, 4), 0.3), LOGIT_SCALE) == pytest.approx(0.25)


def test_concealment_averages_both_softmax_directions() -> None:
    """A prompt that every view likes hurts the column direction, not the rows.

    With an equal diagonal the two directions see the same two margins, so the
    diagonal is made unequal here to keep the check from passing by accident.
    """
    s = torch.tensor([[0.30, 0.20], [0.29, 0.35]])
    rows = (s * LOGIT_SCALE).softmax(-1).diagonal().mean()
    cols = (s * LOGIT_SCALE).softmax(-2).diagonal().mean()
    assert concealment(s, LOGIT_SCALE) == pytest.approx(float((rows + cols) / 2))
    assert rows != pytest.approx(cols)


# -- perceptual helpers kept from the two-view days ---------------------------


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


@pytest.mark.parametrize("fn", [far_view, far_view_resize])
def test_perceptual_helpers_treat_a_batch_as_a_batch(fn) -> None:
    """(B, C, H, W) must give the same answer per image as (C, H, W) does."""
    torch.manual_seed(0)
    batch = torch.rand(2, 3, 64, 64)
    out = fn(batch)
    assert out.shape == batch.shape
    torch.testing.assert_close(out[1], fn(batch[1]), rtol=1e-4, atol=1e-5)


@pytest.mark.slow
def test_known_good_example_scores_above_chance() -> None:
    """Step 0-1. Requires CUDA and the CLIP/BLIP weights."""
    from ava.image.judge import ClipBlipJudge, load_image
    from ava.image.tasks import get_task
    from ava.image.views import ViewSet

    if not SMOKE_IMAGE.exists():
        pytest.skip(f"{SMOKE_IMAGE} not generated")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    judge = ClipBlipJudge(use_blip=False)
    viewset = ViewSet.build(get_task("hybrid"))
    verdict = judge.evaluate(load_image(SMOKE_IMAGE), SMOKE_SPEC, viewset)

    assert verdict.holds == [True, True], verdict.to_json()
