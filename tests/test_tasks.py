"""The task registry: every view must satisfy its paper's algebraic condition.

Factorized Diffusion decomposes the epsilon prediction into components that
must sum back to the identity; that is what justifies `reduction == "sum"`.
Visual Anagrams transforms the noisy image and needs the transform to be
orthogonal so the noise stays Gaussian; a permutation or a sign flip preserves
the multiset of absolute values exactly. If either broke, every score
downstream would be measuring the wrong thing.
"""

from __future__ import annotations

import pytest
import torch

from ava.image import perceive
from ava.image.tasks import FD, TASKS, VA, IllusionTask, get_task
from ava.image.views import ViewSet
from ava.spec import KERNEL_SIZE, SIGMA

SEED = 0
FD_TASKS = [t for t in TASKS.values() if t.paper == FD]
VA_TASKS = [t for t in TASKS.values() if t.paper == VA]

# Building the jigsaw permutation at 1024 px takes a few seconds; do it once.
_VIEWSETS: dict[str, ViewSet] = {}


def viewset(task: IllusionTask) -> ViewSet:
    if task.name not in _VIEWSETS:
        _VIEWSETS[task.name] = ViewSet.build(task, seed=SEED)
    return _VIEWSETS[task.name]


def _noise(size: int) -> torch.Tensor:
    torch.manual_seed(SEED)
    # 6 channels mimics the sampler's layout: 3 noise + 3 variance channels.
    return torch.randn(6, size, size)


# -- registry shape -----------------------------------------------------------


def test_every_paper_example_type_is_registered() -> None:
    expected = {
        "flip",
        "rotate_cw",
        "rotate_ccw",
        "rotate_180",
        "skew",
        "jigsaw",
        "inner_circle",
        "negate",
        "patch_permute",
        "pixel_permute",
        "square_hinge",
        "three_view",
        "four_view",
        "hybrid",
        "triple_hybrid",
        "color_hybrid",
        "motion_hybrid",
        "inverse_hybrid",
    }
    assert set(TASKS) == expected


def test_reduction_follows_the_paper() -> None:
    for task in FD_TASKS:
        assert task.reduction == "sum", task.name
    for task in VA_TASKS:
        assert task.reduction == "mean", task.name


def test_va_slots_are_symmetric_and_fd_slots_are_not() -> None:
    for task in VA_TASKS:
        assert task.symmetric, task.name
    for task in FD_TASKS:
        assert not task.symmetric, task.name
        assert len(set(task.roles)) == task.n_views


def test_every_task_cites_its_source() -> None:
    for task in TASKS.values():
        assert task.citation, task.name


def test_sr_slot_is_the_finest_component_for_fd_and_identity_for_va() -> None:
    """The upscaler knows nothing about views (FD App. A.4; VA Sec. 4)."""
    for task in VA_TASKS:
        assert task.slots[task.sr_slot].name == "identity", task.name
    assert TASKS["hybrid"].slots[TASKS["hybrid"].sr_slot].name == "high"
    assert TASKS["triple_hybrid"].slots[TASKS["triple_hybrid"].sr_slot].name == "high"
    assert TASKS["color_hybrid"].slots[TASKS["color_hybrid"].sr_slot].name == "color"


def test_inverse_hybrid_pins_the_low_slot() -> None:
    task = get_task("inverse_hybrid")
    assert task.ref_slot == 0
    assert task.searched_slots() == [1]
    assert get_task("hybrid").searched_slots() == [0, 1]


def test_unknown_task_names_are_rejected() -> None:
    with pytest.raises(KeyError):
        get_task("hexagon")


def test_registry_rejects_mismatched_slots_and_views() -> None:
    with pytest.raises(ValueError):
        IllusionTask(
            name="broken",
            paper=VA,
            slots=TASKS["flip"].slots,
            view_names=("identity",),
            reduction="mean",
            citation="test",
        )


# -- Factorized Diffusion: components sum to the identity ----------------------


@pytest.mark.parametrize("task", FD_TASKS, ids=lambda t: t.name)
@pytest.mark.parametrize("size", [64, 256])
def test_fd_components_reconstruct_the_input(task: IllusionTask, size: int) -> None:
    noise = _noise(size)
    # inverse_view mutates noise[:3] in place, so each view gets its own copy.
    total = sum(v.inverse_view(noise.clone())[:3] for v in viewset(task).views)
    torch.testing.assert_close(total, noise[:3], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("task", FD_TASKS, ids=lambda t: t.name)
def test_fd_views_leave_the_variance_channels_untouched(task: IllusionTask) -> None:
    noise = _noise(64)
    for view in viewset(task).views:
        torch.testing.assert_close(view.inverse_view(noise.clone())[3:], noise[3:])


def test_mismatched_sigma_breaks_the_hybrid_decomposition() -> None:
    """Guards the reason sigma is a shared module constant rather than per-view."""
    from visual_anagrams.views.view_hybrid import HybridHighPassView, HybridLowPassView

    noise = _noise(64)
    lp = HybridLowPassView(SIGMA, KERNEL_SIZE).inverse_view(noise.clone())
    hp = HybridHighPassView(SIGMA * 3, KERNEL_SIZE).inverse_view(noise.clone())
    assert not torch.allclose(lp[:3] + hp[:3], noise[:3], rtol=1e-3, atol=1e-3)


# -- Visual Anagrams: views are invertible and orthogonal ----------------------


@pytest.mark.parametrize("task", VA_TASKS, ids=lambda t: t.name)
@pytest.mark.parametrize("size", [64, 256])
def test_va_views_invert_exactly(task: IllusionTask, size: int) -> None:
    noise = _noise(size)
    for view in viewset(task).views:
        back = view.inverse_view(view.view(noise.clone()))
        # Only the noise channels: negation deliberately leaves the variance
        # estimate alone (VA App. A), so a 6-channel round trip is not identity.
        torch.testing.assert_close(back[:3], noise[:3])


@pytest.mark.parametrize("task", VA_TASKS, ids=lambda t: t.name)
def test_va_views_preserve_the_noise_distribution(task: IllusionTask) -> None:
    """Orthogonal: the multiset of |values| is unchanged, so Gaussian stays Gaussian."""
    noise = _noise(64)
    for view in viewset(task).views:
        moved = view.view(noise.clone())[:3]
        a = torch.sort(moved.flatten().abs()).values
        b = torch.sort(noise[:3].flatten().abs()).values
        torch.testing.assert_close(a, b, rtol=0, atol=1e-6)


# -- perceptual views ---------------------------------------------------------


@pytest.mark.parametrize("task", list(TASKS.values()), ids=lambda t: t.name)
def test_perceive_returns_one_image_per_slot_in_range(task: IllusionTask) -> None:
    torch.manual_seed(SEED)
    img = torch.rand(3, 256, 256)
    views = viewset(task).perceive(img)
    assert len(views) == task.n_views
    for v in views:
        assert v.shape == img.shape
        assert float(v.min()) >= 0.0 and float(v.max()) <= 1.0


def test_va_identity_slot_is_the_image_itself() -> None:
    torch.manual_seed(SEED)
    img = torch.rand(3, 64, 64)
    for task in VA_TASKS:
        torch.testing.assert_close(viewset(task).perceive(img)[0], img)


def test_negate_is_seen_as_the_photographic_negative() -> None:
    torch.manual_seed(SEED)
    img = torch.rand(3, 64, 64)
    torch.testing.assert_close(viewset(get_task("negate")).perceive(img)[1], 1.0 - img)


def test_rotation_view_matches_a_plain_rotation() -> None:
    torch.manual_seed(SEED)
    img = torch.rand(3, 64, 64)
    seen = viewset(get_task("rotate_cw")).perceive(img)[1]
    torch.testing.assert_close(seen, torch.rot90(img, k=-1, dims=(1, 2)))


def test_hybrid_far_view_is_the_documented_blur() -> None:
    torch.manual_seed(SEED)
    img = torch.rand(3, 256, 256)
    far, near = viewset(get_task("hybrid")).perceive(img)
    torch.testing.assert_close(far, perceive.far_view(img))
    assert near is img


def test_triple_hybrid_views_get_progressively_sharper() -> None:
    torch.manual_seed(SEED)
    img = torch.rand(3, 256, 256)
    far, mid, near = viewset(get_task("triple_hybrid")).perceive(img)
    assert float(far.std()) < float(mid.std()) < float(near.std())


def test_color_hybrid_gray_view_has_no_colour() -> None:
    torch.manual_seed(SEED)
    img = torch.rand(3, 64, 64)
    gray, _ = viewset(get_task("color_hybrid")).perceive(img)
    torch.testing.assert_close(gray[0], gray[1])
    torch.testing.assert_close(gray[1], gray[2])
    torch.testing.assert_close(gray[0], img.mean(0))


def test_motion_view_smears_along_the_diagonal_only() -> None:
    img = torch.zeros(3, 64, 64)
    img[:, 32, 32] = 1.0
    moving, _ = viewset(get_task("motion_hybrid")).perceive(img)
    lit = (moving[0] > 0).nonzero()
    assert len(lit) == 7  # MOTION_SIZE at the 64 px stage
    assert all(int(y) == int(x) for y, x in lit)  # a diagonal line


# -- randomised views ---------------------------------------------------------


def test_randomised_views_depend_only_on_the_seed() -> None:
    task = get_task("patch_permute")
    a = ViewSet.build(task, seed=3).state()["patch_permute"]
    b = ViewSet.build(task, seed=3).state()["patch_permute"]
    c = ViewSet.build(task, seed=4).state()["patch_permute"]
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_randomised_views_do_not_disturb_the_global_rng() -> None:
    torch.manual_seed(SEED)
    before = torch.rand(4)
    torch.manual_seed(SEED)
    ViewSet.build(get_task("pixel_permute"), seed=7)
    after = torch.rand(4)
    torch.testing.assert_close(before, after)


def test_view_state_round_trips_through_load_state() -> None:
    task = get_task("patch_permute")
    torch.manual_seed(SEED)
    img = torch.rand(3, 64, 64)
    source = ViewSet.build(task, seed=3)
    other = ViewSet.build(task, seed=4)
    assert not torch.equal(source.perceive(img)[1], other.perceive(img)[1])
    other.load_state(source.state())
    torch.testing.assert_close(source.perceive(img)[1], other.perceive(img)[1])


def test_deterministic_tasks_have_no_state_to_save() -> None:
    for task in TASKS.values():
        if not task.randomised:
            assert viewset(task).state() == {}, task.name
