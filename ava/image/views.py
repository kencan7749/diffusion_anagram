"""Constructing the upstream view objects for a task, and looking through them.

`ava.image.tasks` says which views a task uses; this module builds them and
pairs them with the perceptual views the judge scores. Kept apart from the
registry because building views needs torch and the vendored checkout, and the
search layer only needs the registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from visual_anagrams.views import VIEW_MAP
from visual_anagrams.views.permutations import get_inv_perm

from ava.image import perceive
from ava.image.tasks import VA, IllusionTask
from ava.spec import (
    JIGSAW_SEED,
    KERNEL_SIZE,
    MOTION_SIZE,
    PATCH_GRID,
    PIXEL_GRID,
    SIGMA,
    SKEW_FACTOR,
    TRIPLE_KERNEL_SIZE,
    TRIPLE_SIGMA_1,
    TRIPLE_SIGMA_2,
)

# Constructor arguments for the upstream views, in one place. `get_views` in
# visual_anagrams/views/__init__.py forwards a single CLI argument to five view
# types; this is the equivalent for every view the registry uses.
_VIEW_ARGS: dict[str, tuple[Any, ...]] = {
    "low_pass": (SIGMA, KERNEL_SIZE),
    "high_pass": (SIGMA, KERNEL_SIZE),
    "triple_low_pass": (TRIPLE_SIGMA_1, TRIPLE_SIGMA_2, TRIPLE_KERNEL_SIZE),
    "triple_medium_pass": (TRIPLE_SIGMA_1, TRIPLE_SIGMA_2, TRIPLE_KERNEL_SIZE),
    "triple_high_pass": (TRIPLE_SIGMA_1, TRIPLE_SIGMA_2, TRIPLE_KERNEL_SIZE),
    "motion": (MOTION_SIZE,),
    "motion_res": (MOTION_SIZE,),
    "skew": (SKEW_FACTOR,),
    "patch_permute": (PATCH_GRID,),
    "pixel_permute": (PIXEL_GRID,),
    "jigsaw": (JIGSAW_SEED,),
}


def build_views(task: IllusionTask, seed: int = 0) -> list[Any]:
    """Construct the upstream view objects for a task, in slot order.

    Randomised views draw their permutation from torch's global RNG at
    construction, so they are built under a forked RNG seeded with `seed`.
    Deterministic views ignore the seed.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return [VIEW_MAP[name](*_VIEW_ARGS.get(name, ())) for name in task.view_names]


@dataclass
class ViewSet:
    """A task together with its constructed views: what a run actually uses.

    The generator needs the view objects (they implement the decomposition);
    the judge needs them too for VA tasks, because the viewer sees `view(image)`
    and for a random permutation that is only defined by the object that was
    built. Building once per run and passing this around keeps the two sides
    looking at the same thing.
    """

    task: IllusionTask
    views: list[Any]
    seed: int

    @classmethod
    def build(cls, task: IllusionTask, seed: int = 0) -> ViewSet:
        return cls(task=task, views=build_views(task, seed), seed=seed)

    # -- the perceptual views ------------------------------------------------

    def perceive(self, img: torch.Tensor) -> list[torch.Tensor]:
        """The N images a person sees, in slot order. `img` is (C, H, W) in [0, 1]."""
        if self.task.paper == VA:
            return [perceive.va_transform(v, img) for v in self.views]
        name = self.task.name
        if name in ("hybrid", "inverse_hybrid"):
            return [perceive.blur(img, SIGMA, KERNEL_SIZE), perceive.near_view(img)]
        if name == "triple_hybrid":
            far_sigma = perceive.compound_sigma(TRIPLE_SIGMA_1, TRIPLE_SIGMA_2)
            return [
                perceive.blur(img, far_sigma, TRIPLE_KERNEL_SIZE),
                perceive.blur(img, TRIPLE_SIGMA_1, TRIPLE_KERNEL_SIZE),
                perceive.near_view(img),
            ]
        if name == "color_hybrid":
            return [perceive.grayscale(img), perceive.near_view(img)]
        if name == "motion_hybrid":
            return [perceive.motion_blur(img, MOTION_SIZE), perceive.near_view(img)]
        raise NotImplementedError(f"no perceptual views defined for task {name!r}")

    # -- persistence of what was drawn ---------------------------------------

    def state(self) -> dict[str, torch.Tensor]:
        """Random permutations keyed by slot name; empty for deterministic tasks."""
        out: dict[str, torch.Tensor] = {}
        for slot, view in zip(self.task.slots, self.views, strict=True):
            if hasattr(view, "perm") and hasattr(view, "perm_inv"):
                out[slot.name] = view.perm.clone()
        return out

    def load_state(self, state: dict[str, torch.Tensor]) -> None:
        """Restore permutations saved by `state()`, so a resumed run sees the same."""
        for slot, view in zip(self.task.slots, self.views, strict=True):
            if slot.name in state:
                perm = state[slot.name].clone()
                view.perm = perm
                view.perm_inv = get_inv_perm(perm)
