"""Transition clips for a candidate, made with the upstream `animate.py`.

The still views a run persists show each reading of an illusion; the clip
shows the *transition* between them, which is how the papers present their
results. `animate_two_view` in the vendored checkout draws the identity frame
with its prompt, eases into the transformed frame (`view.make_frame(im, t)`),
draws that prompt, and boomerangs back. Motion hybrids use the upstream
motion-blur variant, as the upstream CLI does.

This is a figure step in the sense of `.claude/rules/coding.md`: the only
inputs are the persisted `sample_256.png`, the task's view objects (rebuilt
the way the run built them, permutations reloaded) and the prompts. Nothing is
generated or scored here.

One clip per view other than the task's *plain* view (`task.sr_slot`: the
identity view of a Visual Anagram, the near / colour / still reading of a
Factorized Diffusion hybrid), written to `anim_<slot>.mp4` beside the stills.
Views the checkout cannot animate (the triple hybrid has no `make_frame`)
raise `UnsupportedView` so a caller can say so rather than fail silently.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from visual_anagrams.animate import animate_two_view, animate_two_view_motion_blur
from visual_anagrams.views.view_base import BaseView
from visual_anagrams.views.view_motion import MotionBlurResView, MotionBlurView
from visual_anagrams.views.view_permute import PermuteView

from ava.image.tasks import IllusionTask, get_task
from ava.image.views import ViewSet
from ava.spec import CandidateSpec

ANIMATION_PREFIX = "anim_"
ANIMATION_SUFFIX = ".mp4"
FRAME_SCALE = 1.5  # the upstream CLI: frame_size = 1.5 * im_size


class UnsupportedView(Exception):
    """The checkout has no `make_frame` for this view; there is no clip to make."""


@dataclass(frozen=True)
class Timing:
    """Frame counts at 30 fps. The defaults are the upstream CLI's."""

    hold: int = 120
    text_fade: int = 10
    transition: int = 45
    motion_hold: int = 60
    motion_transition: int = 2000

    def __post_init__(self) -> None:
        # The upstream motion-blur animator averages a sliding window whose
        # width is derived from transition / 20 in steps of 10 frames; for any
        # count that is not a multiple of 200 the first window can be empty
        # and torch.stack fails. The upstream CLI uses 2000.
        if self.motion_transition <= 0 or self.motion_transition % 200:
            raise ValueError(
                f"motion_transition must be a positive multiple of 200, "
                f"got {self.motion_transition}"
            )


DEFAULT_TIMING = Timing()


def animation_path(out_dir: Path, slot: str) -> Path:
    return out_dir / f"{ANIMATION_PREFIX}{slot}{ANIMATION_SUFFIX}"


def slot_of_animation(path: Path) -> str:
    """Inverse of `animation_path` for a file name."""
    name = path.name
    if not (name.startswith(ANIMATION_PREFIX) and name.endswith(ANIMATION_SUFFIX)):
        raise ValueError(f"not an animation file name: {name}")
    return name[len(ANIMATION_PREFIX) : -len(ANIMATION_SUFFIX)]


def can_animate(view: Any) -> bool:
    """True if the view overrides `make_frame` with an implementation.

    `BaseView.make_frame` and `PermuteView.make_frame` both raise; the patch
    and pixel permutations override it, the triple hybrid views do not.
    """
    impl = type(view).make_frame
    return impl is not BaseView.make_frame and impl is not PermuteView.make_frame


def animatable_slots(task: IllusionTask, views: Sequence[Any]) -> list[int]:
    """Indices of the slots a clip is made for, or `UnsupportedView`."""
    out = []
    for k, view in enumerate(views):
        if k == task.sr_slot:
            continue
        if not can_animate(view):
            raise UnsupportedView(
                f"{task.name}: view {task.view_names[k]!r} has no make_frame"
            )
        out.append(k)
    return out


def animate_candidate(
    image: Image.Image,
    viewset: ViewSet,
    prompts: Sequence[str],
    out_dir: Path,
    *,
    timing: Timing = DEFAULT_TIMING,
    force: bool = False,
) -> list[Path]:
    """Write one clip per non-plain view; return the paths, existing ones included.

    `prompts` are the full prompts (style applied) in slot order; each end of a
    clip is captioned with the prompt that reads there. A clip is written to a
    temporary name and renamed, so a viewer never serves a half-written file.
    """
    task = viewset.task
    if len(prompts) != len(viewset.views):
        raise ValueError(f"{len(prompts)} prompts for {len(viewset.views)} views")
    plain = task.sr_slot
    im = image.convert("RGB")
    im_size = im.size[0]
    frame_size = int(im_size * FRAME_SCALE)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for k in animatable_slots(task, viewset.views):
        path = animation_path(out_dir, task.slots[k].name)
        if path.exists() and not force:
            paths.append(path)
            continue
        view = viewset.views[k]
        tmp = path.with_name(path.name + ".part.mp4")
        if isinstance(view, MotionBlurView | MotionBlurResView):
            animate_two_view_motion_blur(
                im,
                view,
                prompts[plain],
                prompts[k],
                save_video_path=str(tmp),
                hold_duration=timing.motion_hold,
                text_fade_duration=timing.text_fade,
                transition_duration=timing.motion_transition,
                im_size=im_size,
                frame_size=frame_size,
            )
        else:
            animate_two_view(
                im,
                view,
                prompts[plain],
                prompts[k],
                save_video_path=str(tmp),
                hold_duration=timing.hold,
                text_fade_duration=timing.text_fade,
                transition_duration=timing.transition,
                im_size=im_size,
                frame_size=frame_size,
            )
        os.replace(tmp, path)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Working from a run directory
# ---------------------------------------------------------------------------


def viewsets_for_run(root: Path) -> dict[str, ViewSet]:
    """Rebuild the run's view objects from `config.yaml` and `views/<task>.pt`.

    Same construction as the loop: the recorded `view_seed`, and a randomised
    view's persisted permutation instead of a fresh draw. A run that never
    persisted one (it was not randomised, or it predates the file) gets the
    seeded construction, which is what the loop would have drawn.
    """
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    seed = int(config.get("view_seed", 0))
    out: dict[str, ViewSet] = {}
    for name in config.get("tasks", []):
        viewset = ViewSet.build(get_task(name), seed=seed)
        state_path = root / "views" / f"{name}.pt"
        if state_path.exists():
            viewset.load_state(torch.load(state_path))
        out[name] = viewset
    return out


def candidate_dir(root: Path, row: dict[str, Any]) -> Path:
    return root / f"round_{int(row['round']):03d}" / str(row["uid"])


def animate_row(
    root: Path,
    row: dict[str, Any],
    viewsets: dict[str, ViewSet],
    *,
    timing: Timing = DEFAULT_TIMING,
    force: bool = False,
) -> list[Path]:
    """Clips for one `scores.jsonl` row, from its persisted sample image."""
    task = str(row["task"])
    if task not in viewsets:
        raise KeyError(f"no view set for task {task!r}")
    out_dir = candidate_dir(root, row)
    sample = out_dir / Path(str(row["image_path"])).name
    if not sample.exists():
        raise FileNotFoundError(sample)
    spec = CandidateSpec(
        task=task, prompts=tuple(row["prompts"]), style=str(row.get("style", ""))
    )
    with Image.open(sample) as image:
        return animate_candidate(
            image,
            viewsets[task],
            spec.full_prompts,
            out_dir,
            timing=timing,
            force=force,
        )
