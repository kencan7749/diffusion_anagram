"""The illusion tasks: which views an image must satisfy, and how to look at it.

One registry entry per example type in the two papers, built on the upstream
view classes in dev/visual_anagrams. The two papers are two different
mechanisms and the entry records which one applies:

  VA  Visual Anagrams (Geng et al. 2024). The view transforms the noisy image
      before denoising, so it must be orthogonal (a permutation, or negation)
      to leave the noise Gaussian. Noise estimates are averaged (`mean`).
      The viewer sees `view(image)`, so the judge looks at exactly that.
  FD  Factorized Diffusion (Geng et al. 2025). The view is the identity on the
      image and `inverse_view` extracts one linear component of the noise
      estimate; the components sum to the identity, so the estimates are
      summed (`sum`). The viewer sees a blurred / grayscale / motion-blurred
      picture, which `ava.image.perceive` defines.

This module is pure data so the search layer can read slot names and roles
without importing torch; `ava.image.views` builds the actual view objects.
The audio track registers its own task here (`ava.audio.tasks`), because the
search layer only needs a task's slots and roles and does not care whether the
views are spatial or temporal.

A slot is one prompt position. Its `role` names the bandit arm family a word
in that slot contributes evidence to. FD slots are asymmetric (a low-frequency
subject is a different job from a high-frequency one), so each slot has its own
role. VA slots are symmetric -- swapping the two prompts of a flip illusion
gives the same illusion flipped -- so every VA slot shares the role `subject`.

Not registered, on purpose: the ambigram (rotate_180 with a text template;
CLIP cannot read cursive reliably enough to judge it), the spatial mask
decomposition and the scaling decomposition (neither is an illusion a judge
can score), and the latent-space orthogonal transform (needs Stable Diffusion).
"""

from __future__ import annotations

from dataclasses import dataclass

VA = "VA"
FD = "FD"
AUDIO = "audio"  # the time-reversal anagram; views are defined by ava.audio

SUBJECT = "subject"
STYLE = "style"


@dataclass(frozen=True)
class Slot:
    name: str  # what the view is called in reports and file names
    role: str  # bandit arm family for words in this slot


@dataclass(frozen=True)
class IllusionTask:
    name: str
    paper: str
    slots: tuple[Slot, ...]
    view_names: tuple[str, ...]  # upstream VIEW_MAP keys, in slot order
    reduction: str  # 'mean' (VA) or 'sum' (FD)
    citation: str
    # Which slot's prompt conditions the x4 upscaler, which knows nothing about
    # views: the identity view for VA, the finest component for FD (App. A.4).
    sr_slot: int = 0
    # Inverse problems pin this slot to a reference image; its prompt is only
    # used by the judge.
    ref_slot: int | None = None
    # True when building the views draws random numbers (patch permutations),
    # so a run must fix a seed and persist what it drew.
    randomised: bool = False

    def __post_init__(self) -> None:
        if len(self.slots) != len(self.view_names):
            raise ValueError(
                f"{self.name}: {len(self.slots)} slots, {len(self.view_names)} views"
            )
        if self.reduction not in ("mean", "sum"):
            raise ValueError(f"{self.name}: reduction must be mean or sum")
        if self.paper not in (VA, FD, AUDIO):
            raise ValueError(f"{self.name}: paper must be VA, FD or audio")

    @property
    def n_views(self) -> int:
        return len(self.slots)

    @property
    def slot_names(self) -> list[str]:
        return [s.name for s in self.slots]

    @property
    def roles(self) -> list[str]:
        return [s.role for s in self.slots]

    @property
    def symmetric(self) -> bool:
        """True when every slot shares one role, so prompt order carries no meaning."""
        return len(set(self.roles)) == 1

    def searched_slots(self) -> list[int]:
        """Slot indices the proposer fills; the reference slot is fixed."""
        return [i for i in range(self.n_views) if i != self.ref_slot]


def _va(name: str, view: str, citation: str, slot: str | None = None) -> IllusionTask:
    return IllusionTask(
        name=name,
        paper=VA,
        slots=(Slot("identity", SUBJECT), Slot(slot or view, SUBJECT)),
        view_names=("identity", view),
        reduction="mean",
        citation=citation,
        randomised=view in ("patch_permute", "pixel_permute"),
    )


TASKS: dict[str, IllusionTask] = {
    t.name: t
    for t in (
        # ---- Visual Anagrams -------------------------------------------
        _va("flip", "flip", "VA Fig. 1, 2, 8; readme flip.campfire.man"),
        _va(
            "rotate_cw",
            "rotate_cw",
            "VA Fig. 8, 12, 16; readme rotate_cw.village.horse",
        ),
        _va("rotate_ccw", "rotate_ccw", "readme rotate_ccw.village.horse"),
        _va("rotate_180", "rotate_180", "VA Sec. 3.4 (180 degree rotation)"),
        _va("skew", "skew", "VA Fig. 1, 8; readme skew.tudor.skull"),
        _va(
            "jigsaw",
            "jigsaw",
            "VA Fig. 1, 8, 12, 17, App. F; readme jigsaw.houseplants.marilyn",
        ),
        _va(
            "inner_circle",
            "inner_circle",
            "VA Fig. 1, 17; readme inner.einstein.marilyn",
        ),
        _va(
            "negate",
            "negate",
            "VA Fig. 1, 12, 16, App. A; readme negate.landscape.houseplants",
        ),
        _va(
            "patch_permute",
            "patch_permute",
            "VA Fig. 6, 17 (8x8); readme patch.lemur.kangaroo",
        ),
        _va(
            "pixel_permute",
            "pixel_permute",
            "VA Fig. 6, 17 (64x64); readme pixel.duck.rabbit",
        ),
        _va(
            "square_hinge",
            "square_hinge",
            "upstream readme hinge.duck.rabbit (not in the paper)",
        ),
        IllusionTask(
            name="three_view",
            paper=VA,
            slots=(
                Slot("identity", SUBJECT),
                Slot("rotate_cw", SUBJECT),
                Slot("rotate_ccw", SUBJECT),
            ),
            view_names=("identity", "rotate_cw", "rotate_ccw"),
            reduction="mean",
            citation="VA Fig. 1, 17 (Three Views); readme threeview",
        ),
        IllusionTask(
            name="four_view",
            paper=VA,
            slots=(
                Slot("identity", SUBJECT),
                Slot("rotate_cw", SUBJECT),
                Slot("rotate_180", SUBJECT),
                Slot("rotate_ccw", SUBJECT),
            ),
            view_names=("identity", "rotate_cw", "rotate_180", "rotate_ccw"),
            reduction="mean",
            citation="VA Fig. 1, 11 (Four Views: 0/90/180/270)",
        ),
        # ---- Factorized Diffusion ----------------------------------------
        IllusionTask(
            name="hybrid",
            paper=FD,
            slots=(Slot("low", "low"), Slot("high", "high")),
            view_names=("low_pass", "high_pass"),
            reduction="sum",
            citation="FD Sec. 3.4, Fig. 1, 3, 9, 16, 18; FD readme hybrid",
            sr_slot=1,
        ),
        IllusionTask(
            name="triple_hybrid",
            paper=FD,
            slots=(Slot("low", "low"), Slot("mid", "mid"), Slot("high", "high")),
            view_names=("triple_low_pass", "triple_medium_pass", "triple_high_pass"),
            reduction="sum",
            citation="FD Fig. 1, 2, 14, App. A.3; FD readme triple_hybrid",
            sr_slot=2,
        ),
        IllusionTask(
            name="color_hybrid",
            paper=FD,
            slots=(Slot("gray", "gray"), Slot("color", "color")),
            view_names=("grayscale", "color"),
            reduction="sum",
            citation="FD Fig. 1, 5, 9, 17, 18; readme_factorized_diffusion color",
            sr_slot=1,
        ),
        IllusionTask(
            name="motion_hybrid",
            paper=FD,
            slots=(Slot("moving", "moving"), Slot("still", "still")),
            view_names=("motion", "motion_res"),
            reduction="sum",
            citation="FD Fig. 1, 6, 9, 15, 18; readme_factorized_diffusion motion",
            sr_slot=1,
        ),
        IllusionTask(
            name="inverse_hybrid",
            paper=FD,
            slots=(Slot("low", "low"), Slot("high", "high")),
            view_names=("low_pass", "high_pass"),
            reduction="sum",
            citation="FD Sec. 3.5, Fig. 1, 8, 19-20; FD readme inverse",
            sr_slot=1,
            ref_slot=0,
        ),
    )
}


# Tasks from other tracks (`ava.audio.tasks`). Kept apart from TASKS, which
# stays the list of the two papers' example types that the image views, the
# seed vocabularies and their tests enumerate.
REGISTERED: dict[str, IllusionTask] = {}


def register_task(task: IllusionTask) -> None:
    """Make a task from another track visible to `get_task`. Idempotent."""
    existing = TASKS.get(task.name) or REGISTERED.get(task.name)
    if existing is None:
        REGISTERED[task.name] = task
    elif existing != task:
        raise ValueError(f"a different task named {task.name!r} is already registered")


def all_tasks() -> dict[str, IllusionTask]:
    """Every task `get_task` can resolve: the papers' plus the registered ones."""
    return {**TASKS, **REGISTERED}


def get_task(name: str) -> IllusionTask:
    task = TASKS.get(name) or REGISTERED.get(name)
    if task is None:
        raise KeyError(f"unknown task {name!r}; known: {sorted(all_tasks())}")
    return task
