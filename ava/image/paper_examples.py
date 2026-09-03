"""Prompt pairs quoted from the two papers, as seed vocabulary with citations.

Every entry is a figure the authors published, split into the style they used
and the subjects in slot order for the matching task in `ava.image.tasks`.
Recombining `style` with each subject through `ava.spec.apply_style` gives the
paper's prompt back verbatim; that is what `docs/paper_examples.md` lists and
what a reader can check against the PDF.

These enter the vocabulary as `source='paper'` arms with a uniform Beta(1, 1)
prior and the figure as citation. The prior is uniform on purpose: the papers
show that these pairs *can* work, hand-picked from many samples, which says
little about their yield. The data decides.

Sources:
  VA  Geng, Park, Owens. Visual Anagrams. CVPR 2024 (arXiv 2311.17919v2)
  FD  Geng, Park, Owens. Factorized Diffusion. ECCV 2024 (arXiv 2404.11615v2)

Slot orders (see ava.image.tasks): VA tasks are (identity, transformed) and
symmetric; hybrid is (low, high), so a figure's "H / L" caption is reversed
here; triple is (low, mid, high); color is (gray, color); motion is (moving,
still); inverse_hybrid seeds only its high slot, the low slot being a reference
image.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from ava.vocab import PAPER_SOURCE, STYLE, SUBJECT, UNIFORM_PRIOR, add_arm


@dataclass(frozen=True)
class PaperExample:
    task: str
    style: str  # prefix, or template containing "{}"
    prompts: tuple[str, ...]  # subjects, in the task's slot order
    citation: str


def _many(
    task: str, style: str, citation: str, *pairs: tuple[str, ...]
) -> list[PaperExample]:
    return [PaperExample(task, style, p, citation) for p in pairs]


PAPER_EXAMPLES: tuple[PaperExample, ...] = tuple(
    # ---------------------------------------------------------------- VA ----
    # Fig. 1 flips
    _many(
        "flip",
        "a painting of",
        "VA Fig. 1",
        ("a red panda", "kitchenware"),
        ("a sloth", "vases"),
        ("a turtle", "wine and cheese"),
    )
    + _many("flip", "a lithograph of", "VA Fig. 1", ("a duck", "a fish"))
    + _many(
        "flip",
        "an oil painting of",
        "VA Fig. 1, 2, 8",
        ("people at a campfire", "an old man"),
    )
    + _many("flip", "an oil painting of", "VA Fig. 1", ("a bird", "a ship"))
    + _many("flip", "a drawing of", "VA Fig. 1", ("a penguin", "a giraffe"))
    + _many("flip", "a photo of", "VA Fig. 1", ("an old woman", "a young lady"))
    # Fig. 1 jigsaw / inner rotation / skew / negate
    + _many(
        "jigsaw", "an oil painting of", "VA Fig. 1, 8", ("a fruit bowl", "a monkey")
    )
    + _many("jigsaw", "a watercolor of", "VA Fig. 1", ("a kitten", "a puppy"))
    + _many(
        "jigsaw", "a painting of", "VA Fig. 1, 8", ("houseplants", "marilyn monroe")
    )
    + _many(
        "inner_circle",
        "a pop art of",
        "VA Fig. 1",
        ("albert einstein", "marilyn monroe"),
    )
    + _many(
        "skew", "an oil painting of", "VA Fig. 1, 8", ("a tudor portrait", "a skull")
    )
    + _many("skew", "an oil painting of", "VA Fig. 1", ("a soldier", "a houseplant"))
    + _many(
        "negate",
        "a lithograph of",
        "VA Fig. 1",
        ("a teddy bear", "a rabbit"),
        ("a landscape", "houseplants"),
    )
    + _many("negate", "a photo of", "VA Fig. 1", ("a man", "a woman"))
    # Fig. 1 three and four views
    + _many(
        "three_view",
        "a painting of",
        "VA Fig. 1",
        ("a teddy bear", "a rabbit", "waterfalls"),
        ("einstein", "elvis", "houseplants"),
    )
    + _many(
        "four_view",
        "an oil painting of",
        "VA Fig. 1, 11",
        ("a rabbit", "a giraffe", "a teddy bear", "a bird"),
    )
    # Fig. 5 / 10 flips (baseline comparisons)
    + _many(
        "flip",
        "a painting of",
        "VA Fig. 5",
        ("a truck", "a deer"),
        ("a horse", "a bird"),
        ("a dog", "an airplane"),
    )
    + _many("flip", "an ink drawing of", "VA Fig. 5", ("a house", "a castle"))
    + _many("flip", "a watercolor painting of", "VA Fig. 5", ("an owl", "a dog"))
    + _many("flip", "a street art of", "VA Fig. 5", ("a rabbit", "a violin"))
    + _many(
        "flip",
        "a painting of",
        "VA Fig. 10",
        ("a dog", "a cat"),
        ("a frog", "a deer"),
        ("a truck", "a ship"),
    )
    + _many("flip", "a pop art of", "VA Fig. 10", ("a giraffe", "a bird"))
    + _many("flip", "a sketch of", "VA Fig. 10", ("a cup", "a frog"))
    + _many("flip", "an oil painting of", "VA Fig. 10", ("a cat", "a mouse"))
    # Fig. 6 / 17 permutations
    + _many(
        "patch_permute", "a pencil sketch of", "VA Fig. 6", ("a lemur", "a kangaroo")
    )
    + _many("patch_permute", "a watercolor of", "VA Fig. 6", ("a rabbit", "a duck"))
    + _many(
        "patch_permute", "a pencil sketch of", "VA Fig. 17", ("a zebra", "a motorcycle")
    )
    + _many(
        "patch_permute",
        "an oil painting of",
        "VA Fig. 6, 17",
        ("a young man", "an old man"),
    )
    + _many("pixel_permute", "a mosaic of", "VA Fig. 6", ("a duck", "a rabbit"))
    + _many("pixel_permute", "a photo of", "VA Fig. 17", ("a soccer ball", "a panda"))
    # Fig. 8 / 12 / 16 rotations (90 degrees; the paper does not say which way)
    + _many(
        "rotate_cw",
        "an oil painting of",
        "VA Fig. 8",
        ("a snowy mountain village", "a horse"),
    )
    + _many(
        "rotate_cw",
        "a watercolor painting of",
        "VA Fig. 12",
        ("a village in the mountains", "a ship"),
    )
    + _many("rotate_cw", "an oil painting of", "VA Fig. 12", ("a theater", "a library"))
    + _many(
        "rotate_cw",
        "a lithograph of",
        "VA Fig. 16",
        ("a village in the mountains", "a ship"),
        ("a theater", "a ship"),
    )
    # Fig. 8 / 12 / 17 jigsaw
    + _many("jigsaw", "a painting of", "VA Fig. 8", ("wine and cheese", "a turtle"))
    + _many(
        "jigsaw",
        "an oil painting of",
        "VA Fig. 12",
        ("a woman staring out a window", "abraham lincoln"),
        ("a fruit bowl", "albert einstein"),
        ("flower arrangements", "a quokka"),
    )
    + _many(
        "jigsaw",
        "a painting of",
        "VA Fig. 17",
        ("a kitchen", "a camel"),
        ("a dining table", "a polar bear"),
        ("houseplants", "an octopus"),
        ("flower arrangements", "a sloth"),
    )
    # Fig. 12 / 16 negate
    + _many("negate", "a blockprint of", "VA Fig. 12", ("a red panda", "elvis presley"))
    + _many("negate", "a lithograph of", "VA Fig. 12, 16", ("waterfalls", "a table"))
    + _many(
        "negate",
        "an ink drawing of",
        "VA Fig. 16",
        ("a red panda", "elvis"),
        ("waterfalls", "wine and cheese"),
    )
    # Fig. 12 / 16 flips
    + _many(
        "flip",
        "an oil painting of",
        "VA Fig. 12",
        ("a museum", "a quokka"),
        ("a kitchen", "a quokka"),
    )
    + _many("flip", "a photo of", "VA Fig. 16", ("a wedding dress", "an old woman"))
    + _many(
        "flip",
        "an oil painting of",
        "VA Fig. 16",
        ("albert einstein", "elvis"),
        ("a red panda", "a teddy bear"),
        ("a kitchen", "a botanical garden"),
    )
    + _many(
        "flip",
        "a painting of",
        "VA Fig. 16",
        ("a museum", "a camel"),
        ("a sculpture garden", "a deer"),
        ("a still life of trophies and awards", "a quokka"),
        ("wine and cheese", "a pig"),
    )
    # Fig. 13 / 14: one flipped subject against many upright ones
    + _many(
        "flip",
        "a painting of",
        "VA Fig. 13",
        *[
            (u, "albert einstein")
            for u in (
                "sunflowers",
                "a vampire",
                "waterfalls",
                "a red panda",
                "a quokka",
                "a deer",
                "elvis presley",
                "a garden",
                "a horse",
                "marilyn monroe",
                "a penguin",
                "students in a classroom",
            )
        ],
    )
    + _many(
        "flip",
        "a painting of",
        "VA Fig. 14",
        *[
            (u, "abraham lincoln")
            for u in (
                "sunflowers",
                "a turkey",
                "waterfalls",
                "a red panda",
                "a horse",
                "a deer",
            )
        ],
        *[
            (u, "michael jackson")
            for u in (
                "elvis presley",
                "albert einstein",
                "a vampire",
                "a quokka",
                "jeans",
                "a garden",
            )
        ],
    )
    # Fig. 15 CIFAR flips
    + _many(
        "flip",
        "a painting of",
        "VA Fig. 15",
        ("a ship", "a car"),
        ("a horse", "a cat"),
        ("a ship", "an airplane"),
        ("a bird", "a car"),
    )
    # Fig. 17 three views, inner rotation
    + _many(
        "three_view",
        "a painting of",
        "VA Fig. 17",
        ("ancient ruins", "a red panda", "a teddy bear"),
        ("a coral reef", "a quokka", "elvis"),
        ("waterfalls", "a red panda", "elvis"),
    )
    + _many(
        "inner_circle", "an oil painting of", "VA Fig. 17", ("an old man", "a chair")
    )
    # ---------------------------------------------------------------- FD ----
    # Fig. 1 hybrids, written (low, high)
    + _many(
        "hybrid",
        "a photo of",
        "FD Fig. 1",
        ("marilyn monroe", "houseplants"),
        ("an old man", "a rabbit"),
        ("a john lennon", "new york city"),
        ("a yin yang", "rome"),
        ("a teddy bear", "mountains"),
    )
    + _many(
        "hybrid",
        "a lithograph of",
        "FD Fig. 1",
        ("a pig", "waterfalls"),
        ("a panda", "flower arrangements"),
        ("a deer", "a ski slope in the alps"),
    )
    # Fig. 1 / 2 / 14 triple hybrids, written (low, mid, high)
    + _many(
        "triple_hybrid",
        "a photo of",
        "FD Fig. 1, 2",
        ("a yin yang", "a skull", "waterfalls"),
    )
    + _many(
        "triple_hybrid",
        "a photo of",
        "FD Fig. 1, 14",
        ("a pyramid", "a dog", "flower arrangements"),
        ("a yin yang", "a rabbit", "flower arrangements"),
        ("the eiffel tower", "a quokka", "houseplants"),
        ("a diamond", "an old man", "a fish"),
        ("a pyramid", "a rabbit", "flower arrangements"),
    )
    # Fig. 1 / 5 / 17 colour hybrids, written (gray, color)
    + _many(
        "color_hybrid",
        "a watercolor of",
        "FD Fig. 1",
        ("houseplants", "the statue of liberty"),
    )
    + _many(
        "color_hybrid",
        "a painting of",
        "FD Fig. 1",
        ("a barn", "a bumblebee"),
        ("a flower arrangement", "a teddy bear"),
    )
    + _many(
        "color_hybrid",
        "a painting of",
        "FD Fig. 5",
        ("a landscape", "a tiger"),
        ("a volcano", "a rabbit"),
        ("a dining table", "a polar bear"),
    )
    + _many(
        "color_hybrid",
        "oil painting style, {}",
        "FD Fig. 5",
        ("the grand canyon", "a bird"),
    )
    + _many(
        "color_hybrid",
        "a photo of",
        "FD Fig. 5",
        ("a bird", "a frog"),
        ("the grand canyon", "a heart"),
    )
    + _many(
        "color_hybrid",
        "a photo of",
        "FD Fig. 17",
        ("a car", "a ship"),
        ("a ship", "a frog"),
    )
    + _many(
        "color_hybrid",
        "a watercolor of",
        "FD Fig. 17",
        ("ancient ruins", "an old man"),
        ("flower arrangements", "an old woman"),
        ("a desert", "a camel"),
        ("flower arrangements", "a duck"),
        ("a landscape", "a pig"),
        ("a rainforest", "a crocodile"),
        ("waterfalls", "a skull"),
    )
    + _many(
        "color_hybrid",
        "a painting of",
        "FD Fig. 17",
        ("a theater", "a duck"),
        ("city skyscrapers", "a rabbit"),
    )
    + _many(
        "color_hybrid",
        "oil painting style, {}",
        "FD Fig. 17",
        ("mountains", "a rabbit"),
        ("the grand canyon", "john lennon"),
        ("a flower arrangement", "a bird"),
        ("a volcano", "a duck"),
        ("mountains", "a tiger"),
    )
    + _many(
        "color_hybrid",
        "an oil painting of",
        "FD Fig. 17",
        ("an old man", "students in a classroom"),
    )
    + _many(
        "color_hybrid",
        "a lithograph of",
        "FD Fig. 17",
        ("the grand canyon", "a bumblebee"),
    )
    # Fig. 1 / 6 / 15 motion hybrids, written (moving, still)
    + _many(
        "motion_hybrid",
        "a photo of",
        "FD Fig. 1",
        ("a car", "a canyon"),
        ("a horse", "a sports stadium"),
    )
    + _many("motion_hybrid", "a painting of", "FD Fig. 1", ("a dog", "a beehive"))
    + _many(
        "motion_hybrid",
        "a photo of",
        "FD Fig. 6",
        ("sunflowers", "a van gogh portrait"),
        ("a rabbit", "a duck"),
        ("abraham lincoln", "a mountain"),
        ("a panda", "a canyon"),
        ("a car", "a stadium"),
        ("a teddy bear", "new york city"),
    )
    + _many("motion_hybrid", "a watercolor of", "FD Fig. 15", ("a dog", "a bazaar"))
    + _many(
        "motion_hybrid",
        "oil painting style, {}",
        "FD Fig. 15",
        ("a heart", "the grand canyon"),
        ("an old man", "a forest"),
    )
    + _many(
        "motion_hybrid",
        "a photo of",
        "FD Fig. 15",
        ("a bird", "waterfalls"),
        ("a teddy bear", "ancient ruins"),
        ("a skull", "an old man"),
    )
    # Fig. 9 / 16 / 18 hybrids, written (low, high)
    + _many("hybrid", "a photo of", "FD Fig. 9", ("an old man", "a rabbit"))
    + _many("hybrid", "a watercolor of", "FD Fig. 9", ("a panda", "mountains"))
    + _many(
        "hybrid", "a headshot of", "FD Fig. 3", ("albert einstein", "marilyn monroe")
    )
    + _many(
        "hybrid", "a photo of", "FD Fig. 3", ("a snowy mountain village", "a skull")
    )
    + _many(
        "hybrid",
        "oil painting style, {}",
        "FD Fig. 16",
        ("a bumblebee", "a bazaar"),
        ("abraham lincoln", "a flower arrangement"),
        ("a bird", "texture of feathers"),
        ("a panda", "the grand canyon"),
        ("john lennon", "the grand canyon"),
        ("a panda", "mountains"),
        ("a panda", "new york city"),
        ("an old man", "a bazaar"),
    )
    + _many(
        "hybrid",
        "a photo of",
        "FD Fig. 16",
        ("an old woman", "houseplants"),
        ("audrey hepburn", "an english breakfast"),
        ("an old woman", "a library"),
        ("gandhi", "a forest"),
        ("an old man", "texture of granite"),
        ("a panda", "a barn"),
        ("abraham lincoln", "a bazaar"),
        ("john lennon", "a flower arrangement"),
        ("gandhi", "a sunset"),
        ("an old man", "houseplants"),
        ("elvis", "the grand canyon"),
        ("an old woman", "flower arrangements"),
        ("a teddy bear", "the grand canyon"),
        ("abraham lincoln", "a flower arrangement"),
    )
    + _many(
        "hybrid",
        "a lithograph of",
        "FD Fig. 16",
        ("a skull", "waterfalls"),
        ("a quokka", "houseplants"),
        ("houseplants", "waterfalls"),
        ("a skull", "houseplants"),
    )
    + _many(
        "hybrid",
        "a watercolor of",
        "FD Fig. 16",
        ("king tut", "a sunset"),
        ("a panda", "a library"),
        ("a teddy bear", "new york city"),
        ("a bird", "a bazaar"),
    )
    + _many(
        "hybrid",
        "a photo of",
        "FD Fig. 18",
        ("a skull", "waterfalls"),
        ("a teddy bear", "a sunset"),
    )
    + _many(
        "hybrid",
        "lithograph style, {}",
        "FD Fig. 18",
        ("a bumblebee", "a flower arrangement"),
    )
    # Fig. 9 / 18 colour and motion
    + _many(
        "color_hybrid", "an oil painting of", "FD Fig. 9", ("waterfalls", "a tiger")
    )
    + _many(
        "color_hybrid",
        "a watercolor painting of",
        "FD Fig. 9",
        ("flower arrangements", "a duck"),
    )
    + _many(
        "motion_hybrid",
        "a photo of",
        "FD Fig. 9",
        ("a rabbit", "a duck"),
        ("a skull", "an old man"),
    )
    + _many(
        "color_hybrid",
        "oil painting style, {}",
        "FD Fig. 18",
        ("a library", "a bumblebee"),
    )
    + _many(
        "color_hybrid",
        "a watercolor of",
        "FD Fig. 18",
        ("a flower arrangement", "a bird"),
    )
    + _many(
        "color_hybrid",
        "an oil painting of",
        "FD Fig. 18",
        ("vases", "a duck"),
        ("a volcano", "a rabbit"),
    )
    + _many(
        "motion_hybrid",
        "oil painting style, {}",
        "FD Fig. 18",
        ("a car", "a sports stadium"),
        ("a heart", "the grand canyon"),
    )
    + _many(
        "motion_hybrid", "a watercolor of", "FD Fig. 18", ("a dog", "new york city")
    )
    + _many("motion_hybrid", "a photo of", "FD Fig. 18", ("a teddy bear", "a bazaar"))
)


# Inverse hybrids: the low slot is a reference image, so only the high-frequency
# prompt is a searchable word.
INVERSE_HIGH: tuple[tuple[str, str, str], ...] = (
    # (style, subject, citation)
    ("a photo of", "a cat", "FD Fig. 1"),
    ("an oil painting of", "a sunset", "FD Fig. 1"),
    ("a watercolor of", "Duomo di Milano", "FD Fig. 1"),
    ("a photo of", "a leopard", "FD Fig. 8"),
    ("a photo of", "a lightbulb", "FD Fig. 8"),
    ("a photo of", "waterfalls", "FD Fig. 8"),
)


def paper_arms() -> list[tuple[str, str, str, str]]:
    """Every (word, task, role, citation) the examples imply, deduplicated.

    VA slots share one role, so both subjects of a flip land on `subject`; FD
    subjects land on the slot they were used in. The style becomes that task's
    style arm.
    """
    from ava.image.tasks import get_task

    seen: dict[tuple[str, str, str], str] = {}

    def put(word: str, task: str, role: str, citation: str) -> None:
        key = (word, task, role)
        if key not in seen:
            seen[key] = citation

    for ex in PAPER_EXAMPLES:
        task = get_task(ex.task)
        if len(ex.prompts) != task.n_views:
            raise ValueError(f"{ex} does not match {task.name}'s {task.n_views} slots")
        for slot, word in zip(task.slots, ex.prompts, strict=True):
            put(word, ex.task, slot.role, ex.citation)
        put(ex.style, ex.task, STYLE, ex.citation)
    for style, word, citation in INVERSE_HIGH:
        put(word, "inverse_hybrid", "high", citation)
        put(style, "inverse_hybrid", STYLE, citation)
    return [(w, t, r, c) for (w, t, r), c in seen.items()]


def seed_paper_vocab(conn: sqlite3.Connection) -> int:
    """Insert the paper-derived arms with a uniform prior. Idempotent."""
    added = 0
    for word, task, role, citation in paper_arms():
        added += add_arm(
            conn, word, task, role, PAPER_SOURCE, UNIFORM_PRIOR, citation=citation
        )
    return added


__all__ = [
    "INVERSE_HIGH",
    "PAPER_EXAMPLES",
    "SUBJECT",
    "PaperExample",
    "paper_arms",
    "seed_paper_vocab",
]
