"""Deterministic stand-ins for the search layer's injected dependencies.

`FakeEmbedder` gives every word a unit vector that depends only on the word
(hashed with sha1, not `hash()`, which is salted per process), so two runs in
two processes see identical embeddings and archive files can be compared byte
for byte. Words listed under a category share a direction, with a small
per-word offset: that is what makes `"a horse"` and `"horse"` near-duplicates
and puts animals in one cluster and faces in another.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from ava.image.tasks import get_task
from ava.spec import CandidateSpec, Verdict

DIM = 16

CATEGORIES: dict[str, tuple[str, ...]] = {
    "animal": ("a horse", "horse", "a duck", "a rabbit", "a lemur", "a kangaroo"),
    "face": ("marilyn monroe", "albert einstein", "an old man", "a tudor portrait"),
    "scene": (
        "a landscape",
        "a snowy mountain village",
        "a canyon",
        "people around a campfire",
        "a waterfall",
    ),
    "thing": ("houseplants", "a skull", "a teddy bear", "a panda"),
}


def _unit(seed_text: str, dim: int = DIM) -> np.ndarray:
    digest = hashlib.sha1(seed_text.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    v = rng.normal(size=dim)
    return v / np.linalg.norm(v)


class FakeEmbedder:
    """Category axis plus a small word-specific offset; unit rows.

    Categories sit on orthogonal axes so the planted clusters are genuinely
    separable; a random direction per category can land two of them at a
    cosine of 0.7 in 16 dimensions, which is a bad fixture, not a bad k-means.
    """

    def __init__(self, spread: float = 0.15, dim: int = DIM) -> None:
        self.spread = spread
        self.dim = dim
        self.calls = 0
        self._category_of = {w: c for c, words in CATEGORIES.items() for w in words}

    def __call__(self, prompts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        rows = []
        for p in prompts:
            category = self._category_of.get(p)
            if category is None:
                v = _unit("word:" + p, self.dim)
            else:
                axis = np.zeros(self.dim)
                axis[list(CATEGORIES).index(category)] = 1.0
                v = axis + self.spread * _unit("word:" + p, self.dim)
            rows.append(v / np.linalg.norm(v))
        return np.stack(rows) if rows else np.zeros((0, self.dim))


def verdict_for(spec: CandidateSpec, *p: float, sep: Sequence[float] = ()) -> Verdict:
    """A verdict whose margins agree with the requested per-view probabilities."""
    task = get_task(spec.task)
    n = task.n_views
    if len(p) != n:
        raise ValueError(f"{task.name} has {n} views, got {len(p)} probabilities")
    scores = [[0.2] * n for _ in range(n)]
    margins = list(sep) if sep else [(0.1 if pi > 0.5 else -0.1) for pi in p]
    for i, m in enumerate(margins):
        scores[i][i] = 0.2 + m
    return Verdict(
        uid=spec.uid(),
        task=spec.task,
        slots=task.slot_names,
        scores=scores,
        p=list(p),
        j=min(p),
        alignment=min(scores[i][i] for i in range(n)),
        concealment=0.5,
        captions=[f"a view of {w}" for w in spec.prompts],
    )
