"""Rejecting near-duplicate children before they cost a generation.

The cheapest evaluation is the one not run. `a horse` / `marilyn monroe` and
`horse` / `marilyn monroe` are, to CLIP, the same pair, and evaluating both
buys one noisy Bernoulli trial twice. A child whose prompts are within cosine
`eta` of an already evaluated pair -- in the same task and the same style --
is dropped before the surrogate sees it.

Comparison is within (task, style) on purpose: the style arm is a first-class
component with its own posterior, so the same subjects in a different style
is a distinct experiment, and `restyle` children must not be rejected as
duplicates of their parent.

Every maximum similarity that was checked is recorded, so the run reports
the distribution `eta` was applied to; whether 0.95 was the right line is
then a question the data can answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import permutations
from typing import Any

import numpy as np

from ava.image.tasks import get_task
from ava.search.embed import Embedder
from ava.spec import CandidateSpec

QUANTILES = (0.5, 0.9, 0.95, 0.99)


@dataclass
class DedupStats:
    n_checked: int = 0
    n_rejected: int = 0
    # Every max-cosine seen; small floats, bounded by the number of children a
    # run considers, and what the eta decision is later made on.
    max_similarities: list[float] = field(default_factory=list)

    def summary(self, eta: float) -> dict[str, Any]:
        sims = np.asarray(self.max_similarities, dtype=np.float64)
        quantiles = (
            {f"q{int(q * 100)}": float(np.quantile(sims, q)) for q in QUANTILES}
            if len(sims)
            else {}
        )
        return {
            "eta": eta,
            "n_checked": self.n_checked,
            "n_rejected": self.n_rejected,
            "max_similarity_quantiles": quantiles,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_checked": self.n_checked,
            "n_rejected": self.n_rejected,
            "max_similarities": [round(s, 6) for s in self.max_similarities],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DedupStats:
        return cls(
            n_checked=int(d["n_checked"]),
            n_rejected=int(d["n_rejected"]),
            max_similarities=[float(s) for s in d["max_similarities"]],
        )


class Deduplicator:
    """Embedding-cosine rejection against everything already evaluated."""

    def __init__(self, embed: Embedder, eta: float, stats: DedupStats | None = None):
        if not 0.0 < eta <= 1.0:
            raise ValueError(f"eta must be in (0, 1], got {eta}")
        self.embed = embed
        self.eta = eta
        self.stats = stats if stats is not None else DedupStats()
        self._index: dict[tuple[str, str], list[np.ndarray]] = {}

    # -- vectors -----------------------------------------------------------

    def _slot_vectors(self, spec: CandidateSpec) -> np.ndarray:
        task = get_task(spec.task)
        words = [spec.prompts[i] for i in task.searched_slots()]
        return self.embed(words)

    def _orderings(self, spec: CandidateSpec) -> list[np.ndarray]:
        """The concatenated slot vectors, under every slot order that means the same."""
        rows = self._slot_vectors(spec)
        if get_task(spec.task).symmetric:
            return [rows[list(p)].reshape(-1) for p in permutations(range(len(rows)))]
        return [rows.reshape(-1)]

    # -- the index ---------------------------------------------------------

    def add(self, spec: CandidateSpec) -> None:
        self._index.setdefault((spec.task, spec.style), []).append(
            self._orderings(spec)[0]
        )

    def add_all(self, specs: Sequence[CandidateSpec]) -> None:
        for spec in specs:
            self.add(spec)

    def max_similarity(self, spec: CandidateSpec) -> float:
        """Highest mean slot cosine against evaluated pairs of the same task and style.

        Every stored vector is a concatenation of unit slot vectors, so the dot
        product of two of them is the sum of per-slot cosines; dividing by the
        slot count gives the mean cosine over slots.
        """
        known = self._index.get((spec.task, spec.style), [])
        if not known:
            return 0.0
        n_slots = len(get_task(spec.task).searched_slots())
        stack = np.stack(known)
        return max(float((stack @ v).max()) / n_slots for v in self._orderings(spec))

    def check(self, spec: CandidateSpec) -> tuple[bool, float]:
        """Record and return (accepted, max similarity)."""
        sim = self.max_similarity(spec)
        self.stats.n_checked += 1
        self.stats.max_similarities.append(sim)
        accepted = sim <= self.eta
        if not accepted:
            self.stats.n_rejected += 1
        return accepted, sim

    def summary(self) -> dict[str, Any]:
        return self.stats.summary(self.eta)
