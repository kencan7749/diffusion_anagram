"""Text embeddings as an injected dependency.

The search layer needs a vector per prompt string for three things: the
behaviour descriptor (which semantic cluster a prompt falls in), duplicate
rejection, and the surrogate's features. It must not load a model of its own:
in a run the vectors come from the judge's CLIP text tower, so the proposer
and the judge agree on what "similar" means, and in tests they come from a
deterministic stand-in. Hence a callable, not an import.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def __call__(self, prompts: Sequence[str]) -> np.ndarray:
        """Return one unit-norm row per prompt: shape (len(prompts), d)."""
        ...


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected (n, d), got {x.shape}")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


class CachingEmbedder:
    """Memoise an embedder per string.

    A round asks for the same few hundred words many times over (every child
    of every parent), so the underlying call, which may be a GPU forward pass,
    happens once per distinct string.
    """

    def __init__(self, embed: Embedder) -> None:
        self._embed = embed
        self._cache: dict[str, np.ndarray] = {}

    def __call__(self, prompts: Sequence[str]) -> np.ndarray:
        missing = sorted({p for p in prompts if p not in self._cache})
        if missing:
            rows = normalize_rows(self._embed(missing))
            if rows.shape[0] != len(missing):
                raise ValueError(
                    f"embedder returned {rows.shape[0]} rows for {len(missing)} prompts"
                )
            for p, row in zip(missing, rows, strict=True):
                self._cache[p] = row
        if not prompts:
            return np.zeros((0, self.dim), dtype=np.float64)
        return np.stack([self._cache[p] for p in prompts])

    @property
    def dim(self) -> int:
        if not self._cache:
            raise RuntimeError("no embedding computed yet; dimension unknown")
        return int(next(iter(self._cache.values())).shape[0])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
