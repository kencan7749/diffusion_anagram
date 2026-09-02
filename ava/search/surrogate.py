"""A surrogate that ranks children before they are generated -- when it can.

One generation costs about thirty seconds; a Gaussian process prediction
costs nothing. If the prediction carries information, picking the children
with the highest upper confidence bound raises the yield of every round. If
it does not, picking by it is just a biased coin, and the search should say
so and fall back to a fair one.

Whether it carries information is not assumed. Every round the process is
refitted on everything evaluated so far and its leave-one-out predictions are
ranked against the actual `sep_min` values; the Spearman correlation is logged.
Only when the recent correlations clear a threshold does the acquisition
function drive selection; otherwise children are drawn uniformly, and the
predictions are still recorded so the check can be redone later.

The features are what is free at proposal time: the CLIP text embedding of
each slot's prompt and of the style, the cosines between slot embeddings (a
pair whose prompts already coincide is a `views_collapsed` waiting to happen),
the task, and the v1 component posteriors. The kernel is an RBF over those,
multiplied by a task kernel that lets evidence from one view inform another
at a discount. Numpy only: a few hundred points need nothing more.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ava.image.tasks import STYLE, get_task
from ava.search.embed import Embedder
from ava.spec import CandidateSpec
from ava.vocab import list_arms

UCB = "ucb"
UNIFORM = "uniform"


@dataclass(frozen=True)
class SurrogateConfig:
    min_train: int = 8  # fewer points than this and the LOO check is noise
    rho_threshold: float = 0.2
    window: int = 3  # rounds averaged for the skill decision
    beta: float = 1.0  # UCB: mean + beta * std
    task_similarity: float = 0.5  # kernel value between two different tasks
    noise: float = 0.1  # noise variance, as a fraction of signal variance
    max_slots: int = 4

    def __post_init__(self) -> None:
        if not 0.0 <= self.task_similarity <= 1.0:
            raise ValueError("task_similarity must be in [0, 1]")
        if self.noise <= 0.0:
            raise ValueError("noise must be positive")
        if self.window < 1 or self.min_train < 2:
            raise ValueError("window must be >= 1 and min_train >= 2")


# ---------------------------------------------------------------------------
# Rank correlation
# ---------------------------------------------------------------------------


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not break the correlation."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a: ArrayLike, b: ArrayLike) -> float:
    """Spearman's rho; 0.0 when either side is constant."""
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("need two sequences of equal length >= 2")
    rx, ry = _ranks(x), _ranks(y)
    if rx.std() == 0.0 or ry.std() == 0.0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


class FeatureMap:
    """Turn a candidate into a fixed-length vector, whatever its view count."""

    def __init__(
        self,
        embed: Embedder,
        conn: sqlite3.Connection,
        tasks: Sequence[str],
        max_slots: int = 4,
    ) -> None:
        self.embed = embed
        self.conn = conn
        self.tasks = list(tasks)
        self.max_slots = max_slots

    def _posterior_mean(self, word: str, task: str, role: str) -> float:
        for arm in list_arms(self.conn, task, role):
            if arm.word == word:
                return arm.mean
        return 0.5  # an unregistered word: the uniform prior's mean

    def __call__(self, spec: CandidateSpec) -> np.ndarray:
        task = get_task(spec.task)
        searched = task.searched_slots()
        if len(searched) > self.max_slots:
            raise ValueError(f"{task.name} has more than {self.max_slots} slots")
        words = [spec.prompts[i] for i in searched]
        slot_vecs = self.embed(words)
        style_vec = self.embed([spec.style])[0]
        d = slot_vecs.shape[1]

        slots = np.zeros((self.max_slots, d))
        slots[: len(words)] = slot_vecs

        n_pairs = self.max_slots * (self.max_slots - 1) // 2
        cosines = np.zeros(n_pairs)
        for k, (i, j) in enumerate(combinations(range(len(words)), 2)):
            cosines[k] = float(np.dot(slot_vecs[i], slot_vecs[j]))

        posteriors = np.full(self.max_slots + 1, 0.5)
        for k, i in enumerate(searched):
            posteriors[k] = self._posterior_mean(
                words[k], task.name, task.slots[i].role
            )
        posteriors[-1] = self._posterior_mean(spec.style, task.name, STYLE)

        one_hot = np.zeros(len(self.tasks))
        if task.name in self.tasks:
            one_hot[self.tasks.index(task.name)] = 1.0

        return np.concatenate(
            [slots.reshape(-1), style_vec, cosines, posteriors, one_hot]
        )


# ---------------------------------------------------------------------------
# Gaussian process
# ---------------------------------------------------------------------------


class GaussianProcess:
    """Zero-mean GP on standardised targets with an RBF x task kernel.

    Hyperparameters are set by rule rather than by marginal likelihood: the
    lengthscale is the median pairwise distance of the training inputs and
    the noise a fixed fraction of the signal variance. With a few hundred
    points and a skill check downstream, a deterministic fit that cannot
    diverge is worth more than a tuned one that sometimes does.
    """

    def __init__(self, task_similarity: float = 0.5, noise: float = 0.1) -> None:
        self.task_similarity = task_similarity
        self.noise = noise
        self._x: np.ndarray | None = None
        self._tasks: list[str] = []
        self._alpha: np.ndarray | None = None
        self._k_inv: np.ndarray | None = None
        self._ys: np.ndarray | None = None  # standardised training targets
        self._y_mean = 0.0
        self._y_std = 1.0
        self.lengthscale = 1.0

    def _kernel(
        self, a: np.ndarray, ta: Sequence[str], b: np.ndarray, tb: Sequence[str]
    ) -> np.ndarray:
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        rbf = np.exp(-0.5 * d2 / self.lengthscale**2)
        same = np.array(
            [[1.0 if x == y else self.task_similarity for y in tb] for x in ta]
        )
        return rbf * same

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        tasks: Sequence[str],
        lengthscale: float | None = None,
    ) -> None:
        """Fit; `lengthscale` None means the median heuristic on `x`."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if len(x) != len(y) or len(x) != len(tasks) or len(x) < 2:
            raise ValueError("need at least two matching inputs, targets and tasks")
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) or 1.0
        ys = (y - self._y_mean) / self._y_std

        if lengthscale is None:
            d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
            upper = d2[np.triu_indices(len(x), k=1)]
            median = (
                float(np.sqrt(np.median(upper[upper > 0])))
                if (upper > 0).any()
                else 0.0
            )
            lengthscale = median if median > 0 else 1.0
        self.lengthscale = float(lengthscale)

        self._x, self._tasks = x, list(tasks)
        k = self._kernel(x, tasks, x, tasks) + self.noise * np.eye(len(x))
        self._k_inv = np.linalg.inv(k)
        self._alpha = self._k_inv @ ys
        self._ys = ys

    def predict(
        self, x: np.ndarray, tasks: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._x is None or self._alpha is None or self._k_inv is None:
            raise RuntimeError("fit() first")
        x = np.asarray(x, dtype=np.float64)
        ks = self._kernel(x, tasks, self._x, self._tasks)
        mean = ks @ self._alpha
        var = 1.0 - np.einsum("ij,jk,ik->i", ks, self._k_inv, ks)
        std = np.sqrt(np.maximum(var, 0.0))
        return mean * self._y_std + self._y_mean, std * self._y_std

    def loo_predictions(self) -> np.ndarray:
        """Leave-one-out posterior means, in closed form (Rasmussen & Williams 5.4.2).

        mu_-i = y_i - (K^-1 y)_i / (K^-1)_ii, on the standardised targets. The
        standardisation itself is not left out, which is why a brute-force
        refit differs in the third decimal; the ranking is what is checked.
        """
        if self._alpha is None or self._k_inv is None or self._ys is None:
            raise RuntimeError("fit() first")
        loo = self._ys - self._alpha / np.diag(self._k_inv)
        return loo * self._y_std + self._y_mean


# ---------------------------------------------------------------------------
# The surrogate with its skill check
# ---------------------------------------------------------------------------


@dataclass
class SkillEntry:
    round_index: int
    n_train: int
    rho: float | None
    mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_index,
            "n_train": self.n_train,
            "rho": self.rho,
            "mode": self.mode,
        }


@dataclass
class Surrogate:
    cfg: SurrogateConfig
    features: FeatureMap
    log: list[SkillEntry] = field(default_factory=list)
    _x: list[np.ndarray] = field(default_factory=list)
    _y: list[float] = field(default_factory=list)
    _tasks: list[str] = field(default_factory=list)
    _gp: GaussianProcess | None = None

    @property
    def n_train(self) -> int:
        return len(self._y)

    def add(self, spec: CandidateSpec, y: float) -> None:
        self._x.append(self.features(spec))
        self._y.append(float(y))
        self._tasks.append(spec.task)

    def refit(self, round_index: int) -> SkillEntry:
        """Refit on everything seen and log this round's leave-one-out skill."""
        rho: float | None = None
        if self.n_train >= self.cfg.min_train:
            gp = GaussianProcess(self.cfg.task_similarity, self.cfg.noise)
            gp.fit(np.stack(self._x), np.asarray(self._y), self._tasks)
            self._gp = gp
            rho = spearman(gp.loo_predictions(), self._y)
        entry = SkillEntry(round_index, self.n_train, rho, UNIFORM)
        self.log.append(entry)
        entry.mode = UCB if self.skilled else UNIFORM
        return entry

    @property
    def skilled(self) -> bool:
        """Recent LOO rhos average above threshold, and every one was testable."""
        recent = self.log[-self.cfg.window :]
        if len(recent) < self.cfg.window:
            return False
        rhos = [e.rho for e in recent]
        if any(r is None for r in rhos):
            return False
        return (
            float(np.mean([r for r in rhos if r is not None])) >= self.cfg.rho_threshold
        )

    @property
    def mode(self) -> str:
        return UCB if self.skilled else UNIFORM

    def predict(self, specs: Sequence[CandidateSpec]) -> list[dict[str, float]]:
        """Mean, std and UCB per candidate; empty dicts when nothing is fitted yet."""
        if self._gp is None or not specs:
            return [{} for _ in specs]
        x = np.stack([self.features(s) for s in specs])
        mean, std = self._gp.predict(x, [s.task for s in specs])
        return [
            {
                "mean": float(m),
                "std": float(s),
                "ucb": float(m + self.cfg.beta * s),
            }
            for m, s in zip(mean, std, strict=True)
        ]

    def select(
        self, specs: Sequence[CandidateSpec], n: int, rng: np.random.Generator
    ) -> tuple[list[int], list[dict[str, float]], str]:
        """Pick `n` of `specs`: by UCB when skilled, uniformly otherwise.

        Predictions are returned for every candidate either way, so the
        record shows what the surrogate would have said even when it was not
        trusted.
        """
        preds = self.predict(specs)
        n = min(n, len(specs))
        if n == 0:
            return [], preds, self.mode
        if self.skilled and preds and preds[0]:
            order = sorted(range(len(specs)), key=lambda i: (-preds[i]["ucb"], i))
            return order[:n], preds, UCB
        chosen = sorted(int(i) for i in rng.choice(len(specs), size=n, replace=False))
        return chosen, preds, UNIFORM

    def log_dicts(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.log]

    @staticmethod
    def log_from_dicts(rows: Sequence[dict[str, Any]]) -> list[SkillEntry]:
        return [
            SkillEntry(int(r["round"]), int(r["n_train"]), r.get("rho"), str(r["mode"]))
            for r in rows
        ]
