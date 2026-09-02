"""MAP-Elites archive: the best pair per (task, semantic cell), plus everything.

The archive is where diversity lives. Rather than putting novelty into the
objective, the search keeps one elite per *cell* and treats an empty cell as
a place worth exploring. A cell is the task plus the semantic cluster of each
searched prompt, so "a flip of an animal and a landscape" and "a flip of two
faces" compete in different cells and one cannot crowd the other out.

Clusters come from k-means over CLIP text embeddings of the whole vocabulary,
fitted once at run start under a fixed seed and persisted, so every round of a
run and every later analysis of it use the same map.

The unit stored is a *pair* -- (task, prompts, style) -- not a single
generation. Every seed evaluated for a pair is kept on its individual, the
fitness is the mean of `sep_min` over those seeds, and nothing is ever deleted:
`archive.jsonl` carries every individual with every evaluation, and the elite
flag is derived from it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ava.image.tasks import get_task
from ava.search.embed import Embedder, normalize_rows
from ava.spec import CandidateSpec, Verdict

PairKey = tuple[str, tuple[str, ...], str]
CellKey = tuple[Any, ...]  # (task, cluster_1, cluster_2[, ...])


def pair_key(spec: CandidateSpec) -> PairKey:
    """What makes two candidates the same pair: everything but the seed."""
    return spec.task, tuple(spec.prompts), spec.style


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


@dataclass
class Clusterer:
    """k-means over unit-norm embeddings; the fitted vocabulary is kept too."""

    centroids: np.ndarray
    words: list[str]
    labels: np.ndarray
    seed: int

    @property
    def k(self) -> int:
        return int(self.centroids.shape[0])

    @classmethod
    def fit(
        cls,
        words: Sequence[str],
        embed: Embedder,
        k: int,
        seed: int,
        max_iter: int = 100,
        n_init: int = 10,
    ) -> Clusterer:
        """Lloyd's algorithm with k-means++ seeding, entirely under `seed`.

        `n_init` restarts are run from the one seeded generator and the
        lowest-inertia solution kept, because a single k-means++ start can
        split one planted cluster and merge two others.

        Fewer distinct words than clusters is allowed and simply yields fewer
        clusters; a tiny vocabulary is a test's problem, not the archive's.
        """
        unique = sorted(set(words))
        if not unique:
            raise ValueError("cannot cluster an empty vocabulary")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        x = normalize_rows(embed(unique))
        k = min(k, len(unique))
        rng = np.random.default_rng(seed)

        best: tuple[float, np.ndarray, np.ndarray] | None = None
        for _ in range(max(1, n_init)):
            c, labels = _lloyd(x, k, rng, max_iter)
            inertia = float(((x - c[labels]) ** 2).sum())
            if best is None or inertia < best[0]:
                best = (inertia, c, labels)
        assert best is not None
        return cls(centroids=best[1], words=unique, labels=best[2], seed=seed)

    def assign(self, vectors: np.ndarray) -> np.ndarray:
        return _nearest(normalize_rows(vectors), self.centroids)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            centroids=self.centroids,
            words=np.asarray(self.words, dtype=object),
            labels=self.labels,
            seed=np.asarray(self.seed),
        )

    @classmethod
    def load(cls, path: Path) -> Clusterer:
        with np.load(path, allow_pickle=True) as z:
            return cls(
                centroids=np.asarray(z["centroids"], dtype=np.float64),
                words=[str(w) for w in z["words"]],
                labels=np.asarray(z["labels"], dtype=np.int64),
                seed=int(z["seed"]),
            )


def _lloyd(
    x: np.ndarray, k: int, rng: np.random.Generator, max_iter: int
) -> tuple[np.ndarray, np.ndarray]:
    """One k-means run: k-means++ seeding, then Lloyd iterations to convergence."""
    n = len(x)
    # k-means++: each next centre is drawn proportional to squared distance.
    centroids = [x[int(rng.integers(n))]]
    for _ in range(1, k):
        d2 = np.min(
            ((x[:, None, :] - np.stack(centroids)[None, :, :]) ** 2).sum(-1), axis=1
        )
        total = float(d2.sum())
        if total <= 0.0:  # every remaining point coincides with a centre
            centroids.append(x[int(rng.integers(n))])
            continue
        centroids.append(x[int(rng.choice(n, p=d2 / total))])
    c = np.stack(centroids).copy()

    labels = np.full(n, -1, dtype=np.int64)
    for _ in range(max_iter):
        new_labels = _nearest(x, c)
        for j in range(k):
            members = x[new_labels == j]
            if len(members):
                c[j] = members.mean(axis=0)
            else:  # empty cluster: move its centre to the farthest point
                far = int(np.argmax(((x - c[new_labels]) ** 2).sum(-1)))
                c[j] = x[far]
                new_labels[far] = j
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    return c, labels


def _nearest(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    d2 = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    return np.argmin(d2, axis=1).astype(np.int64)


# ---------------------------------------------------------------------------
# Individuals
# ---------------------------------------------------------------------------


@dataclass
class Evaluation:
    """One generation of a pair: one seed, one verdict, reduced to what search needs."""

    uid: str
    seed: int
    sep_min: float
    sep: list[float]
    j: float
    ok: bool  # every view read as its own prompt
    lost: list[str]  # slot names that did not
    round_index: int
    origin: str

    @classmethod
    def from_verdict(
        cls, spec: CandidateSpec, verdict: Verdict, round_index: int, origin: str
    ) -> Evaluation:
        return cls(
            uid=spec.uid(),
            seed=spec.seed,
            sep_min=float(verdict.sep_min),
            sep=[float(s) for s in verdict.sep],
            j=float(verdict.j),
            ok=verdict.diagnose() == "ok",
            lost=verdict.lost_slots(),
            round_index=round_index,
            origin=origin,
        )


@dataclass
class Individual:
    """A pair and every evaluation of it. The seed on `spec` is the first seen."""

    spec: CandidateSpec
    cell: tuple[int, ...]
    first_round: int
    operator: str | None = None
    parents: list[str] = field(default_factory=list)
    n_children: int = 0
    evaluations: list[Evaluation] = field(default_factory=list)

    @property
    def key(self) -> PairKey:
        return pair_key(self.spec)

    @property
    def key_str(self) -> str:
        return key_to_str(self.key)

    @property
    def cell_key(self) -> CellKey:
        return (self.spec.task, *self.cell)

    @property
    def n_seeds(self) -> int:
        return len(self.evaluations)

    @property
    def n_ok(self) -> int:
        return sum(1 for e in self.evaluations if e.ok)

    @property
    def seeds(self) -> list[int]:
        return [e.seed for e in self.evaluations]

    @property
    def fitness(self) -> float:
        """Mean weakest-view margin over the seeds evaluated so far."""
        if not self.evaluations:
            raise ValueError(f"{self.key_str}: no evaluation yet")
        return float(np.mean([e.sep_min for e in self.evaluations]))

    @property
    def latest(self) -> Evaluation:
        if not self.evaluations:
            raise ValueError(f"{self.key_str}: no evaluation yet")
        return self.evaluations[-1]

    def rank_key(self) -> tuple[int, float, int]:
        """Elite order within a cell: held at least once, then fitness, then seeds."""
        return (1 if self.n_ok > 0 else 0, self.fitness, self.n_seeds)

    def to_dict(self, elite: bool) -> dict[str, Any]:
        return {
            "key": self.key_str,
            "spec": asdict(self.spec),
            "cell": list(self.cell),
            "first_round": self.first_round,
            "operator": self.operator,
            "parents": list(self.parents),
            "n_children": self.n_children,
            "evaluations": [asdict(e) for e in self.evaluations],
            "fitness": self.fitness,
            "n_seeds": self.n_seeds,
            "n_ok": self.n_ok,
            "elite": elite,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Individual:
        return cls(
            spec=CandidateSpec(**d["spec"]),
            cell=tuple(int(c) for c in d["cell"]),
            first_round=int(d["first_round"]),
            operator=d.get("operator"),
            parents=[str(p) for p in d.get("parents", [])],
            n_children=int(d.get("n_children", 0)),
            evaluations=[Evaluation(**e) for e in d["evaluations"]],
        )


def key_to_str(key: PairKey) -> str:
    task, prompts, style = key
    return json.dumps([task, list(prompts), style], ensure_ascii=False)


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------


class Archive:
    """Every pair ever evaluated, and the elite of every occupied cell.

    Cells for a symmetric task (every slot shares the role `subject`) sort
    their cluster ids, because swapping the two prompts of a flip gives the
    same illusion flipped and should not open a second cell.
    """

    def __init__(self, clusterer: Clusterer, embed: Embedder) -> None:
        self.clusterer = clusterer
        self.embed = embed
        self.individuals: dict[PairKey, Individual] = {}
        self.elites: dict[CellKey, PairKey] = {}
        self._cluster_cache: dict[str, int] = {}

    # -- descriptors -----------------------------------------------------

    def cluster_of(self, word: str) -> int:
        if word not in self._cluster_cache:
            self._cluster_cache[word] = int(
                self.clusterer.assign(self.embed([word]))[0]
            )
        return self._cluster_cache[word]

    def cell_for(self, spec: CandidateSpec) -> tuple[int, ...]:
        task = get_task(spec.task)
        ids = [self.cluster_of(spec.prompts[i]) for i in task.searched_slots()]
        if task.symmetric:
            ids.sort()
        return tuple(ids)

    # -- updates ---------------------------------------------------------

    def has_uid(self, uid: str) -> bool:
        return any(
            e.uid == uid for ind in self.individuals.values() for e in ind.evaluations
        )

    def add(
        self,
        spec: CandidateSpec,
        verdict: Verdict,
        round_index: int,
        origin: str,
        operator: str | None = None,
        parents: Sequence[str] = (),
    ) -> Individual:
        """Fold one evaluation in. Re-adding a uid already seen changes nothing."""
        key = pair_key(spec)
        ind = self.individuals.get(key)
        if ind is None:
            ind = Individual(
                spec=spec,
                cell=self.cell_for(spec),
                first_round=round_index,
                operator=operator,
                parents=list(parents),
            )
            self.individuals[key] = ind
            for parent in parents:
                p = self.by_key_str(parent)
                if p is not None:
                    p.n_children += 1
        uid = spec.uid()
        if not any(e.uid == uid for e in ind.evaluations):
            ind.evaluations.append(
                Evaluation.from_verdict(spec, verdict, round_index, origin)
            )
        self._refresh_cell(ind.cell_key)
        return ind

    def _refresh_cell(self, cell_key: CellKey) -> None:
        members = [i for i in self.individuals.values() if i.cell_key == cell_key]
        if not members:
            self.elites.pop(cell_key, None)
            return
        best = max(members, key=lambda i: i.rank_key())
        self.elites[cell_key] = best.key

    # -- queries ---------------------------------------------------------

    def get(self, spec: CandidateSpec) -> Individual | None:
        return self.individuals.get(pair_key(spec))

    def by_key_str(self, key: str) -> Individual | None:
        for ind in self.individuals.values():
            if ind.key_str == key:
                return ind
        return None

    def elite_individuals(self, task: str | None = None) -> list[Individual]:
        """Elites in a fixed order (by cell key), optionally for one task."""
        out = []
        for cell_key in sorted(self.elites, key=repr):
            ind = self.individuals[self.elites[cell_key]]
            if task is None or ind.spec.task == task:
                out.append(ind)
        return out

    def fitness_values(self) -> list[float]:
        return [i.fitness for i in self.individuals.values() if i.evaluations]

    def coverage(self) -> dict[str, dict[str, int]]:
        """Per task: cells occupied at all, and cells whose elite held at least once."""
        out: dict[str, dict[str, int]] = {}
        for ind in self.elite_individuals():
            entry = out.setdefault(ind.spec.task, {"filled": 0, "held": 0})
            entry["filled"] += 1
            if ind.n_ok > 0:
                entry["held"] += 1
        return out

    # -- parent selection ------------------------------------------------

    def parent_weights(
        self, lam: float = 1.0, task: str | None = None
    ) -> tuple[list[Individual], np.ndarray]:
        """ShinkaEvolve's weighting: sigmoid(standardised fitness) / (1 + children).

        Fitness is standardised (median, std) rather than used raw so that
        `lam` means the same thing whatever scale `sep_min` runs at. The
        1/(1 + children) factor is what keeps the search from digging in one
        place: a parent that has already been mutated many times yields to one
        that has not.
        """
        elites = self.elite_individuals(task)
        if not elites:
            return [], np.zeros(0)
        f = np.array([e.fitness for e in elites])
        z = (f - np.median(f)) / (f.std() + 1e-9)
        w = (
            1.0
            / (1.0 + np.exp(-lam * z))
            / (1.0 + np.array([e.n_children for e in elites]))
        )
        return elites, w / w.sum()

    def sample_parent(
        self, rng: np.random.Generator, lam: float = 1.0, task: str | None = None
    ) -> Individual | None:
        elites, w = self.parent_weights(lam, task)
        if not elites:
            return None
        return elites[int(rng.choice(len(elites), p=w))]

    # -- persistence -----------------------------------------------------

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        elite_keys = set(self.elites.values())
        with path.open("w", encoding="utf-8") as f:
            for ind in self.individuals.values():
                f.write(
                    json.dumps(
                        ind.to_dict(elite=ind.key in elite_keys), ensure_ascii=False
                    )
                    + "\n"
                )

    @classmethod
    def read_jsonl(cls, path: Path, clusterer: Clusterer, embed: Embedder) -> Archive:
        archive = cls(clusterer, embed)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ind = Individual.from_dict(json.loads(line))
            archive.individuals[ind.key] = ind
        for cell_key in {i.cell_key for i in archive.individuals.values()}:
            archive._refresh_cell(cell_key)
        return archive

    def __len__(self) -> int:
        return len(self.individuals)

    def __iter__(self) -> Iterable[Individual]:
        return iter(self.individuals.values())
