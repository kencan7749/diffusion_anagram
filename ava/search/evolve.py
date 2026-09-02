"""EvolutionaryProposer: one round of quality-diversity search, step by step.

    round r:
      1. racing       promotable pairs get their next seed (a share of k)
      2. parents      weighted sampling from the archive's elites
      3. mutation     the operator bandit picks an operator; M children, M >> k
      4. rejection    children within cosine eta of an evaluated pair are dropped
      5. selection    the surrogate's UCB picks the rest of k, if it has skill;
                      uniformly otherwise
      6. reflection   (in `observe`) archive, pair Beta, operator reward,
                      surrogate refit, prediction-vs-outcome, all persisted

The first round has no archive to mutate, so it is bootstrapped from the v1
posteriors: candidates filled by Thompson sampling, exactly what the bandit
proposer's exploit quota would have drawn. From then on every new candidate
is a child of something that was evaluated.

The v1 knowledge layer is untouched. The loop still assigns per-component
credit to the vocabulary database for every evaluation, and the operators
draw replacement words from those posteriors; this proposer adds the pair-level
structure (archive, racing, surrogate) on top.

Persisted under the run directory, and reloaded on resume:

    archive.jsonl       every pair with every seed; the elite flag per cell
    clusters.npz        the k-means map the cells are defined on
    search_state.json   operator statistics, pending children, dedup
                        statistics, the surrogate's skill log, per-round summary
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ava.image.tasks import STYLE, IllusionTask, get_task
from ava.propose import MAX_DRAW_ATTEMPTS, CandidateBuilder, Proposal
from ava.search.archive import Archive, Clusterer
from ava.search.embed import CachingEmbedder, Embedder
from ava.search.novelty import Deduplicator, DedupStats
from ava.search.operators import (
    OPERATOR_NAMES,
    OPERATORS,
    OperatorBandit,
    OperatorContext,
    RunningStats,
    improvement_reward,
)
from ava.search.racing import RACE, RacingConfig, race_proposals
from ava.search.surrogate import FeatureMap, Surrogate, SurrogateConfig
from ava.spec import CandidateSpec, RunState
from ava.vocab import draw_arm, pooled_arms

BOOTSTRAP = "bootstrap"
EVOLVE = "evolve"

ARCHIVE_FILENAME = "archive.jsonl"
CLUSTERS_FILENAME = "clusters.npz"
STATE_FILENAME = "search_state.json"


@dataclass(frozen=True)
class SearchConfig:
    """Every knob of the evolutionary proposer. Recorded in config.yaml."""

    clusters: int = 8  # k-means cells per prompt slot
    cluster_seed: int = 0
    eta: float = 0.95  # duplicate rejection cosine
    children_per_slot: int = 32  # M = children_per_slot * (k - races)
    parent_lambda: float = 1.0  # sharpness of the fitness weighting
    operator_c: float = 1.0  # UCB1 exploration constant
    racing: RacingConfig = field(default_factory=RacingConfig)
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)

    def __post_init__(self) -> None:
        if self.clusters < 1:
            raise ValueError("clusters must be at least 1")
        if self.children_per_slot < 1:
            raise ValueError("children_per_slot must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def vocabulary_words(conn: sqlite3.Connection, tasks: Sequence[str]) -> list[str]:
    """Every subject word any configured task can draw: the whole pooled vocabulary."""
    return sorted(
        {
            a.word
            for t in tasks
            for task in [get_task(t)]
            for i in task.searched_slots()
            for a in pooled_arms(conn, t, task.slots[i].role)
        }
    )


class EvolutionaryProposer:
    """Quality-diversity search over pairs; see the module docstring."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        rng: np.random.Generator,
        embed: Embedder,
        tasks: Sequence[str] = ("hybrid",),
        cfg: SearchConfig | None = None,
        screening_seed: int = 0,
        guidance_scale: float = 10.0,
        num_inference_steps: int = 30,
        ref_image: str | None = None,
        ref_prompt: str | None = None,
        state_dir: Path | None = None,
    ) -> None:
        self.conn = conn
        self.rng = rng
        self.embed = CachingEmbedder(embed)
        self.cfg = cfg if cfg is not None else SearchConfig()
        self.state_dir = state_dir
        self.builder = CandidateBuilder(
            tasks,
            screening_seed,
            guidance_scale,
            num_inference_steps,
            ref_image,
            ref_prompt,
        )
        task_names = [t.name for t in self.builder.tasks]

        clusters_path = state_dir / CLUSTERS_FILENAME if state_dir else None
        if clusters_path is not None and clusters_path.exists():
            self.clusterer = Clusterer.load(clusters_path)
        else:
            self.clusterer = Clusterer.fit(
                vocabulary_words(conn, task_names),
                self.embed,
                self.cfg.clusters,
                seed=self.cfg.cluster_seed,
            )
            if clusters_path is not None:
                self.clusterer.save(clusters_path)

        archive_path = state_dir / ARCHIVE_FILENAME if state_dir else None
        if archive_path is not None and archive_path.exists():
            self.archive = Archive.read_jsonl(archive_path, self.clusterer, self.embed)
        else:
            self.archive = Archive(self.clusterer, self.embed)

        self.dedup = Deduplicator(self.embed, self.cfg.eta)
        self.dedup.add_all([ind.spec for ind in self.archive])
        self.operators = OperatorBandit(c=self.cfg.operator_c)
        self.fitness_stats = RunningStats()
        # uid -> how a proposed candidate came about; consumed by observe().
        self.pending: dict[str, dict[str, Any]] = {}
        self.surrogate = Surrogate(
            self.cfg.surrogate,
            FeatureMap(self.embed, conn, task_names, self.cfg.surrogate.max_slots),
        )
        for ind in self.archive:
            for e in ind.evaluations:
                spec = CandidateSpec(**{**asdict(ind.spec), "seed": e.seed})
                self.surrogate.add(spec, e.sep_min)
        self.rounds: list[dict[str, Any]] = []

        state_path = state_dir / STATE_FILENAME if state_dir else None
        if state_path is not None and state_path.exists():
            self._load_state(json.loads(state_path.read_text(encoding="utf-8")))

    # -- properties ------------------------------------------------------

    @property
    def tasks(self) -> list[IllusionTask]:
        return self.builder.tasks

    def _context(self) -> OperatorContext:
        return OperatorContext(self.conn, self.rng, self.builder, self.archive)

    # -- persistence -----------------------------------------------------

    def _state_dict(self) -> dict[str, Any]:
        return {
            "operators": self.operators.to_dict(),
            "fitness_stats": self.fitness_stats.to_dict(),
            "pending": self.pending,
            "dedup": self.dedup.stats.to_dict(),
            "surrogate_log": self.surrogate.log_dicts(),
            "rounds": self.rounds,
        }

    def _load_state(self, d: dict[str, Any]) -> None:
        self.operators = OperatorBandit.from_dict(d["operators"])
        self.fitness_stats = RunningStats.from_dict(d["fitness_stats"])
        self.pending = dict(d.get("pending", {}))
        self.dedup.stats = DedupStats.from_dict(d["dedup"])
        self.surrogate.log = Surrogate.log_from_dicts(d.get("surrogate_log", []))
        self.rounds = list(d.get("rounds", []))
        if self.surrogate.n_train >= self.cfg.surrogate.min_train:
            # Refit without logging: the log already has this round's entry.
            self.surrogate.refit(-1)
            self.surrogate.log.pop()

    def persist(self) -> None:
        if self.state_dir is None:
            return
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.archive.write_jsonl(self.state_dir / ARCHIVE_FILENAME)
        (self.state_dir / STATE_FILENAME).write_text(
            json.dumps(
                self._state_dict(), indent=2, ensure_ascii=False, sort_keys=True
            ),
            encoding="utf-8",
        )

    # -- reflection ------------------------------------------------------

    def observe(self, state: RunState) -> None:
        """Fold a finished round into the archive and every statistic. Idempotent."""
        round_index = state.round_index - 1
        ingested = 0
        for spec, verdict in state.last_round:
            uid = spec.uid()
            if self.archive.has_uid(uid):
                continue
            meta = self.pending.pop(uid, {})
            operator = meta.get("operator")
            ind = self.archive.add(
                spec,
                verdict,
                round_index,
                str(meta.get("origin", "external")),
                operator=operator,
                parents=[str(p) for p in meta.get("parents", [])],
            )
            if ind.n_seeds == 1:
                self.dedup.add(spec)
            self.fitness_stats.update(float(verdict.sep_min))
            self.surrogate.add(spec, float(verdict.sep_min))
            parent_fitness = meta.get("parent_fitness")
            if operator in OPERATOR_NAMES and parent_fitness is not None:
                self.operators.update(
                    str(operator),
                    improvement_reward(
                        ind.fitness, float(parent_fitness), self.fitness_stats.std
                    ),
                )
            ingested += 1

        skill = self.surrogate.refit(round_index)
        self.rounds.append(
            {
                "round": round_index,
                "ingested": ingested,
                "pairs": len(self.archive),
                "coverage": self.archive.coverage(),
                "fitness_std": self.fitness_stats.std,
                "surrogate": skill.to_dict(),
                "operators": self.operators.to_dict()["means"],
                "dedup": self.dedup.summary(),
            }
        )
        self.persist()

    # -- proposal --------------------------------------------------------

    def _bootstrap_child(self) -> tuple[CandidateSpec, str] | None:
        """A fresh candidate from the v1 posteriors, for a round with no parents."""
        task = self.builder.next_task()
        words: dict[int, str] = {}
        try:
            for i in task.searched_slots():
                taken = frozenset(words.values()) | self.builder.reserved(task)
                words[i] = draw_arm(
                    self.conn, task.name, task.slots[i].role, self.rng, exclude=taken
                ).word
            style = draw_arm(self.conn, task.name, STYLE, self.rng).word
        except LookupError:
            return None
        detail = f"bootstrap task={task.name} " + " ".join(
            f"{task.slots[i].name}={words[i]!r}" for i in sorted(words)
        )
        return self.builder.build(task, words, style), f"{detail} style={style!r}"

    def _children(
        self, n_new: int, seen: set[str]
    ) -> list[tuple[CandidateSpec, str, dict[str, Any]]]:
        """Generate the candidate pool: mutated children, bootstrap-filled if short."""
        target = self.cfg.children_per_slot * n_new
        pool: list[tuple[CandidateSpec, str, dict[str, Any]]] = []
        ctx = self._context()

        def consider(spec: CandidateSpec, detail: str, meta: dict[str, Any]) -> bool:
            uid = spec.uid()
            if uid in seen:
                return False
            seen.add(uid)
            accepted, sim = self.dedup.check(spec)
            if not accepted:
                return False
            meta["similarity"] = round(sim, 4)
            pool.append((spec, detail, meta))
            return True

        if self.archive.elites:
            attempts = 0
            while len(pool) < target and attempts < target * 4:
                attempts += 1
                parent = self.archive.sample_parent(self.rng, self.cfg.parent_lambda)
                if parent is None:
                    break
                name = self.operators.choose(self.rng)
                result = OPERATORS[name](parent, ctx)
                if result is None:
                    continue
                spec, detail = result
                consider(
                    spec,
                    detail,
                    {
                        "operator": name,
                        "parents": [parent.key_str],
                        "parent_fitness": parent.fitness,
                    },
                )

        # Without parents (round 0) the whole pool is bootstrapped; with parents,
        # a shortfall below k is topped up so a round is never short-changed.
        floor = target if not self.archive.elites else n_new
        attempts = 0
        while len(pool) < floor and attempts < floor * MAX_DRAW_ATTEMPTS:
            attempts += 1
            built = self._bootstrap_child()
            if built is None:
                break
            consider(
                built[0],
                built[1],
                {"operator": None, "parents": [], "parent_fitness": None},
            )
        return pool

    def propose(self, state: RunState, k: int) -> list[Proposal]:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        seen = set(state.evaluated_uids)
        out: list[Proposal] = []

        # Racing never takes the whole round: at least one slot is always a
        # new candidate, or a round could consist of nothing at all.
        n_race = min(int(round(k * self.cfg.racing.fraction)), k - 1)
        for p in race_proposals(self.archive, self.cfg.racing, n_race, seen):
            uid = p.spec.uid()
            seen.add(uid)
            self.pending[uid] = {
                "origin": RACE,
                "operator": None,
                "parents": [p.extra["pair"]],
                "parent_fitness": None,
            }
            out.append(p)

        n_new = k - len(out)
        pool = self._children(n_new, seen)
        chosen, preds, mode = self.surrogate.select(
            [c[0] for c in pool], n_new, self.rng
        )
        for i in chosen:
            spec, detail, meta = pool[i]
            origin = BOOTSTRAP if meta["operator"] is None else EVOLVE
            self.pending[spec.uid()] = {"origin": origin, **meta}
            extra = {
                **meta,
                "surrogate": preds[i],
                "selection": mode,
                "pool_size": len(pool),
            }
            out.append(Proposal(spec, origin, detail, extra))

        # Pending provenance must survive a crash between proposing and observing.
        self.persist()
        return out
