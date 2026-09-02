"""Mutation operators over the vocabulary, and the bandit that picks among them.

Every operator takes a parent individual and returns one child candidate, or
None when it does not apply (a crossover with nobody to cross with, an
injection with nothing untried left). Children are built through the same
`CandidateBuilder` the v1 proposer uses, so the diffusion seed, guidance and
step count are the run's screening settings whatever operator made the child.

Words come out of the vocabulary database by Thompson sampling, which is how
the v1 component posteriors keep steering v2: the archive decides *which pair*
to mutate, the posteriors decide *which word* to try in its place.

Operator choice is a UCB1 bandit rewarded by the child's improvement over its
parent, so an operator that keeps producing better children gets used more,
and one that never does is not abandoned but visited at the UCB rate.
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ava.image.tasks import STYLE, IllusionTask, get_task
from ava.propose import CandidateBuilder
from ava.search.archive import Archive, Individual, pair_key
from ava.spec import CandidateSpec
from ava.vocab import add_arm, thompson_sample, untried_arms

SWAP_SLOT = "swap_slot"
CROSSOVER = "crossover"
RESTYLE = "restyle"
TRANSPOSE = "transpose"
TRANSFER_TASK = "transfer_task"
INJECT = "inject"
OPERATOR_NAMES: tuple[str, ...] = (
    SWAP_SLOT,
    CROSSOVER,
    RESTYLE,
    TRANSPOSE,
    TRANSFER_TASK,
    INJECT,
)

# Provenance of arms that transfer_task registers in the target task.
TRANSFER_SOURCE = "transfer"

Child = tuple[CandidateSpec, str]  # the candidate and a one-line explanation


@dataclass
class OperatorContext:
    """What an operator may touch: the vocabulary, the RNG, the builder, the archive."""

    conn: sqlite3.Connection
    rng: np.random.Generator
    builder: CandidateBuilder
    archive: Archive

    @property
    def tasks(self) -> list[IllusionTask]:
        return self.builder.tasks


Operator = Callable[[Individual, OperatorContext], "Child | None"]


def _words(parent: Individual, task: IllusionTask) -> dict[int, str]:
    return {i: parent.spec.prompts[i] for i in task.searched_slots()}


def _draw(
    ctx: OperatorContext, task: IllusionTask, role: str, exclude: frozenset[str]
) -> str | None:
    try:
        return thompson_sample(ctx.conn, task.name, role, ctx.rng, exclude=exclude).word
    except LookupError:
        return None


def _describe(task: IllusionTask, words: dict[int, str]) -> str:
    return " ".join(f"{task.slots[i].name}={words[i]!r}" for i in sorted(words))


def _register(
    ctx: OperatorContext,
    task: IllusionTask,
    words: dict[int, str],
    style: str,
    citation: str,
) -> None:
    """Make sure every (word, task, role) the child draws on is an arm.

    An operator that moves a word into a role it was never registered for
    (a transfer to another task, a transpose of an asymmetric task) would
    otherwise leave credit assignment with nowhere to put the evidence.
    Existing arms are untouched; new ones carry `source='transfer'`.
    """
    for i, word in words.items():
        add_arm(
            ctx.conn,
            word,
            task.name,
            task.slots[i].role,
            TRANSFER_SOURCE,
            citation=citation,
        )
    add_arm(ctx.conn, style, task.name, STYLE, TRANSFER_SOURCE, citation=citation)


# ---------------------------------------------------------------------------
# The six operators
# ---------------------------------------------------------------------------


def swap_slot(parent: Individual, ctx: OperatorContext) -> Child | None:
    """Redraw the slot that failed, or the weakest one if every view held.

    The v1 targeted swap, applied to an archived parent instead of last
    round's failures: keep what worked, redraw what did not.
    """
    task = get_task(parent.spec.task)
    words = _words(parent, task)
    latest = parent.latest
    lost = [i for i in task.searched_slots() if task.slots[i].name in latest.lost]
    if lost:
        i = lost[int(ctx.rng.integers(len(lost)))]
    else:
        i = min(task.searched_slots(), key=lambda j: latest.sep[j])
    new = _draw(ctx, task, task.slots[i].role, frozenset(words.values()))
    if new is None:
        return None
    old, words[i] = words[i], new
    return (
        ctx.builder.build(task, words, parent.spec.style),
        f"swap_slot {task.slots[i].name} {old!r}->{new!r} (sep={latest.sep[i]:+.3f})",
    )


def crossover(parent: Individual, ctx: OperatorContext) -> Child | None:
    """One slot from the parent, the others from another elite of the same task."""
    task = get_task(parent.spec.task)
    searched = task.searched_slots()
    if len(searched) < 2:
        return None
    partners = [
        e for e in ctx.archive.elite_individuals(task.name) if e.key != parent.key
    ]
    if not partners:
        return None
    partner = partners[int(ctx.rng.integers(len(partners)))]
    keep = searched[int(ctx.rng.integers(len(searched)))]
    words = {j: partner.spec.prompts[j] for j in searched}
    words[keep] = parent.spec.prompts[keep]
    if len(set(words.values())) < len(words):
        return None
    child = ctx.builder.build(task, words, parent.spec.style)
    if pair_key(child) in (parent.key, partner.key):
        return None
    return child, (
        f"crossover keep {task.slots[keep].name} from parent, rest from "
        f"{partner.key_str}: {_describe(task, words)}"
    )


def restyle(parent: Individual, ctx: OperatorContext) -> Child | None:
    """Same prompts, a different style drawn from the task's style posteriors."""
    task = get_task(parent.spec.task)
    style = _draw(ctx, task, STYLE, frozenset({parent.spec.style}))
    if style is None:
        return None
    return (
        ctx.builder.build(task, _words(parent, task), style),
        f"restyle {parent.spec.style!r}->{style!r}",
    )


def transpose(parent: Individual, ctx: OperatorContext) -> Child | None:
    """Permute the prompts across the searched slots.

    For an asymmetric task this asks whether the pair works with the roles
    reversed (the low-frequency subject as the high-frequency one). For a
    symmetric task it is the same illusion with the views exchanged, which is
    a different generation and a different image, but lands in the same cell.
    """
    task = get_task(parent.spec.task)
    searched = task.searched_slots()
    if len(searched) < 2 or task.ref_slot is not None:
        return None
    perm = np.arange(len(searched))
    if len(searched) == 2:
        perm = perm[::-1]
    else:
        while np.array_equal(perm, np.arange(len(searched))):
            perm = ctx.rng.permutation(len(searched))
    words = {
        searched[a]: parent.spec.prompts[searched[int(perm[a])]]
        for a in range(len(searched))
    }
    if not task.symmetric:
        _register(ctx, task, words, parent.spec.style, f"transposed within {task.name}")
    return (
        ctx.builder.build(task, words, parent.spec.style),
        f"transpose {_describe(task, words)}",
    )


def transfer_task(parent: Individual, ctx: OperatorContext) -> Child | None:
    """Keep the pair, change the view: does a flip that works also work as a jigsaw?

    The words may not be registered under the target task yet; they are added
    with `source='transfer'` so credit assignment finds an arm to update and
    the component table records that the word arrived by transfer.
    """
    task = get_task(parent.spec.task)
    if task.ref_slot is not None:
        return None
    targets = [
        t
        for t in ctx.tasks
        if t.name != task.name and t.n_views == task.n_views and t.ref_slot is None
    ]
    if not targets:
        return None
    target = targets[int(ctx.rng.integers(len(targets)))]
    words = {i: parent.spec.prompts[i] for i in target.searched_slots()}
    _register(ctx, target, words, parent.spec.style, f"transferred from {task.name}")
    return (
        ctx.builder.build(target, words, parent.spec.style),
        f"transfer_task {task.name}->{target.name}",
    )


def inject(parent: Individual, ctx: OperatorContext) -> Child | None:
    """Put a never-tried word into one slot of the parent."""
    task = get_task(parent.spec.task)
    words = _words(parent, task)
    order = [
        task.searched_slots()[int(j)]
        for j in ctx.rng.permutation(len(task.searched_slots()))
    ]
    for i in order:
        pool = [
            a
            for a in untried_arms(ctx.conn, task.name, task.slots[i].role)
            if a.word not in words.values()
        ]
        if not pool:
            continue
        pool.sort(key=lambda a: a.word)
        arm = pool[int(ctx.rng.integers(len(pool)))]
        old, words[i] = words[i], arm.word
        return (
            ctx.builder.build(task, words, parent.spec.style),
            f"inject {task.slots[i].name} {old!r}->{arm.word!r} (source={arm.source})",
        )
    return None


OPERATORS: dict[str, Operator] = {
    SWAP_SLOT: swap_slot,
    CROSSOVER: crossover,
    RESTYLE: restyle,
    TRANSPOSE: transpose,
    TRANSFER_TASK: transfer_task,
    INJECT: inject,
}


# ---------------------------------------------------------------------------
# Reward and the operator bandit
# ---------------------------------------------------------------------------


@dataclass
class RunningStats:
    """Welford's running mean and standard deviation, for normalising fitness."""

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {"n": self.n, "mean": self.mean, "m2": self.m2}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunningStats:
        return cls(n=int(d["n"]), mean=float(d["mean"]), m2=float(d["m2"]))


def improvement_reward(
    child_fitness: float, parent_fitness: float, scale: float, floor: float = 0.02
) -> float:
    """ShinkaEvolve's `exp(max(dF, 0)) - 1`, with dF in units of the fitness spread.

    The raw reward is unbounded, and UCB1's exploration term assumes rewards in
    [0, 1], so it is squashed through r / (1 + r). `scale` is the running
    standard deviation of `sep_min`; `floor` stops the first few rounds, when
    that deviation is still near zero, from turning a hair's improvement into
    a large reward.
    """
    delta = max(child_fitness - parent_fitness, 0.0) / max(scale, floor)
    r = math.exp(delta) - 1.0
    return r / (1.0 + r)


@dataclass
class OperatorBandit:
    """UCB1 over the operators. Untried operators are drawn first."""

    names: tuple[str, ...] = OPERATOR_NAMES
    c: float = 1.0
    counts: dict[str, int] = field(default_factory=dict)
    totals: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in self.names:
            self.counts.setdefault(name, 0)
            self.totals.setdefault(name, 0.0)

    def mean(self, name: str) -> float:
        return self.totals[name] / self.counts[name] if self.counts[name] else 0.0

    def choose(self, rng: np.random.Generator) -> str:
        untried = [n for n in self.names if self.counts[n] == 0]
        if untried:
            return untried[int(rng.integers(len(untried)))]
        total = sum(self.counts.values())
        scores = {
            n: self.mean(n) + self.c * math.sqrt(2.0 * math.log(total) / self.counts[n])
            for n in self.names
        }
        return max(self.names, key=lambda n: scores[n])

    def update(self, name: str, reward: float) -> None:
        if name not in self.counts:
            raise KeyError(f"unknown operator {name!r}")
        self.counts[name] += 1
        self.totals[name] += reward

    def to_dict(self) -> dict[str, Any]:
        return {
            "c": self.c,
            "counts": dict(self.counts),
            "totals": dict(self.totals),
            "means": {n: self.mean(n) for n in self.names},
        }

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], names: Sequence[str] = OPERATOR_NAMES
    ) -> OperatorBandit:
        return cls(
            names=tuple(names),
            c=float(d.get("c", 1.0)),
            counts={n: int(v) for n, v in d["counts"].items()},
            totals={n: float(v) for n, v in d["totals"].items()},
        )
