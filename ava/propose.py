"""Proposer: decides which candidates to try next, and assigns credit.

This is the core of the system. The view parameters are fixed, so a candidate
is a task, one word per searched slot, and a style; that is the whole space.

Credit assignment is per component, not per candidate. One evaluation of a
flip with words (w_1, w_2) in style s yields evidence about three arms:

    p_1 -> (w_1, 'flip', 'subject')    did view 1 read as its prompt?
    p_2 -> (w_2, 'flip', 'subject')    did view 2?
    J   -> (s,   'flip', 'style')      did the pair work at all in this style?

A round of k candidates is split three ways rather than being pure Thompson
sampling. Pure exploitation would converge onto the seed vocabulary and freeze
it; the fixed injection quota keeps new words entering at a bounded rate even
once there are thousands of arms.

    ~50% exploit  Thompson sampling over the existing posteriors
    ~25% inject   an arm that has never been tried, guaranteed; words that
                  only other tasks know count as untried here
    ~25% swap     last round's failures, resampling only the slots that
                  failed (keeping a working component is more sample-efficient
                  than redrawing the whole candidate)

Tasks are visited round-robin within a round, so every configured task gets
its share of the budget; which task deserves more is a question for the search
layer, not for this proposer.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np

from ava.image.tasks import STYLE, IllusionTask, get_task
from ava.spec import CandidateSpec, RunState, Verdict
from ava.vocab import Arm, draw_arm, ensure_arm, list_arms, untried_pool, update_arm

# Origins recorded in candidates.jsonl so a round can be explained afterwards.
EXPLOIT = "exploit"
INJECT = "inject"
SWAP = "swap"
UNIFORM = "uniform"  # validation sweeps only; see UniformProposer

MAX_DRAW_ATTEMPTS = 40

# Words too generic to signal that two views collapsed onto the same subject.
_STOPWORDS = frozenset(
    "a an the of in on at is are there with and to it its this that "
    "painting photo image picture arafed araffe".split()
)


@dataclass(frozen=True)
class ProposalMix:
    """Fractions of a round's budget. Exploit takes whatever is left over."""

    inject: float = 0.25
    swap: float = 0.25

    def counts(self, k: int) -> tuple[int, int, int]:
        n_inject = int(round(k * self.inject))
        n_swap = int(round(k * self.swap))
        n_exploit = k - n_inject - n_swap
        if n_exploit < 0:  # a lopsided mix must not eat the exploit budget
            n_exploit, n_swap = 0, k - n_inject
        return n_exploit, n_inject, n_swap


DEFAULT_MIX = ProposalMix()


@dataclass(frozen=True)
class Proposal:
    """A candidate plus why it was proposed.

    The reason is kept beside the spec rather than inside it: CandidateSpec's
    uid is a content hash used for deduplication, and provenance must not
    change a candidate's identity.
    """

    spec: CandidateSpec
    origin: str
    detail: str
    # Machine-readable provenance (the operator that made it, the surrogate's
    # prediction, ...). Written to candidates.jsonl and scores.jsonl so a
    # search's decisions can be audited against what actually happened.
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "uid": self.spec.uid(),
            "origin": self.origin,
            "detail": self.detail,
            "extra": dict(self.extra),
            "spec": asdict(self.spec),
        }


class Proposer(Protocol):
    def propose(self, state: RunState, k: int) -> list[Proposal]: ...

    def observe(self, state: RunState) -> None:
        """Called once per round, after `state.last_round` holds its results.

        The bandit proposers read `state.last_round` lazily inside `propose`
        and need nothing here; a proposer that keeps its own history (the
        evolutionary one keeps an archive) folds the round in at this point,
        so its persisted state is complete after the final round too.
        """
        ...


# ---------------------------------------------------------------------------
# Credit assignment
# ---------------------------------------------------------------------------


def spec_arms(spec: CandidateSpec) -> tuple[tuple[str, str, str], ...]:
    """The (word, task, role) arms a candidate draws on: searched slots, then style."""
    task = get_task(spec.task)
    arms = [
        (spec.prompts[i], task.name, task.slots[i].role) for i in task.searched_slots()
    ]
    arms.append((spec.style, task.name, STYLE))
    return tuple(arms)


def assign_credit(
    conn: sqlite3.Connection, spec: CandidateSpec, verdict: Verdict
) -> None:
    """Fold one evaluation into every arm the candidate drew on.

    Each signal is routed to the component it actually speaks about. Giving
    every arm the same J would make a good word indistinguishable from the bad
    word it happened to be paired with. The reference slot of an inverse
    problem is not an arm and receives nothing.
    """
    task = get_task(spec.task)
    for i in task.searched_slots():
        update_arm(conn, spec.prompts[i], task.name, task.slots[i].role, verdict.p[i])
    update_arm(conn, spec.style, task.name, STYLE, verdict.j)


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------


def _content_words(caption: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z]+", caption.lower())) - _STOPWORDS


def colliding_views(verdict: Verdict, threshold: float = 0.6) -> tuple[int, int] | None:
    """The first pair of views whose captions describe the same thing, if any.

    A candidate can score well on every view and still be worthless: if one
    object happens to match two prompts, there is no illusion, just an image.
    The BLIP captions are used only for this boolean and never as a number,
    because CLIP's text-to-text similarity is too weak to trust quantitatively.
    """
    words = [_content_words(c) for c in verdict.captions]
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            if not words[i] or not words[j]:
                continue
            if len(words[i] & words[j]) / len(words[i] | words[j]) >= threshold:
                return i, j
    return None


def captions_collide(verdict: Verdict, threshold: float = 0.6) -> bool:
    return colliding_views(verdict, threshold) is not None


def diagnose(verdict: Verdict, collision_threshold: float = 0.6) -> str:
    """Verdict.diagnose(), plus the views-show-the-same-thing case."""
    base = verdict.diagnose()
    if base == "ok" and captions_collide(verdict, collision_threshold):
        return "views_collapsed"
    return base


def repairable(diagnosis: str) -> bool:
    return diagnosis == "views_collapsed" or diagnosis.startswith("lost:")


# ---------------------------------------------------------------------------
# Shared candidate assembly
# ---------------------------------------------------------------------------


class CandidateBuilder:
    """What both proposers share: turning drawn words into a CandidateSpec."""

    def __init__(
        self,
        tasks: Sequence[str],
        screening_seed: int,
        guidance_scale: float,
        num_inference_steps: int,
        ref_image: str | None,
        ref_prompt: str | None,
    ) -> None:
        if not tasks:
            raise ValueError("at least one task is required")
        self.tasks: list[IllusionTask] = [get_task(t) for t in tasks]
        for task in self.tasks:
            if task.ref_slot is not None and (ref_image is None or ref_prompt is None):
                raise ValueError(
                    f"task {task.name!r} pins a slot to a reference image; "
                    "pass ref_image and ref_prompt"
                )
        self.screening_seed = screening_seed
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.ref_image = ref_image
        self.ref_prompt = ref_prompt
        self._task_cursor = 0

    def next_task(self) -> IllusionTask:
        task = self.tasks[self._task_cursor % len(self.tasks)]
        self._task_cursor += 1
        return task

    def reserved(self, task: IllusionTask) -> frozenset[str]:
        """Words no searched slot may take: the reference prompt of an inverse task.

        With the vocabulary pooled across tasks the reference subject is
        drawable like any other word, and a candidate whose free slot repeats
        the reference is not an illusion.
        """
        if task.ref_slot is not None and self.ref_prompt is not None:
            return frozenset({self.ref_prompt})
        return frozenset()

    def build(
        self, task: IllusionTask, words: dict[int, str], style: str
    ) -> CandidateSpec:
        prompts = []
        for i in range(task.n_views):
            if i == task.ref_slot:
                assert self.ref_prompt is not None
                prompts.append(self.ref_prompt)
            else:
                prompts.append(words[i])
        return CandidateSpec(
            task=task.name,
            prompts=tuple(prompts),
            style=style,
            seed=self.screening_seed,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            ref_image=self.ref_image if task.ref_slot is not None else None,
        )


def _words_of(spec: CandidateSpec, task: IllusionTask) -> dict[int, str]:
    return {i: spec.prompts[i] for i in task.searched_slots()}


# ---------------------------------------------------------------------------
# Bandit proposer
# ---------------------------------------------------------------------------


class BanditProposer:
    """Thompson sampling over (word, task, role) arms, with injection and swaps.

    Dependencies are injected: the database connection, the RNG (so a run is
    reproducible from its seed) and, optionally, a semantic distance over
    prompt strings used to push apart components that collapsed onto the same
    subject. Without that callable the collapsed case falls back to redrawing
    the offending slot, which is correct but less directed.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        rng: np.random.Generator,
        tasks: Sequence[str] = ("hybrid",),
        mix: ProposalMix | None = None,
        screening_seed: int = 0,
        guidance_scale: float = 10.0,
        num_inference_steps: int = 30,
        text_distance: Callable[[str, str], float] | None = None,
        ref_image: str | None = None,
        ref_prompt: str | None = None,
    ) -> None:
        self.conn = conn
        self.rng = rng
        self.mix = mix if mix is not None else DEFAULT_MIX
        # One diffusion seed for every screening candidate. Holding the noise
        # fixed means a difference between two candidates is a difference
        # between prompts, not between noise draws. Seed luck is dealt with at
        # harvest time, where the survivors are regenerated across many seeds.
        self.builder = CandidateBuilder(
            tasks,
            screening_seed,
            guidance_scale,
            num_inference_steps,
            ref_image,
            ref_prompt,
        )
        self.text_distance = text_distance

    @property
    def tasks(self) -> list[IllusionTask]:
        return self.builder.tasks

    def observe(self, state: RunState) -> None:
        """Nothing to fold in: swaps read `state.last_round` at proposal time."""

    # -- helpers ---------------------------------------------------------

    def _draw(
        self, task: IllusionTask, role: str, exclude: frozenset[str] = frozenset()
    ) -> Arm:
        # Over the pooled vocabulary: every task can try every word.
        return draw_arm(self.conn, task.name, role, self.rng, exclude=exclude)

    def _fill(self, task: IllusionTask, fixed: dict[int, str]) -> dict[int, str]:
        """Draw a word for every searched slot not already fixed, all distinct."""
        words = dict(fixed)
        for i in task.searched_slots():
            if i in words:
                continue
            taken = frozenset(words.values()) | self.builder.reserved(task)
            words[i] = self._draw(task, task.slots[i].role, exclude=taken).word
        return words

    def _accept(self, spec: CandidateSpec, seen: set[str], state: RunState) -> bool:
        uid = spec.uid()
        if uid in seen or uid in state.evaluated_uids:
            return False
        seen.add(uid)
        return True

    def _distant_alternative(self, task: IllusionTask, role: str, avoid: str) -> str:
        """Pick a word for `role` semantically far from `avoid`.

        Used when two views collapsed onto one subject. Falls back to plain
        Thompson sampling when no distance function was supplied.
        """
        distance = self.text_distance
        exclude = frozenset({avoid}) | self.builder.reserved(task)
        if distance is None:
            return self._draw(task, role, exclude=exclude).word
        candidates = [self._draw(task, role, exclude=exclude).word for _ in range(8)]
        return max(candidates, key=lambda w: distance(avoid, w))

    # -- the three proposal kinds ---------------------------------------

    def _exploit_one(self) -> tuple[CandidateSpec, str]:
        task = self.builder.next_task()
        words = self._fill(task, {})
        style = self._draw(task, STYLE)
        detail = f"thompson task={task.name} " + " ".join(
            f"{task.slots[i].name}={words[i]!r}" for i in sorted(words)
        )
        return self.builder.build(
            task, words, style.word
        ), detail + f" style={style.word!r}"

    def _inject_one(
        self, pool: dict[tuple[str, str], list[Arm]]
    ) -> tuple[CandidateSpec, str] | None:
        """Build a candidate around an arm that has never been tried.

        (task, role) pairs with untried arms are chosen at random so injection
        does not starve one while another has a long backlog.
        """
        keys = [k for k, arms in pool.items() if arms]
        if not keys:
            return None
        task_name, role = keys[int(self.rng.integers(len(keys)))]
        arms = pool[(task_name, role)]
        arm = arms.pop(int(self.rng.integers(len(arms))))
        ensure_arm(self.conn, arm)  # a word from another task becomes an arm here
        task = get_task(task_name)

        if role == STYLE:
            spec = self.builder.build(task, self._fill(task, {}), arm.word)
        else:
            slot = next(i for i in task.searched_slots() if task.slots[i].role == role)
            words = self._fill(task, {slot: arm.word})
            spec = self.builder.build(task, words, self._draw(task, STYLE).word)
        detail = (
            f"inject untried ({arm.word!r}, {task.name}, {role}) source={arm.source}"
        )
        return spec, detail

    def _swap_one(
        self, spec: CandidateSpec, verdict: Verdict
    ) -> tuple[CandidateSpec, str] | None:
        """Redraw only the slots the diagnosis blames."""
        task = get_task(spec.task)
        d = diagnose(verdict)
        words = _words_of(spec, task)

        if d.startswith("lost:"):
            lost = set(d.removeprefix("lost:").split(","))
            redraw = [i for i in task.searched_slots() if task.slots[i].name in lost]
            if not redraw:  # only the reference slot failed; nothing to redraw
                return None
            kept = {i: w for i, w in words.items() if i not in redraw}
            new = self._fill(task, kept)
            changed = ", ".join(
                f"{task.slots[i].name} {words[i]!r}->{new[i]!r} (p={verdict.p[i]:.3f})"
                for i in redraw
            )
            return self.builder.build(task, new, spec.style), f"swap {changed}"

        if d == "views_collapsed":
            pair = colliding_views(verdict)
            assert pair is not None
            i, j = pair
            if j not in words:  # the later view is the reference slot; redraw the other
                i, j = j, i
            if j not in words:
                return None
            replacement = self._distant_alternative(task, task.slots[j].role, words[i])
            new = dict(words)
            new[j] = replacement
            return (
                self.builder.build(task, new, spec.style),
                f"swap {task.slots[j].name} {words[j]!r}->{replacement!r} "
                "(both views read alike)",
            )
        # 'all_lost' discards the candidate outright; 'ok' needs no repair.
        return None

    # -- entry point -----------------------------------------------------

    def propose(self, state: RunState, k: int) -> list[Proposal]:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        n_exploit, n_inject, n_swap = self.mix.counts(k)
        seen: set[str] = set()
        out: list[Proposal] = []

        # Injection first: it is the quota most easily crowded out, and any
        # shortfall should be absorbed by exploitation rather than the reverse.
        pool: dict[tuple[str, str], list[Arm]] = {}
        for task in self.tasks:
            roles = {task.slots[i].role for i in task.searched_slots()} | {STYLE}
            for role in sorted(roles):
                pool[(task.name, role)] = [
                    a
                    for a in untried_pool(self.conn, task.name, role)
                    if a.word not in self.builder.reserved(task)
                ]
        for _ in range(n_inject):
            for _ in range(MAX_DRAW_ATTEMPTS):
                built = self._inject_one(pool)
                if built is None:
                    break
                spec, detail = built
                if self._accept(spec, seen, state):
                    out.append(Proposal(spec, INJECT, detail))
                    break

        fixable = [(s, v) for s, v in state.last_round if repairable(diagnose(v))]
        for spec_prev, verdict in fixable[:n_swap]:
            for _ in range(MAX_DRAW_ATTEMPTS):
                built = self._swap_one(spec_prev, verdict)
                if built is None:
                    break
                spec, detail = built
                if self._accept(spec, seen, state):
                    out.append(Proposal(spec, SWAP, detail))
                    break

        # Exploitation backfills whatever the other two quotas could not use,
        # so a round always returns k candidates when the vocabulary allows.
        while len(out) < k:
            filled = False
            for _ in range(MAX_DRAW_ATTEMPTS):
                spec, detail = self._exploit_one()
                if self._accept(spec, seen, state):
                    out.append(Proposal(spec, EXPLOIT, detail))
                    filled = True
                    break
            if not filled:
                # The vocabulary is exhausted relative to what has been tried.
                break
        return out


class UniformProposer:
    """Samples arms uniformly instead of by posterior. For validation, not search.

    Thompson sampling is doing its job when it stops drawing an arm whose
    posterior looks bad, but that makes it useless for checking whether the
    credit assignment reproduces a claim about a bad component: the arm never
    gets the trials the check needs. After 40 bandit-chosen candidates
    `a photo of` had been drawn once, so its posterior was still essentially
    the prior it started with.

    Sampling uniformly gives every arm comparable evidence, so a ranking taken
    afterwards reflects measurements rather than priors. Use it to validate the
    pipeline, then go back to BanditProposer to actually search -- uniform
    sampling wastes most of its budget on components already known to fail.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        rng: np.random.Generator,
        tasks: Sequence[str] = ("hybrid",),
        screening_seed: int = 0,
        guidance_scale: float = 10.0,
        num_inference_steps: int = 30,
        roles_to_balance: tuple[str, ...] = (STYLE,),
        ref_image: str | None = None,
        ref_prompt: str | None = None,
    ) -> None:
        self.conn = conn
        self.rng = rng
        self.builder = CandidateBuilder(
            tasks,
            screening_seed,
            guidance_scale,
            num_inference_steps,
            ref_image,
            ref_prompt,
        )
        # Round-robin over these roles so their arms get equal n, rather than
        # merely equal sampling probability, which at n=40 is not the same thing.
        self.roles_to_balance = roles_to_balance
        self._cursor: dict[tuple[str, str], int] = {}

    def observe(self, state: RunState) -> None:
        """A uniform sweep learns nothing from results, by design."""

    def _pick(
        self, task: IllusionTask, role: str, exclude: frozenset[str] = frozenset()
    ) -> str:
        arms = [
            a for a in list_arms(self.conn, task.name, role) if a.word not in exclude
        ]
        if not arms:
            raise LookupError(f"no available arms for ({task.name!r}, {role!r})")
        return arms[int(self.rng.integers(len(arms)))].word

    def _balanced(self, task: IllusionTask, role: str) -> str:
        """Cycle through a role's arms so every one is measured equally often."""
        arms = sorted(list_arms(self.conn, task.name, role), key=lambda a: a.word)
        if not arms:
            raise LookupError(f"no available arms for ({task.name!r}, {role!r})")
        cursor = self._cursor.get((task.name, role), 0)
        return arms[cursor % len(arms)].word

    def _advance(self, task: IllusionTask) -> None:
        for role in self.roles_to_balance:
            key = (task.name, role)
            self._cursor[key] = self._cursor.get(key, 0) + 1

    def _choose(self, task: IllusionTask, role: str, exclude: frozenset[str]) -> str:
        if role in self.roles_to_balance:
            return self._balanced(task, role)
        return self._pick(task, role, exclude)

    def propose(self, state: RunState, k: int) -> list[Proposal]:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        seen: set[str] = set()
        out: list[Proposal] = []
        while len(out) < k:
            task = self.builder.next_task()
            for _ in range(MAX_DRAW_ATTEMPTS):
                words: dict[int, str] = {}
                for i in task.searched_slots():
                    taken = frozenset(words.values()) | self.builder.reserved(task)
                    words[i] = self._choose(task, task.slots[i].role, taken)
                style = self._choose(task, STYLE, frozenset())
                spec = self.builder.build(task, words, style)
                uid = spec.uid()
                if uid in seen or uid in state.evaluated_uids:
                    continue
                seen.add(uid)
                self._advance(task)
                detail = f"uniform task={task.name} " + " ".join(
                    f"{task.slots[i].name}={words[i]!r}" for i in sorted(words)
                )
                out.append(Proposal(spec, UNIFORM, f"{detail} style={style!r}"))
                break
            else:
                break
        return out
