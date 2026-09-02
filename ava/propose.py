"""Proposer: decides which prompt triples to try next, and assigns credit.

This is the core of the system. Sigma is fixed, so the prompt triple
(low, high, style) is the entire search space.

Credit assignment is per component, not per candidate. One evaluation of
(low_i, high_j, style_k) yields evidence about three separate arms:

    p_far -> (low_i, 'low')     did it survive blurring?
    p_near -> (high_j, 'high')  could it be read from fine detail?
    J     -> (style_k, 'style') did the pair work at all in this style?

A round of k candidates is split three ways rather than being pure Thompson
sampling. Pure exploitation would converge onto the seed vocabulary and freeze
it; the fixed injection quota keeps new words entering at a bounded rate even
once there are thousands of arms.

    ~50% exploit  Thompson sampling over the existing posteriors
    ~25% inject   an arm that has never been tried, guaranteed
    ~25% swap     last round's failures, resampling only the component that
                  failed (keeping a working component is more sample-efficient
                  than redrawing the whole triple)

Role swapping needs no special case: (word, 'low') and (word, 'high') are
independent arms, so Thompson sampling tries both on its own.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ava.spec import CandidateSpec, RunState, Verdict
from ava.vocab import (
    ROLES,
    Arm,
    Role,
    list_arms,
    thompson_sample,
    untried_arms,
    update_arm,
)

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

    def to_dict(self) -> dict[str, object]:
        from dataclasses import asdict

        return {
            "uid": self.spec.uid(),
            "origin": self.origin,
            "detail": self.detail,
            "spec": asdict(self.spec),
        }


class Proposer(Protocol):
    def propose(self, state: RunState, k: int) -> list[Proposal]: ...


# ---------------------------------------------------------------------------
# Credit assignment
# ---------------------------------------------------------------------------


def spec_arms(spec: CandidateSpec) -> tuple[tuple[str, Role], ...]:
    """The three arms a candidate draws on."""
    return ((spec.prompt_low, "low"), (spec.prompt_high, "high"), (spec.style, "style"))


def assign_credit(
    conn: sqlite3.Connection, spec: CandidateSpec, verdict: Verdict
) -> None:
    """Fold one evaluation into all three arms' posteriors.

    Each signal is routed to the component it actually speaks about. Giving
    every arm the same J would make a good low word indistinguishable from the
    bad high word it happened to be paired with.
    """
    update_arm(conn, spec.prompt_low, "low", verdict.p_far)
    update_arm(conn, spec.prompt_high, "high", verdict.p_near)
    update_arm(conn, spec.style, "style", verdict.j)


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------


def _content_words(caption: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z]+", caption.lower())) - _STOPWORDS


def captions_collide(verdict: Verdict, threshold: float = 0.6) -> bool:
    """True when both views appear to show the same subject.

    A candidate can score well on both views and still be worthless: if one
    object happens to match both prompts, there is no illusion, just an image.
    The BLIP captions are used only as this boolean flag and never as a number,
    because CLIP's text-to-text similarity is too weak to trust quantitatively.
    """
    far = _content_words(verdict.caption_far)
    near = _content_words(verdict.caption_near)
    if not far or not near:
        return False
    return len(far & near) / len(far | near) >= threshold


def diagnose(verdict: Verdict, collision_threshold: float = 0.6) -> str:
    """Verdict.diagnose(), plus the both-views-show-the-same-thing case."""
    base = verdict.diagnose()
    if base == "ok" and captions_collide(verdict, collision_threshold):
        return "views_collapsed"
    return base


# ---------------------------------------------------------------------------
# Bandit proposer
# ---------------------------------------------------------------------------


class BanditProposer:
    """Thompson sampling over (word, role) arms, with injection and swaps.

    Dependencies are injected: the database connection, the RNG (so a run is
    reproducible from its seed) and, optionally, a semantic distance over
    prompt strings used to push apart components that collapsed onto the same
    subject. Without that callable the collapsed case falls back to redrawing
    both components, which is correct but less directed.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        rng: np.random.Generator,
        mix: ProposalMix | None = None,
        screening_seed: int = 0,
        guidance_scale: float = 10.0,
        num_inference_steps: int = 30,
        text_distance: Callable[[str, str], float] | None = None,
    ) -> None:
        self.conn = conn
        self.rng = rng
        self.mix = mix if mix is not None else DEFAULT_MIX
        # One diffusion seed for every screening candidate. Holding the noise
        # fixed means a difference between two candidates is a difference
        # between prompts, not between noise draws. Seed luck is dealt with at
        # harvest time, where the survivors are regenerated across many seeds.
        self.screening_seed = screening_seed
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.text_distance = text_distance

    # -- helpers ---------------------------------------------------------

    def _build(self, low: str, high: str, style: str) -> CandidateSpec:
        return CandidateSpec(
            prompt_low=low,
            prompt_high=high,
            style=style,
            seed=self.screening_seed,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
        )

    def _draw(self, role: Role, exclude: frozenset[str] = frozenset()) -> Arm:
        return thompson_sample(self.conn, role, self.rng, exclude=exclude)

    def _accept(self, spec: CandidateSpec, seen: set[str], state: RunState) -> bool:
        uid = spec.uid()
        if uid in seen or uid in state.evaluated_uids:
            return False
        seen.add(uid)
        return True

    def _distant_alternative(self, role: Role, avoid: str) -> str:
        """Pick a component semantically far from `avoid`.

        Used when both views collapsed onto one subject. Falls back to plain
        Thompson sampling when no distance function was supplied.
        """
        distance = self.text_distance
        if distance is None:
            return self._draw(role, exclude=frozenset({avoid})).word
        candidates = [
            self._draw(role, exclude=frozenset({avoid})).word for _ in range(8)
        ]
        return max(candidates, key=lambda w: distance(avoid, w))

    # -- the three proposal kinds ---------------------------------------

    def _exploit_one(self) -> tuple[CandidateSpec, str]:
        low = self._draw("low")
        high = self._draw("high", exclude=frozenset({low.word}))
        style = self._draw("style")
        detail = (
            f"thompson low={low.word!r}(n={low.n_trials}) "
            f"high={high.word!r}(n={high.n_trials}) style={style.word!r}"
        )
        return self._build(low.word, high.word, style.word), detail

    def _inject_one(
        self, pool: dict[Role, list[Arm]]
    ) -> tuple[CandidateSpec, str] | None:
        """Build a candidate around an arm that has never been tried.

        Roles with untried arms are chosen at random so injection does not
        starve one role while another has a long backlog.
        """
        roles: list[Role] = [r for r in ROLES if pool[r]]
        if not roles:
            return None
        role: Role = roles[int(self.rng.integers(len(roles)))]
        arm = pool[role].pop(int(self.rng.integers(len(pool[role]))))

        if role == "low":
            spec = self._build(
                arm.word,
                self._draw("high", exclude=frozenset({arm.word})).word,
                self._draw("style").word,
            )
        elif role == "high":
            spec = self._build(
                self._draw("low", exclude=frozenset({arm.word})).word,
                arm.word,
                self._draw("style").word,
            )
        else:
            low = self._draw("low")
            spec = self._build(
                low.word,
                self._draw("high", exclude=frozenset({low.word})).word,
                arm.word,
            )
        return spec, f"inject untried ({arm.word!r}, {role!r}) source={arm.source}"

    def _swap_one(
        self, spec: CandidateSpec, verdict: Verdict
    ) -> tuple[CandidateSpec, str] | None:
        """Redraw only the component the diagnosis blames."""
        d = diagnose(verdict)
        if d == "low_loses":
            new = self._draw("low", exclude=frozenset({spec.prompt_low})).word
            return (
                self._build(new, spec.prompt_high, spec.style),
                f"swap low {spec.prompt_low!r}->{new!r} (p_far={verdict.p_far:.3f})",
            )
        if d == "high_absent":
            new = self._draw("high", exclude=frozenset({spec.prompt_high})).word
            return (
                self._build(spec.prompt_low, new, spec.style),
                f"swap high {spec.prompt_high!r}->{new!r} "
                f"(p_near={verdict.p_near:.3f})",
            )
        if d == "views_collapsed":
            new = self._distant_alternative("high", spec.prompt_high)
            return (
                self._build(spec.prompt_low, new, spec.style),
                f"swap high {spec.prompt_high!r}->{new!r} (both views read alike)",
            )
        # 'pair_mismatch' discards the triple outright; 'ok' needs no repair.
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
        pool: dict[Role, list[Arm]] = {r: untried_arms(self.conn, r) for r in ROLES}
        for _ in range(n_inject):
            for _ in range(MAX_DRAW_ATTEMPTS):
                built = self._inject_one(pool)
                if built is None:
                    break
                spec, detail = built
                if self._accept(spec, seen, state):
                    out.append(Proposal(spec, INJECT, detail))
                    break

        repairable = [
            (s, v)
            for s, v in state.last_round
            if diagnose(v) in ("low_loses", "high_absent", "views_collapsed")
        ]
        for spec_prev, verdict in repairable[:n_swap]:
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
        screening_seed: int = 0,
        guidance_scale: float = 10.0,
        num_inference_steps: int = 30,
        roles_to_balance: tuple[Role, ...] = ("style",),
    ) -> None:
        self.conn = conn
        self.rng = rng
        self.screening_seed = screening_seed
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        # Round-robin over these roles so their arms get equal n, rather than
        # merely equal sampling probability, which at n=40 is not the same thing.
        self.roles_to_balance = roles_to_balance
        self._cursor = 0

    def _pick(self, role: Role, exclude: frozenset[str] = frozenset()) -> str:
        arms = [a for a in list_arms(self.conn, role) if a.word not in exclude]
        if not arms:
            raise LookupError(f"no available arms for role {role!r}")
        return arms[int(self.rng.integers(len(arms)))].word

    def _balanced(self, role: Role) -> str:
        """Cycle through a role's arms so every one is measured equally often."""
        arms = sorted(list_arms(self.conn, role), key=lambda a: a.word)
        if not arms:
            raise LookupError(f"no available arms for role {role!r}")
        word = arms[self._cursor % len(arms)].word
        return word

    def propose(self, state: RunState, k: int) -> list[Proposal]:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        seen: set[str] = set()
        out: list[Proposal] = []
        while len(out) < k:
            for _ in range(MAX_DRAW_ATTEMPTS):
                low = self._pick("low")
                high = self._pick("high", exclude=frozenset({low}))
                style = (
                    self._balanced("style")
                    if "style" in self.roles_to_balance
                    else self._pick("style")
                )
                spec = CandidateSpec(
                    prompt_low=low,
                    prompt_high=high,
                    style=style,
                    seed=self.screening_seed,
                    guidance_scale=self.guidance_scale,
                    num_inference_steps=self.num_inference_steps,
                )
                uid = spec.uid()
                if uid in seen or uid in state.evaluated_uids:
                    continue
                seen.add(uid)
                self._cursor += 1
                out.append(
                    Proposal(
                        spec,
                        UNIFORM,
                        f"uniform low={low!r} high={high!r} style={style!r}",
                    )
                )
                break
            else:
                break
        return out
