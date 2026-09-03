"""Data structures for search candidates and their evaluation results.

A candidate names an illusion task (which views the image must satisfy), one
prompt per view, and a style. The view parameters below are module constants
rather than CandidateSpec fields: nothing searches over them, and keeping them
out of the spec states that fact in code. They are the upstream defaults from
dev/visual_anagrams, which are also the values the papers report.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

# Hybrid images (Factorized Diffusion Sec. 3.4): Gaussian blur at the 64 px
# stage; both views must receive the same value or lp(e1) + hp(e2) is not an
# identity decomposition.
SIGMA = 2.0
KERNEL_SIZE = 33

# Triple hybrids: a two-level Laplacian pyramid. Upstream defaults; the paper
# reports sigma_1 in [0.8, 1.0] and sigma_2 in [1.2, 2.0] (App. A.3).
TRIPLE_SIGMA_1 = 1.0
TRIPLE_SIGMA_2 = 2.0
TRIPLE_KERNEL_SIZE = 25

# Motion hybrids: diagonal line kernel, 7 px at 64 px, which the upstream code
# scales to the paper's 29 px at 256 px.
MOTION_SIZE = 7

# Visual Anagrams views with parameters.
SKEW_FACTOR = 1.5
PATCH_GRID = 8  # patch_permute: 8x8 patches
PIXEL_GRID = 64  # pixel_permute: every pixel of the 64 px stage
JIGSAW_SEED = 4522  # upstream's fixed jigsaw permutation


def apply_style(style: str, prompt: str) -> str:
    """Combine a style with a subject.

    A style containing `{}` is a template (`"oil painting style, {}"`, as the
    Factorized Diffusion figures write it); anything else is a prefix, assembled
    exactly as generate.py does with `f'{args.style} {p}'.strip()`.
    """
    if "{}" in style:
        return style.replace("{}", prompt).strip()
    return f"{style} {prompt}".strip()


@dataclass(frozen=True)
class CandidateSpec:
    """A single point in the search space.

    `prompts` is ordered to match the task's views. `ref_image` is only set for
    inverse problems, where one view is pinned to a reference image; the prompt
    in that slot then describes the reference for the judge and is not fed to
    the generator.
    """

    task: str
    prompts: tuple[str, ...]
    style: str = ""  # prepended to (or wrapped around) every prompt
    seed: int = 0
    guidance_scale: float = 10.0
    num_inference_steps: int = 30
    ref_image: str | None = None

    def __post_init__(self) -> None:
        # JSON round-trips deliver a list; the uid must not depend on that.
        object.__setattr__(self, "prompts", tuple(self.prompts))
        if len(self.prompts) < 2:
            raise ValueError(
                f"an illusion needs at least two prompts, got {self.prompts}"
            )

    @property
    def n_views(self) -> int:
        return len(self.prompts)

    def full_prompt(self, index: int) -> str:
        return apply_style(self.style, self.prompts[index])

    @property
    def full_prompts(self) -> list[str]:
        """Ordered to match the views."""
        return [self.full_prompt(i) for i in range(self.n_views)]

    def uid(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


@dataclass
class Verdict:
    """Evaluation of one candidate.

    Every intermediate quantity is kept so thresholds can be re-drawn later
    without regenerating images.
    """

    uid: str
    task: str
    slots: list[str]
    # CLIP score matrix S[view][prompt]; both axes in slot order.
    scores: list[list[float]]
    # Probability that each view picks its own prompt among the N prompts.
    p: list[float]
    # Illusion score. min, not mean: an illusion only holds if every view holds.
    j: float
    # The two scores the Visual Anagrams paper reports: min of the diagonal, and
    # the both-directions softmax trace (Eq. 9).
    alignment: float
    concealment: float
    # Never used as a numeric signal. Kept as report evidence and as future
    # VLM input. One caption per view.
    captions: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @property
    def n_views(self) -> int:
        return len(self.slots)

    @property
    def sep(self) -> list[float]:
        """Raw CLIP margin per view: its own prompt minus the best other prompt.

        Negative means the view read as some other prompt. For two views these
        are the old `sep_far` and `sep_near`.
        """
        out = []
        for i, row in enumerate(self.scores):
            others = [s for j, s in enumerate(row) if j != i]
            out.append(row[i] - max(others))
        return out

    @property
    def sep_min(self) -> float:
        """Ranking key: the weakest view's raw margin.

        With two views a two-way softmax is a sigmoid of the margin, so
        `J == sigmoid(logit_scale * sep_min)` exactly and the two orderings
        agree; the margin is simply legible where J has saturated at 0.99. With
        three or more views the two orderings can differ, because J looks at
        the full softmax over N prompts while this looks at the runner-up only.
        J still decides pass/fail; this carries the order.
        """
        return min(self.sep)

    @property
    def holds(self) -> list[bool]:
        """Per view: did it read as its own prompt? (argmax == diagonal)"""
        return [s > 0.0 for s in self.sep]

    def lost_slots(self) -> list[str]:
        return [slot for slot, ok in zip(self.slots, self.holds, strict=True) if not ok]

    def diagnose(self) -> str:
        """Failure mode, used by propose.py's targeted swap and by the report.

        `ok`          every view reads as its prompt
        `all_lost`    no view does; discard the whole candidate
        `lost:a,b`    the named slots failed; resample only those
        """
        lost = self.lost_slots()
        if not lost:
            return "ok"
        if len(lost) == self.n_views:
            return "all_lost"
        return "lost:" + ",".join(lost)

    def to_json(self) -> str:
        d = asdict(self)
        d["sep"] = self.sep
        d["sep_min"] = self.sep_min
        d["holds"] = self.holds
        d["diagnosis"] = self.diagnose()
        return json.dumps(d, ensure_ascii=False)


@dataclass
class RunState:
    """Progress of one run. Resumable from state.json.

    Deliberately does NOT hold the Beta posteriors: those live in
    runs/vocab.db so they accumulate across runs. This object is only the
    run-specific bookkeeping needed to pick up where a stopped run left off.

    `last_round` is kept because the targeted-swap proposals read the previous
    round's failures; without it a resumed run would silently lose that
    quarter of its proposal budget.
    """

    run_id: str
    round_index: int = 0
    evaluated_uids: set[str] = field(default_factory=set)
    last_round: list[tuple[CandidateSpec, Verdict]] = field(default_factory=list)

    def record(self, spec: CandidateSpec, verdict: Verdict) -> None:
        self.evaluated_uids.add(spec.uid())

    def to_json(self) -> str:
        return json.dumps(
            {
                "run_id": self.run_id,
                "round_index": self.round_index,
                "evaluated_uids": sorted(self.evaluated_uids),
                "last_round": [
                    {"spec": asdict(spec), "verdict": asdict(verdict)}
                    for spec, verdict in self.last_round
                ],
            },
            ensure_ascii=False,
            indent=2,
        )

    @classmethod
    def from_json(cls, payload: str) -> RunState:
        d = json.loads(payload)
        return cls(
            run_id=d["run_id"],
            round_index=d["round_index"],
            evaluated_uids=set(d["evaluated_uids"]),
            last_round=[
                (CandidateSpec(**item["spec"]), Verdict(**item["verdict"]))
                for item in d["last_round"]
            ],
        )
