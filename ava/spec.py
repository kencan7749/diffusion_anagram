"""Data structures for search candidates and their evaluation results.

sigma / kernel_size are module constants rather than CandidateSpec fields.
Phase 1 does not search over them, and keeping them out of the spec states
that fact in code.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field

# Values passed to HybridLowPassView / HybridHighPassView.
# Both views must receive the same value, otherwise lp(e1) + hp(e2) is not an
# identity decomposition.
SIGMA = 2.0
KERNEL_SIZE = 33

# Required by Factorized Diffusion. Distinct from Visual Anagrams' 'mean'.
REDUCTION = "sum"


@dataclass(frozen=True)
class CandidateSpec:
    """A single point in the search space."""

    prompt_low: str  # seen from far away (blurred)
    prompt_high: str  # seen up close
    style: str = ""  # prepended to both prompts
    seed: int = 0
    guidance_scale: float = 10.0
    num_inference_steps: int = 30

    @property
    def full_low(self) -> str:
        # Same assembly as generate.py's f'{args.style} {p}'.strip()
        return f"{self.style} {self.prompt_low}".strip()

    @property
    def full_high(self) -> str:
        return f"{self.style} {self.prompt_high}".strip()

    @property
    def prompts(self) -> list[str]:
        """Ordered to match the views (low_pass, high_pass)."""
        return [self.full_low, self.full_high]

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
    # CLIP score matrix S[view][prompt]; both axes ordered (low, high)
    s_far_low: float
    s_far_high: float
    s_near_low: float
    s_near_high: float
    # Probability that each view picks its own prompt (chance level 0.5)
    p_far: float
    p_near: float
    # Illusion score. min, not sum: an illusion only holds if both views hold.
    j: float
    # Never used as a numeric signal. Kept as report evidence and as future
    # VLM input.
    caption_far: str = ""
    caption_near: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def sep_far(self) -> float:
        """Margin in the far view. Negative means the low-frequency side lost."""
        return self.s_far_low - self.s_far_high

    @property
    def sep_near(self) -> float:
        """Margin in the near view. Negative means the high-frequency side is absent."""
        return self.s_near_high - self.s_near_low

    @property
    def sep_min(self) -> float:
        """Ranking key: the weaker of the two raw CLIP margins.

        Same logic as J -- an illusion is only as good as its weaker view --
        but on the cosine margins instead of the probabilities.

        This does NOT reorder anything. A two-way softmax is a sigmoid of the
        margin, and both views share CLIP's logit_scale, so
        J == sigmoid(logit_scale * sep_min) exactly; the two induce the same
        ranking. What changes is resolution. In a 40-candidate sweep the top 13
        printed as J >= 0.99 while their margins spanned +0.048 to +0.114 -- a
        2.4x range compressed past the third decimal. Beyond a margin of about
        0.37 the sigmoid reaches 1.0 in float64 and the ties become real.

        So: J decides pass/fail, where saturation is harmless. This carries the
        order, where saturation costs first legibility and eventually the signal.
        """
        return min(self.sep_far, self.sep_near)

    def diagnose(self) -> str:
        """Failure mode, used by propose.py's targeted swap and by the report."""
        lo_ok = self.p_far > 0.5
        hi_ok = self.p_near > 0.5
        if lo_ok and hi_ok:
            return "ok"
        if not lo_ok and not hi_ok:
            return "pair_mismatch"  # discard the whole triple
        if not lo_ok:
            return "low_loses"  # resample prompt_low only
        return "high_absent"  # resample prompt_high only

    def to_json(self) -> str:
        d = asdict(self)
        d["sep_far"] = self.sep_far
        d["sep_near"] = self.sep_near
        d["sep_min"] = self.sep_min
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
    def from_json(cls, payload: str) -> "RunState":
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
