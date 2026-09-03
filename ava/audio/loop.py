"""Orchestration for the time-reversal anagram: propose -> generate -> judge -> record.

The audio counterpart of `ava.loop`, sharing its proposers, its run layout,
its state file and its reports. What differs is the artefact: a candidate is
one waveform written twice, forwards and backwards, and the "contact sheet"
is `audition.md`, a table of the candidates that held with the paths of the
files to listen to.

The search layer sees the audio task through the same `CandidateSpec` the
image track uses (`task='time_reverse'`, two prompts, an empty style); this
module translates it to `AudioCandidateSpec` at the generator's door. The
duration, guidance and step count are run-level settings, so the translation
is total and the uid the search records is the image-style one.

Run from the repository root:
    .venv/bin/python -m ava.audio.loop --run-id audio_evo --rounds 6 -k 6 \\
        --proposer evolve
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import yaml

from ava.audio.perceive import perceive
from ava.audio.spec import (
    DURATION_S,
    GUIDANCE_SCALE,
    NEGATIVE_PROMPT,
    NUM_INFERENCE_STEPS,
    AudioCandidateSpec,
)
from ava.audio.tasks import AUDIO_TASKS, FREQ_HYBRID_750, TIME_REVERSE
from ava.audio.vocab import seed_audio_vocab, seed_hybrid_vocab
from ava.audio.wavfile import write_wav
from ava.image.tasks import get_task
from ava.loop import (
    PROPOSERS,
    SUPPLIES,
    RunPaths,
    build_supply,
    load_state,
    top_up_vocabulary,
    write_prompt_card,
)
from ava.metric import alignment, concealment, multiway_probs
from ava.propose import (
    BanditProposer,
    Proposal,
    ProposalMix,
    Proposer,
    UniformProposer,
    assign_credit,
)
from ava.report import load_scores, rank_score, write_components, write_round_report
from ava.search.evolve import EvolutionaryProposer, SearchConfig
from ava.search.racing import RacingConfig
from ava.spec import CandidateSpec, RunState, Verdict
from ava.supply import SupplyPolicy, VocabSupply
from ava.vocab import connect

DEFAULT_RUNS_DIR = Path("runs")
VOCAB_FILENAME = "vocab.db"
# The search's clip length. Shorter than AudioCandidateSpec's 10 s default:
# an envelope states itself in a couple of seconds, generation time scales
# with length, and CLAP pads anything under its 10 s window anyway.
LOOP_DURATION_S = 5.0


@dataclass
class AudioLoopConfig:
    """Everything that decides what an audio run does. Written to config.yaml."""

    run_id: str
    rounds: int = 4
    k: int = 6
    seed: int = 0
    screening_seed: int = 0
    duration_s: float = LOOP_DURATION_S
    guidance_scale: float = GUIDANCE_SCALE
    num_inference_steps: int = NUM_INFERENCE_STEPS
    negative_prompt: str = NEGATIVE_PROMPT
    inject: float = 0.25
    swap: float = 0.25
    # Candidates below this J are left off audition.md; every one stays in
    # scores.jsonl. None lists everything.
    min_j: float | None = 0.5
    proposer: str = "bandit"
    search: dict[str, Any] = field(default_factory=lambda: SearchConfig().to_dict())
    # In-run vocabulary supply, as in the image loop; the sound format asks
    # GPT-2 for envelope sentences rather than painting subjects.
    supply: str = "none"
    supply_min_untried: int = 8
    supply_draws: int = 10
    device: str = "cuda"
    task: str = TIME_REVERSE.name
    # Which latent the anagram is sampled in: Stable Audio Open's waveform
    # codec or AudioLDM 2's mel codec. A run setting, not a candidate's: the
    # search's uid does not see it, so runs on different backends are
    # compared by their config.yaml, not by shared uids.
    backend: str = "stable_audio"
    # None means the backend's own default checkpoint.
    model_id: str | None = None
    # The frequency hybrid samples with a fitted low-pass projector (Step 0c).
    # It is data, not code, so a run names the file and records its digest.
    projector: str | None = None
    projector_md5: str | None = None

    def __post_init__(self) -> None:
        if self.task not in AUDIO_TASK_NAMES:
            raise ValueError(f"task must be one of {AUDIO_TASK_NAMES}, got {self.task}")
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend}")
        if self.supply not in SUPPLIES:
            raise ValueError(f"supply must be one of {SUPPLIES}, got {self.supply}")
        if self.proposer not in PROPOSERS:
            raise ValueError(
                f"proposer must be one of {PROPOSERS}, got {self.proposer}"
            )

    def mix(self) -> ProposalMix:
        return ProposalMix(inject=self.inject, swap=self.swap)

    def supply_policy(self) -> SupplyPolicy:
        return SupplyPolicy(min_untried=self.supply_min_untried)


# ---------------------------------------------------------------------------
# What the loop needs from the models
# ---------------------------------------------------------------------------


class AudioGenerator(Protocol):
    @property
    def sample_rate(self) -> int: ...

    def generate(self, spec: AudioCandidateSpec) -> np.ndarray: ...


BACKENDS: tuple[str, ...] = ("stable_audio", "audioldm2")


# Which backends can sample which task. The frequency hybrid needs a latent
# with a frequency axis, which only the mel codec has.
TASK_BACKENDS: dict[str, tuple[str, ...]] = {
    TIME_REVERSE.name: BACKENDS,
    FREQ_HYBRID_750.name: ("audioldm2",),
}


def build_engine(
    backend: str,
    device: str,
    model_id: str | None = None,
    task: str = TIME_REVERSE.name,
    projector: Path | None = None,
) -> AudioGenerator:
    """The generator for a (task, backend), loaded lazily on first use.

    Imported here rather than at the top so that the loop -- and its tests,
    which use a fake generator -- do not pay for diffusers. Everything that
    can be wrong about the combination is raised here, before any weights
    are read.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend}")
    if task not in TASK_BACKENDS:
        raise ValueError(f"task must be one of {tuple(TASK_BACKENDS)}, got {task}")
    if backend not in TASK_BACKENDS[task]:
        raise ValueError(
            f"{task} can only be sampled on {TASK_BACKENDS[task]}, not {backend}"
        )
    kwargs: dict[str, Any] = {"device": device}
    if model_id:
        kwargs["model_id"] = model_id
    if task == FREQ_HYBRID_750.name:
        if projector is None:
            raise ValueError(f"{task} needs --projector (Step 0c's fitted low-pass)")
        from ava.audio.bands import LowpassProjector
        from ava.audio.engine_audioldm2 import AudioLDM2HybridEngine

        fitted = LowpassProjector.load(projector)
        if fitted.cutoff_hz != 750.0:
            raise ValueError(
                f"{task} expects a 750 Hz projector, {projector} is {fitted.cutoff_hz}"
            )
        return AudioLDM2HybridEngine(projector=fitted, **kwargs)
    if backend == "audioldm2":
        from ava.audio.engine_audioldm2 import AudioLDM2Engine

        return AudioLDM2Engine(**kwargs)
    from ava.audio.engine import AudioAnagramEngine

    return AudioAnagramEngine(**kwargs)


class AudioJudge(Protocol):
    @property
    def logit_scale(self) -> float: ...

    def score_views(
        self, views: list[np.ndarray], prompts: list[str]
    ) -> torch.Tensor: ...

    def view_similarity(self, views: list[np.ndarray]) -> float: ...

    def text_emb(self, prompts: list[str]) -> torch.Tensor: ...


AUDIO_TASK_NAMES: tuple[str, ...] = tuple(task.name for task in AUDIO_TASKS)


# ---------------------------------------------------------------------------
# Translation between the search's spec and the generator's
# ---------------------------------------------------------------------------


def to_audio_spec(spec: CandidateSpec, config: AudioLoopConfig) -> AudioCandidateSpec:
    """The generator's view of a search candidate."""
    if spec.task != config.task or spec.n_views != 2:
        raise ValueError(f"expected a two-prompt {config.task!r} candidate, got {spec}")
    return AudioCandidateSpec(
        prompt_forward=spec.full_prompt(0),
        prompt_reverse=spec.full_prompt(1),
        seed=spec.seed,
        duration_s=config.duration_s,
        guidance_scale=spec.guidance_scale,
        num_inference_steps=spec.num_inference_steps,
        negative_prompt=config.negative_prompt,
    )


def verdict_for_wave(
    spec: CandidateSpec,
    matrix: torch.Tensor,
    logit_scale: float,
    self_similarity: float,
) -> Verdict:
    """The image track's Verdict, from CLAP's 2x2 matrix. Model-free, testable."""
    p, j = multiway_probs(matrix, logit_scale)
    return Verdict(
        uid=spec.uid(),
        task=spec.task,
        slots=get_task(spec.task).slot_names,
        scores=[[float(x) for x in row] for row in matrix.tolist()],
        p=p,
        j=j,
        alignment=alignment(matrix),
        concealment=concealment(matrix, logit_scale),
        captions=[],
        extra={"logit_scale": logit_scale, "self_similarity": self_similarity},
    )


# ---------------------------------------------------------------------------
# One candidate
# ---------------------------------------------------------------------------


def evaluate_candidate(
    proposal: Proposal,
    engine: AudioGenerator,
    judge: AudioJudge,
    config: AudioLoopConfig,
    out_dir: Path,
) -> tuple[Verdict, dict[str, Any]]:
    """Generate one candidate, write every view of it, score it.

    Returns the verdict and the row extras (wav paths, generation seconds).
    The task names the views; each is written as `<view>.wav` and scored
    against the prompt of its slot.
    """
    write_prompt_card(out_dir, proposal.spec, origin=proposal.origin)
    audio_spec = to_audio_spec(proposal.spec, config)
    task = get_task(proposal.spec.task)
    t0 = time.perf_counter()
    wave = engine.generate(audio_spec)
    seconds = time.perf_counter() - t0

    views = perceive(wave, task, engine.sample_rate)
    paths = {
        name: write_wav(out_dir / f"{name}.wav", view, engine.sample_rate)
        for name, view in zip(task.view_names, views, strict=True)
    }
    matrix = judge.score_views(views, audio_spec.prompts).cpu()
    verdict = verdict_for_wave(
        proposal.spec, matrix, judge.logit_scale, judge.view_similarity(views)
    )
    return verdict, {
        "wav_paths": {k: str(v) for k, v in paths.items()},
        "seconds": seconds,
    }


def audio_row(
    proposal: Proposal, verdict: Verdict, round_index: int, extras: dict[str, Any]
) -> dict[str, Any]:
    """One line of scores.jsonl; a report must be buildable from this alone."""
    row: dict[str, Any] = json.loads(verdict.to_json())
    row.update(
        {
            "round": round_index,
            "origin": proposal.origin,
            "detail": proposal.detail,
            "extra": dict(proposal.extra),
            "task": proposal.spec.task,
            "prompts": list(proposal.spec.prompts),
            "style": proposal.spec.style,
            "seed": proposal.spec.seed,
            "self_similarity": verdict.extra.get("self_similarity"),
            **extras,
        }
    )
    return row


def _card_extra(origin: str, verdict: Verdict) -> dict[str, object]:
    return {
        "origin": origin,
        "J": f"{verdict.j:.4f}",
        "sep": f"{verdict.sep_min:+.4f}",
        "selfsim": f"{float(verdict.extra.get('self_similarity', float('nan'))):.4f}",
        "verdict": verdict.diagnose(),
    }


# ---------------------------------------------------------------------------
# Rounds and the run
# ---------------------------------------------------------------------------


def run_round(
    config: AudioLoopConfig,
    paths: RunPaths,
    conn: sqlite3.Connection,
    proposer: Proposer,
    engine: AudioGenerator,
    judge: AudioJudge,
    state: RunState,
) -> list[dict[str, Any]]:
    index = state.round_index
    round_dir = paths.round_dir(index)
    round_dir.mkdir(parents=True, exist_ok=True)

    proposals = proposer.propose(state, config.k)
    with (round_dir / "candidates.jsonl").open("w", encoding="utf-8") as f:
        for p in proposals:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
    print(f"[round {index}] {len(proposals)} candidates proposed")

    rows: list[dict[str, Any]] = []
    completed: list[tuple[CandidateSpec, Verdict]] = []
    with (round_dir / "scores.jsonl").open("w", encoding="utf-8") as f:
        for i, proposal in enumerate(proposals, start=1):
            uid = proposal.spec.uid()
            verdict, extras = evaluate_candidate(
                proposal, engine, judge, config, round_dir / uid
            )
            assign_credit(conn, proposal.spec, verdict)
            write_prompt_card(
                round_dir / uid, proposal.spec, **_card_extra(proposal.origin, verdict)
            )
            row = audio_row(proposal, verdict, index, extras)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            rows.append(row)
            completed.append((proposal.spec, verdict))
            state.record(proposal.spec, verdict)
            print(
                f"  [{i}/{len(proposals)}] {uid} J={verdict.j:.3f} "
                f"sep={verdict.sep_min:+.3f} selfsim="
                f"{float(verdict.extra['self_similarity']):.3f} "
                f"({verdict.diagnose()}) {proposal.origin} [{extras['seconds']:.0f}s]"
            )

    write_round_report(rows, round_dir / "report.md", index)
    state.last_round = completed
    state.round_index = index + 1
    proposer.observe(state)
    paths.state.write_text(state.to_json(), encoding="utf-8")
    return rows


def write_audition(
    rows: Sequence[dict[str, Any]], out: Path, min_j: float | None
) -> Path:
    """The listening list: candidates that held, best first, with their files.

    The audio equivalent of the contact sheet. The cutoff only decides what
    is listed; how many were left out is stated, and every candidate stays in
    scores.jsonl so the list can be redrawn without generating anything.
    """
    ranked = sorted(rows, key=rank_score, reverse=True)
    shown = ranked if min_j is None else [r for r in ranked if float(r["j"]) >= min_j]
    hidden = len(ranked) - len(shown)
    # One task per run; its slot and view names label the columns.
    task = (
        get_task(ranked[0].get("task", TIME_REVERSE.name)) if ranked else TIME_REVERSE
    )
    slot_cols = " | ".join(f"{name} prompt" for name in task.slot_names)
    wav_cols = " | ".join(f"{name}.wav" for name in task.view_names)
    lines = [
        "# Audition list",
        "",
        f"{len(shown)} of {len(ranked)} candidates listed"
        + (
            f" (J >= {min_j}); {hidden} hidden, all of them in scores.jsonl."
            if hidden
            else "."
        ),
        "",
        "`sep` is the weaker view's raw CLAP margin over the other prompt; "
        "`selfsim` is the cosine between the two views' embeddings "
        "(near 1.0 means CLAP cannot hear the difference between the views).",
        "",
        f"| sep | J | selfsim | {slot_cols} | seed | origin | {wav_cols} |",
        "|---" * (5 + len(task.slots) + len(task.view_names)) + "|",
    ]
    for r in shown:
        wav = r.get("wav_paths", {})
        selfsim = r.get("self_similarity")
        prompts = " | ".join(r["prompts"])
        wavs = " | ".join(wav.get(name, "") for name in task.view_names)
        lines.append(
            f"| {rank_score(r):+.3f} | {float(r['j']):.3f} "
            f"| {'—' if selfsim is None else f'{float(selfsim):.3f}'} "
            f"| {prompts} | {r['seed']} | {r['origin']} | {wavs} |"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def run_loop(
    config: AudioLoopConfig,
    paths: RunPaths,
    conn: sqlite3.Connection,
    proposer: Proposer,
    engine: AudioGenerator,
    judge: AudioJudge,
    supply: VocabSupply | None = None,
) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.config.write_text(
        yaml.safe_dump(asdict(config), sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    state = load_state(paths, config.run_id)
    target = state.round_index + config.rounds
    while state.round_index < target:
        run_round(config, paths, conn, proposer, engine, judge, state)
        top_up_vocabulary(
            [config.task], conn, supply, config.supply_policy(), state.round_index - 1
        )

    all_rows: list[dict[str, Any]] = []
    for path in paths.all_scores():
        all_rows.extend(load_scores(path))
    write_components(conn, paths.components, np.random.default_rng(config.seed))
    audition = write_audition(all_rows, paths.root / "audition.md", config.min_j)
    print(f"\nwrote {audition}")
    print(f"wrote {paths.components}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", default="audio_run")
    p.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("-k", "--k", type=int, default=6)
    p.add_argument("--seed", type=int, default=0, help="seed for proposal sampling")
    p.add_argument("--screening-seed", type=int, default=0)
    p.add_argument(
        "--duration",
        type=float,
        default=LOOP_DURATION_S,
        help=f"clip length in seconds (Step 1 used {DURATION_S})",
    )
    p.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE)
    p.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS)
    p.add_argument("--inject", type=float, default=0.25)
    p.add_argument("--swap", type=float, default=0.25)
    p.add_argument("--min-j", type=float, default=0.5, help="-1 lists everything")
    p.add_argument("--proposer", choices=PROPOSERS, default="bandit")
    p.add_argument("--supply", choices=SUPPLIES, default="llm")
    p.add_argument("--supply-min-untried", type=int, default=8)
    p.add_argument("--supply-draws", type=int, default=10)
    p.add_argument("--clusters", type=int, default=SearchConfig().clusters)
    p.add_argument("--eta", type=float, default=SearchConfig().eta)
    p.add_argument("--race-fraction", type=float, default=RacingConfig().fraction)
    p.add_argument("--max-seeds", type=int, default=RacingConfig().max_seeds)
    p.add_argument(
        "--children-per-slot", type=int, default=SearchConfig().children_per_slot
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--backend",
        choices=BACKENDS,
        default="stable_audio",
        help="which model's latent the anagram is sampled in",
    )
    p.add_argument("--model-id", default=None, help="default: the backend's own")
    p.add_argument(
        "--task",
        choices=AUDIO_TASK_NAMES,
        default=TIME_REVERSE.name,
        help="which audio illusion to search",
    )
    p.add_argument(
        "--projector",
        type=Path,
        default=None,
        help="Step 0c's fitted low-pass (.npz stem), required by freq_hybrid_750",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    search_cfg = SearchConfig(
        clusters=args.clusters,
        cluster_seed=args.seed,
        eta=args.eta,
        children_per_slot=args.children_per_slot,
        racing=RacingConfig(max_seeds=args.max_seeds, fraction=args.race_fraction),
    )
    config = AudioLoopConfig(
        run_id=args.run_id,
        rounds=args.rounds,
        k=args.k,
        seed=args.seed,
        screening_seed=args.screening_seed,
        duration_s=args.duration,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        inject=args.inject,
        swap=args.swap,
        min_j=None if args.min_j < 0 else args.min_j,
        proposer=args.proposer,
        search=search_cfg.to_dict(),
        supply=args.supply,
        supply_min_untried=args.supply_min_untried,
        supply_draws=args.supply_draws,
        device=args.device,
        backend=args.backend,
        model_id=args.model_id,
        task=args.task,
        projector=None if args.projector is None else str(args.projector),
        projector_md5=None
        if args.projector is None
        else hashlib.md5(
            Path(args.projector).with_suffix(".npz").read_bytes()
        ).hexdigest(),
    )
    conn = connect(args.runs_dir / VOCAB_FILENAME)
    added = seed_audio_vocab(conn)
    if added:
        print(f"[vocab] seeded {added} audio arms")
    if config.task == FREQ_HYBRID_750.name:
        added = seed_hybrid_vocab(conn)
        if added:
            print(f"[vocab] seeded {added} hybrid arms")
    paths = RunPaths(args.runs_dir / args.run_id)

    from ava.audio.judge import ClapJudge

    engine = build_engine(
        config.backend,
        config.device,
        config.model_id,
        task=config.task,
        projector=None if config.projector is None else Path(config.projector),
    )
    print(f"[engine] {config.backend} ({config.model_id or 'default checkpoint'})")
    judge = ClapJudge(device=config.device, sample_rate=engine.sample_rate)
    proposer: Proposer
    rng = np.random.default_rng(args.seed)
    if config.proposer == "evolve":
        print("[proposer] evolutionary search (archive, racing, surrogate)")

        def embed(prompts: Sequence[str]) -> np.ndarray:
            # CLAP's text tower: the judge and the search share one notion of
            # "similar prompt".
            return judge.text_emb(list(prompts)).cpu().numpy()

        proposer = EvolutionaryProposer(
            conn,
            rng,
            embed,
            tasks=[config.task],
            cfg=search_cfg,
            screening_seed=config.screening_seed,
            guidance_scale=config.guidance_scale,
            num_inference_steps=config.num_inference_steps,
            state_dir=paths.root,
        )
    elif config.proposer == "uniform":
        print("[proposer] uniform sampling (validation sweep, not a search)")
        proposer = UniformProposer(
            conn,
            rng,
            tasks=[config.task],
            screening_seed=config.screening_seed,
            guidance_scale=config.guidance_scale,
            num_inference_steps=config.num_inference_steps,
        )
    else:
        proposer = BanditProposer(
            conn,
            rng,
            tasks=[config.task],
            mix=config.mix(),
            screening_seed=config.screening_seed,
            guidance_scale=config.guidance_scale,
            num_inference_steps=config.num_inference_steps,
        )

    run_loop(config, paths, conn, proposer, engine, judge, build_supply(config, paths))


if __name__ == "__main__":
    main()
