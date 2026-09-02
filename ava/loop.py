"""Orchestration: propose -> generate -> judge -> record, round after round.

The division of labour this whole system rests on: the loop optimises YIELD
(how often a proposed candidate actually produces a working illusion) and never
tries to judge whether an image is interesting. That judgement stays with the
person looking at contact_sheet.png.

Search runs in two gears:

  screening  one seed per candidate, so the number of candidates seen per hour
             is as large as possible
  harvest    the best candidates only, regenerated across several seeds and
             upscaled, so a good pair is not lost to one unlucky noise draw

Everything a figure needs is written to disk during the run: the scored view
images, the raw CLIP matrix, the proposal provenance, and for randomised views
the permutation that was drawn. Reports are regenerated from those files
alone, so re-drawing never costs a diffusion step.

Run from the repository root:
    .venv/bin/python -m ava.loop --tasks hybrid,flip,jigsaw --rounds 4 --k 8
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ava.image.engine import Engine, Generator, save_sample
from ava.image.judge import ClipBlipJudge, Judge
from ava.image.paper_examples import seed_paper_vocab
from ava.image.tasks import get_task
from ava.image.views import ViewSet
from ava.propose import (
    BanditProposer,
    Proposal,
    ProposalMix,
    Proposer,
    UniformProposer,
    assign_credit,
)
from ava.report import (
    build_contact_sheet,
    load_scores,
    rank_score,
    write_components,
    write_round_report,
)
from ava.spec import (
    JIGSAW_SEED,
    KERNEL_SIZE,
    MOTION_SIZE,
    PATCH_GRID,
    PIXEL_GRID,
    SIGMA,
    SKEW_FACTOR,
    TRIPLE_KERNEL_SIZE,
    TRIPLE_SIGMA_1,
    TRIPLE_SIGMA_2,
    CandidateSpec,
    RunState,
    Verdict,
)
from ava.vocab import connect, seed_author_vocab

DEFAULT_RUNS_DIR = Path("runs")
VOCAB_FILENAME = "vocab.db"
DEFAULT_TASKS = ("hybrid",)

# Recorded in config.yaml, not adjustable: the search space is prompts only.
VIEW_PARAMS: dict[str, float | int] = {
    "sigma": SIGMA,
    "kernel_size": KERNEL_SIZE,
    "triple_sigma_1": TRIPLE_SIGMA_1,
    "triple_sigma_2": TRIPLE_SIGMA_2,
    "triple_kernel_size": TRIPLE_KERNEL_SIZE,
    "motion_size": MOTION_SIZE,
    "skew_factor": SKEW_FACTOR,
    "patch_grid": PATCH_GRID,
    "pixel_grid": PIXEL_GRID,
    "jigsaw_seed": JIGSAW_SEED,
}


@dataclass
class LoopConfig:
    """Everything that decides what a run does. Written to config.yaml verbatim."""

    run_id: str
    tasks: list[str] = field(default_factory=lambda: list(DEFAULT_TASKS))
    rounds: int = 4
    k: int = 8
    seed: int = 0
    screening_seed: int = 0
    # Seed for views that draw a random permutation when built. The drawn
    # permutation is also persisted under runs/<run>/views/.
    view_seed: int = 0
    guidance_scale: float = 10.0
    num_inference_steps: int = 30
    inject: float = 0.25
    swap: float = 0.25
    # J screens. The 0-3 visual check confirmed the low-J tail really is not
    # worth looking at, so the score is allowed to spend the expensive steps
    # only on what held. None disables the cutoff and harvests every pair.
    harvest_top: int | None = 8
    harvest_seeds: int = 4
    # Candidates below this J are kept in scores.jsonl but left off the contact
    # sheet. 0.5 is the point where a two-way view stops beating chance, which
    # is the same boundary Verdict.diagnose() calls `ok`. None shows everything.
    min_j: float | None = 0.5
    harvest_upscale: bool = False
    device: str = "cuda"
    # Inverse problems: the image whose first component is kept, and the words
    # that describe it for the judge.
    ref_image: str | None = None
    ref_prompt: str | None = None
    view_params: dict[str, float | int] = field(
        default_factory=lambda: dict(VIEW_PARAMS), init=False
    )

    def mix(self) -> ProposalMix:
        return ProposalMix(inject=self.inject, swap=self.swap)


@dataclass
class RunPaths:
    root: Path

    def round_dir(self, index: int) -> Path:
        return self.root / f"round_{index:03d}"

    @property
    def state(self) -> Path:
        return self.root / "state.json"

    @property
    def config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def components(self) -> Path:
        return self.root / "components.md"

    @property
    def contact_sheet(self) -> Path:
        return self.root / "contact_sheet.png"

    @property
    def harvest(self) -> Path:
        return self.root / "harvest"

    def view_state(self, task: str) -> Path:
        return self.root / "views" / f"{task}.pt"

    def all_scores(self) -> list[Path]:
        return sorted(self.root.glob("round_*/scores.jsonl"))


def load_state(paths: RunPaths, run_id: str) -> RunState:
    """Resume if a state file exists, so an interrupted run can continue."""
    if paths.state.exists():
        state = RunState.from_json(paths.state.read_text(encoding="utf-8"))
        print(
            f"[resume] round {state.round_index}, {len(state.evaluated_uids)} evaluated"
        )
        return state
    return RunState(run_id=run_id)


def build_viewsets(config: LoopConfig, paths: RunPaths) -> dict[str, ViewSet]:
    """One ViewSet per task, built once per run.

    A randomised view's permutation is what makes its images comparable, so it
    is saved on first build and reloaded on resume instead of being redrawn.
    """
    viewsets: dict[str, ViewSet] = {}
    for name in config.tasks:
        viewset = ViewSet.build(get_task(name), seed=config.view_seed)
        if viewset.task.randomised:
            path = paths.view_state(name)
            if path.exists():
                viewset.load_state(torch.load(path))
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(viewset.state(), path)
        viewsets[name] = viewset
    return viewsets


def record_row(
    proposal: Proposal,
    verdict: Verdict,
    round_index: int,
    image_path: Path,
    view_paths: list[Path],
) -> dict[str, Any]:
    """One line of scores.jsonl: the verdict plus everything a report needs.

    The prompts and image paths are duplicated here on purpose. A report must
    be buildable from this file alone, without re-deriving anything from the
    proposal log or the database.
    """
    row: dict[str, Any] = json.loads(verdict.to_json())
    row.update(
        {
            "round": round_index,
            "origin": proposal.origin,
            "detail": proposal.detail,
            "task": proposal.spec.task,
            "prompts": list(proposal.spec.prompts),
            "style": proposal.spec.style,
            "seed": proposal.spec.seed,
            "ref_image": proposal.spec.ref_image,
            "image_path": str(image_path),
            "view_paths": [str(p) for p in view_paths],
        }
    )
    return row


def write_prompt_card(out_dir: Path, spec: CandidateSpec, **extra: object) -> Path:
    """Leave a readable note beside the images saying what they were.

    A candidate directory is named by a content hash, so on its own it says
    nothing about what was tried. The prompts are in the round's scores.jsonl,
    but a directory that has to be cross-referenced to be understood is a
    directory nobody reads. This is written before generation, so an
    interrupted candidate still says what it was attempting, and rewritten
    afterwards with the verdict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    task = get_task(spec.task)
    width = max(len(s.name) for s in task.slots)
    lines = [f"task   : {spec.task}"]
    for slot, prompt in zip(task.slots, spec.prompts, strict=True):
        lines.append(f"{slot.name:{width}s} : {prompt}")
    lines += [f"style  : {spec.style or '(none)'}", ""]
    for i, slot in enumerate(task.slots):
        lines.append(
            f"view {slot.name:{width}s} should read as : {spec.full_prompt(i)}"
        )
    if spec.ref_image is not None:
        lines.append(f"reference image : {spec.ref_image}")
    lines += [
        "",
        f"uid    : {spec.uid()}",
        f"seed   : {spec.seed}",
        f"steps  : {spec.num_inference_steps}",
        f"cfg    : {spec.guidance_scale}",
    ]
    for key, value in extra.items():
        lines.append(f"{key:7s}: {value}")
    path = out_dir / "prompt.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def save_views(
    judge: Judge, image: torch.Tensor, viewset: ViewSet, out_dir: Path
) -> list[Path]:
    """Persist what the judge looked at, one PNG per slot."""
    from torchvision.utils import save_image

    paths = []
    for slot, view in zip(viewset.task.slots, judge.views(image, viewset), strict=True):
        path = out_dir / f"view_{slot.name}.png"
        save_image(view, path, padding=0)
        paths.append(path)
    return paths


def evaluate_candidate(
    proposal: Proposal,
    engine: Generator,
    judge: Judge,
    viewset: ViewSet,
    out_dir: Path,
) -> tuple[Verdict, torch.Tensor, Path, list[Path]]:
    """Generate one candidate, score it, and persist every scored view.

    Returns the verdict, the 256 px image, its path, and the view image paths.
    """
    write_prompt_card(out_dir, proposal.spec, origin=proposal.origin)
    _, image_256 = engine.generate(proposal.spec, viewset)
    image_path = save_sample(image_256, out_dir)
    view_paths = save_views(judge, image_256, viewset, out_dir)
    verdict = judge.evaluate(image_256, proposal.spec, viewset)
    return verdict, image_256, image_path, view_paths


def _card_extra(proposal_origin: str, verdict: Verdict) -> dict[str, object]:
    extra: dict[str, object] = {
        "origin": proposal_origin,
        "J": f"{verdict.j:.4f}",
        "sep": f"{verdict.sep_min:+.4f}",
        "verdict": verdict.diagnose(),
    }
    for slot, caption in zip(verdict.slots, verdict.captions, strict=False):
        extra[f"cap {slot}"] = caption
    return extra


def run_round(
    config: LoopConfig,
    paths: RunPaths,
    conn: sqlite3.Connection,
    proposer: Proposer,
    engine: Generator,
    judge: Judge,
    viewsets: dict[str, ViewSet],
    state: RunState,
) -> list[dict[str, Any]]:
    """One round: propose k, generate and score each, credit the components."""
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
    scores_path = round_dir / "scores.jsonl"

    with scores_path.open("w", encoding="utf-8") as f:
        for i, proposal in enumerate(proposals, start=1):
            uid = proposal.spec.uid()
            viewset = viewsets[proposal.spec.task]
            verdict, _, image_path, view_paths = evaluate_candidate(
                proposal, engine, judge, viewset, round_dir / uid
            )
            # Credit is assigned per component, so one candidate teaches every arm.
            assign_credit(conn, proposal.spec, verdict)

            write_prompt_card(
                round_dir / uid, proposal.spec, **_card_extra(proposal.origin, verdict)
            )
            row = record_row(proposal, verdict, index, image_path, view_paths)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()  # a killed run keeps every candidate it already paid for
            rows.append(row)
            completed.append((proposal.spec, verdict))
            state.record(proposal.spec, verdict)

            print(
                f"  [{i}/{len(proposals)}] {uid} {proposal.spec.task} "
                f"J={verdict.j:.3f} ({verdict.diagnose()}) {proposal.origin}"
            )

    write_round_report(rows, round_dir / "report.md", index)

    state.last_round = completed
    state.round_index = index + 1
    paths.state.write_text(state.to_json(), encoding="utf-8")
    return rows


def candidate_key(row: dict[str, Any]) -> tuple[str, tuple[str, ...], str]:
    """What makes two rows the same candidate up to the seed."""
    return str(row["task"]), tuple(str(p) for p in row["prompts"]), str(row["style"])


def harvest(
    config: LoopConfig,
    paths: RunPaths,
    engine: Generator,
    judge: Judge,
    viewsets: dict[str, ViewSet],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Regenerate the best candidates across several seeds.

    Screening spends one seed per candidate to cover more of them, which trades
    away the ones that happen to fail on their one seed. This buys those back
    for the candidates that survived screening.

    The cutoff is real: it decides where GPU time goes. What it excluded is
    always logged and always still in scores.jsonl, so the line can be redrawn
    and the skipped candidates harvested later without rerunning the search.
    """
    # Ordered by the raw margin. This gives the same order as J for two views
    # (J is a sigmoid of it) but keeps its resolution where J saturates, so the
    # logged cutoff boundary is meaningful. The cutoff still uses J to decide
    # what qualifies at all.
    ranked = sorted(rows, key=rank_score, reverse=True)
    seen: set[tuple[str, tuple[str, ...], str]] = set()
    picks: list[dict[str, Any]] = []
    for row in ranked:
        key = candidate_key(row)
        if key in seen:
            continue
        seen.add(key)
        picks.append(row)

    if config.harvest_top is not None and len(picks) > config.harvest_top:
        dropped = picks[config.harvest_top :]
        picks = picks[: config.harvest_top]
        kept = f"{rank_score(picks[-1]):+.3f}" if picks else "(nothing kept)"
        print(
            f"[harvest] cutoff at top {config.harvest_top}: skipping "
            f"{len(dropped)} candidates, lowest kept sep={kept}, "
            f"highest skipped sep={rank_score(dropped[0]):+.3f}"
        )
    if not picks:
        print("[harvest] nothing to harvest")
        return []
    print(f"[harvest] {len(picks)} candidates x {config.harvest_seeds} seeds")

    out_rows: list[dict[str, Any]] = []
    paths.harvest.mkdir(parents=True, exist_ok=True)
    with (paths.harvest / "scores.jsonl").open("w", encoding="utf-8") as f:
        for row in picks:
            for seed in range(config.harvest_seeds):
                spec = CandidateSpec(
                    task=str(row["task"]),
                    prompts=tuple(str(p) for p in row["prompts"]),
                    style=str(row["style"]),
                    seed=seed,
                    guidance_scale=config.guidance_scale,
                    num_inference_steps=config.num_inference_steps,
                    ref_image=row.get("ref_image"),
                )
                viewset = viewsets[spec.task]
                out_dir = paths.harvest / spec.uid()
                proposal = Proposal(spec, "harvest", f"seed {seed} of a top candidate")
                verdict, image_256, image_path, view_paths = evaluate_candidate(
                    proposal, engine, judge, viewset, out_dir
                )
                write_prompt_card(out_dir, spec, **_card_extra("harvest", verdict))
                out_row = record_row(proposal, verdict, -1, image_path, view_paths)
                if config.harvest_upscale:
                    upscaled = engine.upscale_1024(spec, image_256, viewset)
                    out_row["image_1024_path"] = str(save_sample(upscaled, out_dir))

                f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                f.flush()
                out_rows.append(out_row)
                print(f"  [harvest] {spec.uid()} seed={seed} J={verdict.j:.3f}")
    return out_rows


def run_loop(
    config: LoopConfig,
    paths: RunPaths,
    conn: sqlite3.Connection,
    proposer: Proposer,
    engine: Generator,
    judge: Judge,
) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.config.write_text(
        yaml.safe_dump(asdict(config), sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    viewsets = build_viewsets(config, paths)

    state = load_state(paths, config.run_id)
    target = state.round_index + config.rounds
    while state.round_index < target:
        run_round(config, paths, conn, proposer, engine, judge, viewsets, state)

    all_rows: list[dict[str, Any]] = []
    for path in paths.all_scores():
        all_rows.extend(load_scores(path))

    if config.harvest_seeds > 0:
        harvest(config, paths, engine, judge, viewsets, all_rows)

    # Reports last, and only from what is already on disk.
    rng = np.random.default_rng(config.seed)
    write_components(conn, paths.components, rng)
    build_contact_sheet(all_rows, paths.contact_sheet, min_j=config.min_j)
    print(f"\nwrote {paths.contact_sheet}")
    print(f"wrote {paths.components}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", default="run", help="run directory name under runs/")
    p.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    p.add_argument(
        "--tasks",
        default=",".join(DEFAULT_TASKS),
        help="comma-separated task names (see ava.image.tasks.TASKS)",
    )
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("-k", "--k", type=int, default=8, help="candidates per round")
    p.add_argument("--seed", type=int, default=0, help="seed for proposal sampling")
    p.add_argument(
        "--screening-seed",
        type=int,
        default=0,
        help="diffusion seed held fixed during screening",
    )
    p.add_argument(
        "--view-seed",
        type=int,
        default=0,
        help="seed for views that draw a random permutation (patch/pixel permute)",
    )
    p.add_argument("--inject", type=float, default=0.25)
    p.add_argument("--swap", type=float, default=0.25)
    p.add_argument(
        "--harvest-top",
        type=int,
        default=8,
        help="harvest only the top candidates (use -1 to harvest every one)",
    )
    p.add_argument(
        "--min-j",
        type=float,
        default=0.5,
        help=(
            "hide candidates below this J from the contact sheet; they stay in "
            "scores.jsonl either way (use -1 to show everything)"
        ),
    )
    p.add_argument("--harvest-seeds", type=int, default=4)
    p.add_argument("--harvest-upscale", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--ref-image", default=None, help="reference image for inverse_hybrid"
    )
    p.add_argument(
        "--ref-prompt",
        default=None,
        help="what the reference image shows, for the judge (e.g. 'albert einstein')",
    )
    p.add_argument(
        "--uniform",
        action="store_true",
        help=(
            "sample components uniformly instead of by posterior. A validation "
            "sweep, not a search: it gives every arm comparable evidence so the "
            "component ranking reflects measurements rather than priors"
        ),
    )
    p.add_argument(
        "--no-blip",
        action="store_true",
        help="skip captioning; disables the collapsed-views diagnosis",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in tasks:
        get_task(t)  # fail early on a typo, before any model is loaded
    config = LoopConfig(
        run_id=args.run_id,
        tasks=tasks,
        rounds=args.rounds,
        k=args.k,
        seed=args.seed,
        screening_seed=args.screening_seed,
        view_seed=args.view_seed,
        inject=args.inject,
        swap=args.swap,
        harvest_top=None if args.harvest_top < 0 else args.harvest_top,
        harvest_seeds=args.harvest_seeds,
        min_j=None if args.min_j < 0 else args.min_j,
        harvest_upscale=args.harvest_upscale,
        device=args.device,
        ref_image=args.ref_image,
        ref_prompt=args.ref_prompt,
    )

    # The vocabulary database is shared across runs on purpose: the posteriors
    # are the asset this system accumulates.
    conn = connect(args.runs_dir / VOCAB_FILENAME)
    added = seed_author_vocab(conn) + seed_paper_vocab(conn)
    if added:
        print(f"[vocab] seeded {added} arms")

    paths = RunPaths(args.runs_dir / args.run_id)
    proposer: Proposer
    if args.uniform:
        print("[proposer] uniform sampling (validation sweep, not a search)")
        proposer = UniformProposer(
            conn,
            np.random.default_rng(args.seed),
            tasks=config.tasks,
            screening_seed=config.screening_seed,
            guidance_scale=config.guidance_scale,
            num_inference_steps=config.num_inference_steps,
            ref_image=config.ref_image,
            ref_prompt=config.ref_prompt,
        )
    else:
        proposer = BanditProposer(
            conn,
            np.random.default_rng(args.seed),
            tasks=config.tasks,
            mix=config.mix(),
            screening_seed=config.screening_seed,
            guidance_scale=config.guidance_scale,
            num_inference_steps=config.num_inference_steps,
            ref_image=config.ref_image,
            ref_prompt=config.ref_prompt,
        )
    engine = Engine(device=config.device)
    judge = ClipBlipJudge(device=config.device, use_blip=not args.no_blip)

    run_loop(config, paths, conn, proposer, engine, judge)


if __name__ == "__main__":
    main()
