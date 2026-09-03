"""Step B-7: does a cheaper generation predict whether the full one holds?

Racing's first rung could be cheaper than a full 30-step, two-stage
generation, if a cheaper one *orders candidates the same way*. Two cheaper
readings are tested, both of which fall out of generations that are run
anyway:

    64@30    the stage-1 (64 px) image of the 30-step generation
    256@15   a 15-step generation, both stages
    64@15    the stage-1 image of that 15-step generation

Each candidate is generated twice (30 and 15 steps) and every image is judged
with the same CLIP judge and the same views. The reference is the full
generation, `256@30`, and its pass/fail (`diagnosis == ok`). For each cheap
reading the ROC AUC of its `sep_min` against that pass/fail is reported,
with the Spearman correlation of the two `sep_min` orderings. The bar in
design_search_v2.md Sec. 6 is AUC >= 0.8; below it, rung 0 stays at full
fidelity.

Candidates are the paper examples (one per task, which mostly hold) plus
uniformly drawn pairs from the vocabulary (which mostly do not), so both
classes are populated. Analysis only: writes scores.jsonl, then fidelity.json
from it. `--analyse-only` recomputes fidelity.json without the GPU.

Run from the repository root:
    .venv/bin/python -m scripts.stepB_fidelity
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ava.image.tasks import get_task
from ava.propose import Proposal, UniformProposer
from ava.rankstats import auc, spearman
from ava.report import load_scores
from ava.spec import CandidateSpec, RunState
from ava.vocab import connect, seed_author_vocab
from scripts.stepA_paper_examples import EXAMPLES, spec_for

REFERENCE = "256@30"
UNIFORM_TASKS = ("flip", "hybrid", "jigsaw", "skew", "negate", "color_hybrid")


def paper_candidates(seed: int) -> list[tuple[CandidateSpec, str]]:
    return [(spec_for(task, seed), "paper") for task in EXAMPLES]


def uniform_candidates(
    db: Path, n: int, tasks: Sequence[str], seed: int
) -> list[tuple[CandidateSpec, str]]:
    """Pairs drawn uniformly from the vocabulary; the database is not written to."""
    conn = connect(db)
    seed_author_vocab(conn)
    proposer = UniformProposer(
        conn, np.random.default_rng(seed), tasks=list(tasks), screening_seed=seed
    )
    proposals = proposer.propose(RunState(run_id="fidelity"), n)
    conn.close()
    return [(p.spec, "uniform") for p in proposals]


def analyse(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """AUC and rank agreement of every cheap reading against the reference."""
    by_fidelity: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        by_fidelity.setdefault(str(r["fidelity"]), {})[str(r["candidate"])] = r
    reference = by_fidelity.get(REFERENCE, {})
    out: dict[str, Any] = {
        "reference": REFERENCE,
        "n_candidates": len(reference),
        "n_reference_held": sum(
            1 for r in reference.values() if r["diagnosis"] == "ok"
        ),
        "readings": {},
        "seconds_per_generation": {},
    }
    for label, table in by_fidelity.items():
        secs = [float(r["seconds"]) for r in table.values() if r.get("seconds")]
        if secs:
            out["seconds_per_generation"][label] = float(np.mean(secs))
        if label == REFERENCE:
            continue
        shared = sorted(set(table) & set(reference))
        if len(shared) < 3:
            continue
        cheap = [float(table[c]["sep_min"]) for c in shared]
        full = [float(reference[c]["sep_min"]) for c in shared]
        held = [reference[c]["diagnosis"] == "ok" for c in shared]
        agree = [
            (table[c]["diagnosis"] == "ok") == h
            for c, h in zip(shared, held, strict=True)
        ]
        a = auc(cheap, held)
        out["readings"][label] = {
            "n": len(shared),
            "auc_vs_reference_pass": a,
            "spearman_vs_reference_sep_min": spearman(cheap, full),
            "pass_fail_agreement": float(np.mean(agree)),
            "usable_for_rung_0": (a is not None and a >= 0.8),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/stepB/fidelity"))
    parser.add_argument("--db", type=Path, default=Path("runs/vocab.db"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-uniform", type=int, default=22)
    parser.add_argument("--uniform-tasks", default=",".join(UNIFORM_TASKS))
    parser.add_argument("--steps-low", type=int, default=15)
    parser.add_argument("--analyse-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    scores_path = args.out / "scores.jsonl"

    if args.analyse_only:
        result = analyse(load_scores(scores_path))
        (args.out / "fidelity.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    import torch

    from ava.image.engine import Engine, save_sample
    from ava.image.judge import ClipBlipJudge
    from ava.image.views import ViewSet
    from ava.loop import record_row, write_prompt_card

    # A private copy: the uniform draw must not touch the shared posteriors.
    db_copy = args.out / "vocab.db"
    if args.db.exists():
        shutil.copy(args.db, db_copy)
    candidates = paper_candidates(args.seed) + uniform_candidates(
        db_copy, args.n_uniform, args.uniform_tasks.split(","), args.seed
    )
    print(f"[fidelity] {len(candidates)} candidates x 2 generations")

    engine = Engine(device=args.device)
    judge = ClipBlipJudge(device=args.device, use_blip=False)
    viewsets: dict[str, ViewSet] = {}
    rows: list[dict[str, Any]] = []
    with scores_path.open("w", encoding="utf-8") as f:
        for index, (spec, source) in enumerate(candidates):
            task = get_task(spec.task)
            if spec.task not in viewsets:
                viewsets[spec.task] = ViewSet.build(task, seed=args.seed)
            viewset = viewsets[spec.task]
            name = f"{index:03d}_{spec.task}"
            out_dir = args.out / name
            write_prompt_card(out_dir, spec, source=source)

            images: list[tuple[str, int, Any]] = []  # (fidelity, steps, image)
            seconds: dict[int, float] = {}
            for steps in (spec.num_inference_steps, args.steps_low):
                s = replace(spec, num_inference_steps=steps)
                t0 = time.perf_counter()
                img64, img256 = engine.generate(s, viewset)
                seconds[steps] = time.perf_counter() - t0
                images += [
                    (f"256@{steps}", steps, img256),
                    (f"64@{steps}", steps, img64),
                ]

            for fidelity, steps, img in images:
                s = replace(spec, num_inference_steps=steps)
                verdict = judge.evaluate(img, s, viewset)
                sub = out_dir / fidelity.replace("@", "_")
                image_path = save_sample(img, sub)
                row = record_row(
                    Proposal(s, "fidelity", f"{fidelity} {source}"),
                    verdict,
                    0,
                    image_path,
                    [],
                )
                row.update(
                    {
                        "candidate": name,
                        "source": source,
                        "fidelity": fidelity,
                        "steps": steps,
                        "size": int(img.shape[-1]),
                        "seconds": seconds[steps],
                    }
                )
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                rows.append(row)
            summary = " ".join(
                f"{fid}={float(r['sep_min']):+.3f}"
                for fid, r in zip([i[0] for i in images], rows[-4:], strict=True)
            )
            print(f"[{index + 1}/{len(candidates)}] {name} {source}: {summary}")
            torch.cuda.empty_cache()

    result = analyse(rows)
    (args.out / "fidelity.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
