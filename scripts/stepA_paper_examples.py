"""Step A: does the N-view wiring reproduce the old hybrid, and run every task?

Three questions, answered in increasing cost and persisted for the report:

  regression   Regenerating a hybrid from an earlier run with the new engine
               must give the same bytes. The wiring was changed; the behaviour
               must not have been. Compared against runs/smoke by md5.
  coverage     One paper example per registered task is generated and judged,
               with the paper's own prompts. The papers hand-picked these, so
               they should mostly hold; a task that fails outright is looked at
               individually, not averaged away.
  judge check  Each two-view Visual Anagrams image is also scored under every
               *other* two-view task's perceptual view. If the judge means
               anything, the transformed view the image was made for should
               read its prompt best. Only the transformed view's margin is
               compared: view 0 is the identity for every VA task, so its row
               of the score matrix is the same under every task and would tie.

Analysis only. Writes results/stepA/scores.jsonl (rows in the loop's own row
format, so the contact sheet is drawn by render_stepA.py from the file alone),
regression.json and summary.json.

Run from the repository root:
    .venv/bin/python -m scripts.stepA_paper_examples
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from ava.image.engine import Engine
from ava.image.judge import ClipBlipJudge
from ava.image.tasks import TASKS, VA, get_task
from ava.image.views import ViewSet
from ava.loop import evaluate_candidate, record_row
from ava.propose import Proposal
from ava.spec import CandidateSpec

# One example per task, from the papers or the upstream readme. Written in the
# task's slot order (see ava.image.paper_examples for the conventions).
EXAMPLES: dict[str, tuple[tuple[str, ...], str, str]] = {
    # task: (prompts, style, citation)
    "flip": (("people at a campfire", "an old man"), "an oil painting of", "VA Fig. 2"),
    "rotate_cw": (
        ("a snowy mountain village", "a horse"),
        "an oil painting of",
        "VA Fig. 8",
    ),
    "rotate_ccw": (
        ("a snowy mountain village", "a horse"),
        "an oil painting of",
        "readme",
    ),
    "rotate_180": (
        ("a snowy mountain village", "a horse"),
        "an oil painting of",
        "VA Sec. 3.4",
    ),
    "skew": (("a tudor portrait", "a skull"), "an oil painting of", "VA Fig. 1"),
    "jigsaw": (("houseplants", "marilyn monroe"), "a painting of", "VA Fig. 1"),
    "inner_circle": (
        ("albert einstein", "marilyn monroe"),
        "a pop art of",
        "VA Fig. 1",
    ),
    "negate": (("a landscape", "houseplants"), "a lithograph of", "VA Fig. 1"),
    "patch_permute": (("a lemur", "a kangaroo"), "a pencil sketch of", "VA Fig. 6"),
    "pixel_permute": (("a duck", "a rabbit"), "a mosaic of", "VA Fig. 6"),
    "square_hinge": (("a duck", "a rabbit"), "a water color of", "readme"),
    "three_view": (
        ("a teddy bear", "a rabbit", "waterfalls"),
        "a painting of",
        "VA Fig. 1",
    ),
    "four_view": (
        ("a rabbit", "a giraffe", "a teddy bear", "a bird"),
        "an oil painting of",
        "VA Fig. 1",
    ),
    "hybrid": (("marilyn monroe", "houseplants"), "a photo of", "FD Fig. 1"),
    "triple_hybrid": (
        ("a yin yang", "a skull", "waterfalls"),
        "a lithograph of",
        "FD readme",
    ),
    "color_hybrid": (
        ("houseplants", "the statue of liberty"),
        "a watercolor of",
        "FD Fig. 1",
    ),
    "motion_hybrid": (("a car", "a canyon"), "a photo of", "FD Fig. 1"),
    "inverse_hybrid": (
        ("albert einstein", "waterfalls"),
        "a lithograph of",
        "FD readme",
    ),
}
REF_IMAGE = "dev/visual_anagrams/assets/einstein.png"

# The candidate regenerated for the byte-identity check: the first candidate
# runs/smoke generated, made by the two-view engine before this refactor. It has
# to be the *first* generation of its process: sampling here is reproducible
# across processes only for the same position in the call sequence (the
# regression checks in results/regression record the same fact as
# `sampling_is_bit_reproducible: false` next to `generation_all_bytes_identical:
# true`), so this check is also the first generation of this process.
REGRESSION_SPEC = CandidateSpec(
    "hybrid", ("a panda", "a flower arrangement"), "a mosaic of"
)
REGRESSION_IMAGE = Path("runs/smoke/round_000/74363fca40e0/sample_256.png")


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def spec_for(task: str, seed: int) -> CandidateSpec:
    prompts, style, _ = EXAMPLES[task]
    return CandidateSpec(
        task=task,
        prompts=prompts,
        style=style,
        seed=seed,
        ref_image=REF_IMAGE if get_task(task).ref_slot is not None else None,
    )


def check_regression(engine: Engine, out: Path) -> dict[str, Any]:
    viewset = ViewSet.build(get_task("hybrid"))
    _, image = engine.generate(REGRESSION_SPEC, viewset)
    from ava.image.engine import save_sample

    path = save_sample(image, out / "regression")
    result: dict[str, Any] = {
        "spec": REGRESSION_SPEC.__dict__ | {"prompts": list(REGRESSION_SPEC.prompts)},
        "regenerated": str(path),
        "reference": str(REGRESSION_IMAGE),
        "reference_exists": REGRESSION_IMAGE.exists(),
    }
    if REGRESSION_IMAGE.exists():
        result["md5_reference"] = md5(REGRESSION_IMAGE)
        result["md5_regenerated"] = md5(path)
        result["identical"] = result["md5_reference"] == result["md5_regenerated"]
    return result


def cross_judge(
    judge: ClipBlipJudge,
    rows: list[dict[str, Any]],
    viewsets: dict[str, ViewSet],
) -> list[dict[str, Any]]:
    """Score each two-view VA image under every other two-view VA task's views.

    Compares the *transformed* view's margin (`sep[1]`), not `sep_min`: the
    identity view is shared by every VA task, so `sep_min` ties whenever the
    identity view is the weaker one.
    """
    from ava.image.judge import load_image

    two_view_va = [
        t.name
        for t in TASKS.values()
        if t.paper == VA and t.n_views == 2 and t.name in viewsets
    ]
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["task"] not in two_view_va:
            continue
        image = load_image(str(row["image_path"]))
        spec = CandidateSpec(row["task"], tuple(row["prompts"]), str(row["style"]))
        scored: dict[str, float] = {}
        for other in two_view_va:
            # Same prompts, judged as if the image had been made for `other`.
            as_other = CandidateSpec(other, spec.prompts, spec.style)
            scored[other] = judge.evaluate(image, as_other, viewsets[other]).sep[1]
        best = max(scored, key=lambda k: scored[k])
        out.append(
            {
                "task": row["task"],
                "transformed_sep_by_view": scored,
                "best_view": best,
                "own_view_is_best": best == row["task"],
                "own_minus_best_other": scored[row["task"]]
                - max(v for k, v in scored.items() if k != row["task"]),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/stepA"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks", default=",".join(EXAMPLES), help="comma-separated")
    parser.add_argument("--skip-regression", action="store_true")
    parser.add_argument(
        "--regression-only", action="store_true", help="only the byte-identity check"
    )
    parser.add_argument(
        "--cross-only",
        action="store_true",
        help="recompute the judge check from the persisted images; no generation",
    )
    args = parser.parse_args()
    tasks = [t for t in args.tasks.split(",") if t]
    args.out.mkdir(parents=True, exist_ok=True)

    judge = ClipBlipJudge(device=args.device, use_blip=True)
    if args.cross_only:
        from ava.report import load_scores

        saved = load_scores(args.out / "scores.jsonl")
        viewsets = {t: ViewSet.build(get_task(t), seed=args.seed) for t in tasks}
        cross = cross_judge(judge, saved, viewsets)
        (args.out / "cross_judge.json").write_text(
            json.dumps(cross, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        for c in cross:
            print(
                f"[{c['task']:15s}] own={c['transformed_sep_by_view'][c['task']]:+.3f} "
                f"best={c['best_view']} margin={c['own_minus_best_other']:+.3f}"
            )
        print(
            f"own view best: {sum(c['own_view_is_best'] for c in cross)}/{len(cross)}"
        )
        return

    engine = Engine(device=args.device)

    regression: dict[str, Any] = {}
    if not args.skip_regression:
        regression = check_regression(engine, args.out)
        (args.out / "regression.json").write_text(json.dumps(regression, indent=2))
        print(f"[regression] {regression}")
        if args.regression_only:
            return

    viewsets = {t: ViewSet.build(get_task(t), seed=args.seed) for t in tasks}
    rows: list[dict[str, Any]] = []
    with (args.out / "scores.jsonl").open("w", encoding="utf-8") as f:
        for task in tasks:
            spec = spec_for(task, args.seed)
            proposal = Proposal(spec, "paper_example", EXAMPLES[task][2])
            verdict, _, image_path, view_paths = evaluate_candidate(
                proposal, engine, judge, viewsets[task], args.out / task
            )
            row = record_row(proposal, verdict, 0, image_path, view_paths)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            rows.append(row)
            print(
                f"[{task:15s}] J={verdict.j:.3f} sep_min={verdict.sep_min:+.3f} "
                f"A={verdict.alignment:.3f} C={verdict.concealment:.3f} "
                f"{verdict.diagnose()}"
            )
            torch.cuda.empty_cache()

    cross = cross_judge(judge, rows, viewsets)
    held = [r["task"] for r in rows if r["diagnosis"] == "ok"]
    summary = {
        "n_tasks": len(rows),
        "held": held,
        "failed": {r["task"]: r["diagnosis"] for r in rows if r["diagnosis"] != "ok"},
        "regression_identical": regression.get("identical"),
        "cross_judge": cross,
        "own_view_best_count": sum(c["own_view_is_best"] for c in cross),
        "own_view_checked": len(cross),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
