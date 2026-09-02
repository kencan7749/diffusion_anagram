"""Does this environment still reproduce runs_gpt2?

Written to be run once before a dependency upgrade and again after it. The
reference is not a snapshot taken for the occasion -- it is the run itself.
`runs_gpt2/search` holds, for all 128 candidates, the prompts, the seed, the
generated PNG, the far view, and every CLIP score. That is a complete record
of what this code did, so "still works" can be asked as "produces the same
bytes and the same numbers".

Three axes, independent and in increasing cost, because they break for
different reasons and a single pass/fail would not say which:

  views       recompute far_view() from the stored PNG, compare to far.png
              -> torchvision. No model, runs on everything, effectively free.
  judge       re-score the stored PNG, compare to scores.jsonl
              -> transformers / CLIP. Cheap; no generation involved.
  generation  regenerate from the spec and seed, compare to sample_256.png
              -> diffusers / torch / visual_anagrams. ~30 s per candidate.

Two guards make the comparison mean what it claims:

  spec identity   the reconstructed CandidateSpec must hash to the uid stored
                  in the row. Otherwise a "difference" could just be a
                  different prompt being generated.
  noise floor     one spec is generated twice in this environment, and both
                  are scored. Sampling is not bit-reproducible in general, so
                  the gap between two runs made minutes apart on one machine
                  is the yardstick: an upgrade has changed something only if
                  it moves a candidate further than that.

The criterion is not byte equality. What "still works" means here is that a
candidate lands in the same place: the same diagnosis, and J within the noise
floor. Hashes and pixel deltas are recorded as diagnostics -- they say how
much moved, not whether it mattered -- and every raw number is written out so
a tolerance can be chosen after seeing them rather than before.

    .venv/bin/python -m scripts.regression_check --label pre-upgrade
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ava.engine import save_sample
from ava.judge import ClipBlipJudge, load_image
from ava.perceive import far_view
from ava.report import load_scores
from ava.spec import CandidateSpec

# The four raw cosines. These are what everything else is derived from, so a
# change here is the root cause and j / sep_min are only symptoms.
SCORE_KEYS = ("s_far_low", "s_far_high", "s_near_low", "s_near_high")
DERIVED_KEYS = ("p_far", "p_near", "j", "sep_far", "sep_near", "sep_min")


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def spec_from_row(row: dict[str, Any]) -> CandidateSpec:
    """Rebuild the exact point in the search space that produced a row.

    guidance_scale and num_inference_steps are not in scores.jsonl; the run's
    config.yaml used the CandidateSpec defaults, and the uid check below is
    what actually proves the reconstruction is right.
    """
    return CandidateSpec(
        prompt_low=str(row["prompt_low"]),
        prompt_high=str(row["prompt_high"]),
        style=str(row["style"]),
        seed=int(row["seed"]),
    )


def compare_images(a: Path, b: Path) -> dict[str, Any]:
    """Byte equality first, then how far apart they are if they are not."""
    same_bytes = md5(a) == md5(b)
    x = (load_image(a) * 255.0).round()
    y = (load_image(b) * 255.0).round()
    if x.shape != y.shape:
        return {"same_bytes": False, "shape_a": list(x.shape), "shape_b": list(y.shape)}
    diff = (x - y).abs()
    return {
        "same_bytes": same_bytes,
        "max_abs_diff_8bit": float(diff.max()),
        "mean_abs_diff_8bit": float(diff.mean()),
        "fraction_pixels_differing": float((diff > 0).float().mean()),
    }


def check_views(rows: list[dict[str, Any]], work: Path) -> list[dict[str, Any]]:
    """Axis: does far_view() still produce the far.png that was stored?"""
    from torchvision.utils import save_image

    out: list[dict[str, Any]] = []
    for row in rows:
        stored_far = Path(str(row["far_image_path"]))
        image = load_image(str(row["image_path"]))
        recomputed = work / str(row["uid"]) / "far_recomputed.png"
        recomputed.parent.mkdir(parents=True, exist_ok=True)
        save_image(far_view(image), recomputed, padding=0)
        out.append({"uid": row["uid"], **compare_images(stored_far, recomputed)})
    return out


def check_judge(
    rows: list[dict[str, Any]], judge: ClipBlipJudge
) -> list[dict[str, Any]]:
    """Axis: does CLIP still give the stored image the stored scores?"""
    out: list[dict[str, Any]] = []
    for row in rows:
        spec = spec_from_row(row)
        verdict = judge.evaluate(load_image(str(row["image_path"])), spec)
        fresh = json.loads(verdict.to_json())
        out.append(
            {
                "uid": row["uid"],
                "uid_matches_spec": spec.uid() == row["uid"],
                "raw": {k: abs(float(fresh[k]) - float(row[k])) for k in SCORE_KEYS},
                "derived": {
                    k: abs(float(fresh[k]) - float(row[k])) for k in DERIVED_KEYS
                },
                "diagnosis_stored": row["diagnosis"],
                "diagnosis_now": fresh["diagnosis"],
            }
        )
    return out


def check_generation(
    rows: list[dict[str, Any]], work: Path, device: str, judge: ClipBlipJudge
) -> dict[str, Any]:
    """Axis: does the sampler still put a candidate in the same place?

    Each regenerated image is scored, because that is the question the search
    loop actually cares about -- a candidate that was "ok" must still be "ok"
    and must still have roughly the same J. Pixel comparison rides along as a
    diagnostic for how far the sampler moved.

    The noise floor is measured first: the same spec is generated twice here
    and both are scored. Whatever J gap that produces is the amount of
    movement this environment generates on its own, and no smaller difference
    after an upgrade can be attributed to the upgrade.
    """
    from ava.engine import Engine

    engine = Engine(device=device)
    results: list[dict[str, Any]] = []
    control: dict[str, Any] = {}

    for index, row in enumerate(rows):
        spec = spec_from_row(row)
        image_path = save_sample(engine.generate(spec)[1], work / str(row["uid"]))
        fresh = json.loads(judge.evaluate(load_image(image_path), spec).to_json())

        if index == 0:
            again_path = save_sample(
                engine.generate(spec)[1], work / f"{row['uid']}_again"
            )
            twice = json.loads(judge.evaluate(load_image(again_path), spec).to_json())
            control = {
                "uid": row["uid"],
                "j_delta": abs(float(twice["j"]) - float(fresh["j"])),
                "sep_min_delta": abs(float(twice["sep_min"]) - float(fresh["sep_min"])),
                "diagnosis_unchanged": twice["diagnosis"] == fresh["diagnosis"],
                **compare_images(image_path, again_path),
            }

        results.append(
            {
                "uid": row["uid"],
                "uid_matches_spec": spec.uid() == row["uid"],
                "j_stored": float(row["j"]),
                "j_now": float(fresh["j"]),
                "j_delta": abs(float(fresh["j"]) - float(row["j"])),
                "sep_min_delta": abs(float(fresh["sep_min"]) - float(row["sep_min"])),
                "diagnosis_stored": row["diagnosis"],
                "diagnosis_now": fresh["diagnosis"],
                "diagnosis_unchanged": fresh["diagnosis"] == row["diagnosis"],
                **compare_images(Path(str(row["image_path"])), image_path),
            }
        )

    return {"noise_floor": control, "candidates": results}


def worst(entries: list[dict[str, Any]], key: str) -> float:
    values = [float(e[key]) for e in entries if key in e]
    return max(values) if values else 0.0


def run(args: argparse.Namespace) -> Path:
    rows = [
        row
        for path in sorted(Path(args.run).glob("round_*/scores.jsonl"))
        for row in load_scores(path)
    ]
    if not rows:
        raise SystemExit(f"no scores found under {args.run}")

    work = Path(args.out) / args.label
    result: dict[str, Any] = {
        "label": args.label,
        "run": str(args.run),
        "candidates_available": len(rows),
        "versions": _versions(),
        "views": check_views(rows, work),
    }

    judge = ClipBlipJudge(device=args.device, use_blip=False)
    result["judge"] = check_judge(rows, judge)

    if args.generate > 0:
        result["generation"] = check_generation(
            rows[: args.generate], work, args.device, judge
        )

    result["summary"] = {
        "views_all_identical": all(v["same_bytes"] for v in result["views"]),
        "views_worst_max_abs_diff_8bit": worst(result["views"], "max_abs_diff_8bit"),
        "judge_all_uids_match": all(j["uid_matches_spec"] for j in result["judge"]),
        "judge_worst_raw_delta": max(
            (max(j["raw"].values()) for j in result["judge"]), default=0.0
        ),
        "judge_worst_derived_delta": max(
            (max(j["derived"].values()) for j in result["judge"]), default=0.0
        ),
        "judge_diagnoses_unchanged": all(
            j["diagnosis_stored"] == j["diagnosis_now"] for j in result["judge"]
        ),
    }
    if "generation" in result:
        candidates = result["generation"]["candidates"]
        floor = result["generation"]["noise_floor"]
        # The criterion.
        result["summary"]["generation_diagnoses_unchanged"] = all(
            c["diagnosis_unchanged"] for c in candidates
        )
        result["summary"]["generation_worst_j_delta"] = worst(candidates, "j_delta")
        result["summary"]["noise_floor_j_delta"] = float(floor.get("j_delta", 0.0))
        # Diagnostics: how far things moved, not whether it mattered.
        result["summary"]["generation_all_bytes_identical"] = all(
            c["same_bytes"] for c in candidates
        )
        result["summary"]["generation_worst_max_abs_diff_8bit"] = worst(
            candidates, "max_abs_diff_8bit"
        )
        result["summary"]["sampling_is_bit_reproducible"] = bool(
            floor.get("same_bytes")
        )

    work.mkdir(parents=True, exist_ok=True)
    path = Path(args.out) / f"{args.label}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _versions() -> dict[str, str]:
    """Recorded so a later diff can be attributed to a specific bump."""
    import diffusers
    import torchvision
    import transformers

    return {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs_gpt2/search"))
    parser.add_argument("--label", required=True, help="e.g. pre-upgrade")
    parser.add_argument("--out", type=Path, default=Path("results/regression"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--generate",
        type=int,
        default=4,
        help="candidates to regenerate; 0 skips the expensive axis",
    )
    args = parser.parse_args()

    path = run(args)
    result = json.loads(path.read_text(encoding="utf-8"))

    print(f"wrote {path}\n")
    print("versions: " + ", ".join(f"{k} {v}" for k, v in result["versions"].items()))
    print(f"candidates: {result['candidates_available']}\n")
    for key, value in result["summary"].items():
        print(f"  {key:38s} {value}")


if __name__ == "__main__":
    main()
