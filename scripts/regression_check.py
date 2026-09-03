"""Does this environment still reproduce an earlier run?

Written to be run once before a dependency upgrade and again after it. The
reference is not a snapshot taken for the occasion -- it is the run itself. A
run directory holds, for every candidate, the prompts, the seed, the generated
PNG, the scored views, and every CLIP score. That is a complete record of what
this code did, so "still works" can be asked as "produces the same bytes and
the same numbers".

Three axes, independent and in increasing cost, because they break for
different reasons and a single pass/fail would not say which:

  views       recompute the perceptual views from the stored PNG, compare to
              the stored view images -> torchvision. No model, effectively free.
  judge       re-score the stored PNG, compare to scores.jsonl
              -> transformers / CLIP. Cheap; no generation involved.
  generation  regenerate from the spec and seed, compare to sample_256.png
              -> diffusers / torch / visual_anagrams. ~30 s per candidate.

Two guards make the comparison mean what it claims:

  spec identity   the reconstructed CandidateSpec must hash to the uid stored
                  in the row. Otherwise a "difference" could just be a
                  different prompt being generated. Runs written before the
                  N-view spec (rows with `prompt_low` / `prompt_high`) cannot
                  pass this guard, because the uid formula changed; they are
                  still compared, and the mismatch is reported as such.
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

    .venv/bin/python -m scripts.regression_check --run runs/myrun --label pre-upgrade
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ava.image.engine import save_sample
from ava.image.judge import ClipBlipJudge, load_image
from ava.image.tasks import get_task
from ava.image.views import ViewSet
from ava.report import load_scores
from ava.spec import CandidateSpec


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def is_legacy(row: dict[str, Any]) -> bool:
    """Rows from before the N-view spec name their two prompts explicitly."""
    return "prompt_low" in row and "task" not in row


def spec_from_row(row: dict[str, Any]) -> CandidateSpec:
    """Rebuild the exact point in the search space that produced a row.

    guidance_scale and num_inference_steps are not in scores.jsonl; the run's
    config.yaml used the CandidateSpec defaults, and the uid check below is
    what actually proves the reconstruction is right.
    """
    if is_legacy(row):
        return CandidateSpec(
            task="hybrid",
            prompts=(str(row["prompt_low"]), str(row["prompt_high"])),
            style=str(row["style"]),
            seed=int(row["seed"]),
        )
    return CandidateSpec(
        task=str(row["task"]),
        prompts=tuple(str(p) for p in row["prompts"]),
        style=str(row["style"]),
        seed=int(row["seed"]),
        ref_image=row.get("ref_image"),
    )


def stored_scores(row: dict[str, Any]) -> dict[str, Any]:
    """The stored raw matrix, per-view probabilities and margins, in one shape."""
    if is_legacy(row):
        return {
            "scores": [
                [float(row["s_far_low"]), float(row["s_far_high"])],
                [float(row["s_near_low"]), float(row["s_near_high"])],
            ],
            "p": [float(row["p_far"]), float(row["p_near"])],
            "sep": [float(row["sep_far"]), float(row["sep_near"])],
            "j": float(row["j"]),
            "sep_min": float(row["sep_min"]),
            "diagnosis": str(row["diagnosis"]),
            "view_paths": [None, str(row["far_image_path"])],
        }
    return {
        "scores": row["scores"],
        "p": row["p"],
        "sep": row["sep"],
        "j": float(row["j"]),
        "sep_min": float(row["sep_min"]),
        "diagnosis": str(row["diagnosis"]),
        "view_paths": list(row["view_paths"]),
    }


def _flat(matrix: list[list[float]]) -> list[float]:
    return [float(x) for r in matrix for x in r]


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


class ViewSets:
    """Builds each task's views once, with the run's view seed."""

    def __init__(self, view_seed: int) -> None:
        self.view_seed = view_seed
        self._cache: dict[str, ViewSet] = {}

    def get(self, task: str) -> ViewSet:
        if task not in self._cache:
            self._cache[task] = ViewSet.build(get_task(task), seed=self.view_seed)
        return self._cache[task]


def check_views(
    rows: list[dict[str, Any]], work: Path, viewsets: ViewSets
) -> list[dict[str, Any]]:
    """Axis: do the perceptual views still produce the stored view images?"""
    from torchvision.utils import save_image

    out: list[dict[str, Any]] = []
    for row in rows:
        spec = spec_from_row(row)
        viewset = viewsets.get(spec.task)
        image = load_image(str(row["image_path"]))
        stored = stored_scores(row)["view_paths"]
        entry: dict[str, Any] = {"uid": row["uid"], "views": {}}
        for slot, view, path in zip(
            viewset.task.slot_names, viewset.perceive(image), stored, strict=True
        ):
            if path is None:
                continue
            recomputed = work / str(row["uid"]) / f"view_{slot}_recomputed.png"
            recomputed.parent.mkdir(parents=True, exist_ok=True)
            save_image(view, recomputed, padding=0)
            entry["views"][slot] = compare_images(Path(path), recomputed)
        entry["same_bytes"] = all(v["same_bytes"] for v in entry["views"].values())
        entry["max_abs_diff_8bit"] = max(
            (v.get("max_abs_diff_8bit", 255.0) for v in entry["views"].values()),
            default=0.0,
        )
        out.append(entry)
    return out


def check_judge(
    rows: list[dict[str, Any]], judge: ClipBlipJudge, viewsets: ViewSets
) -> list[dict[str, Any]]:
    """Axis: does CLIP still give the stored image the stored scores?"""
    out: list[dict[str, Any]] = []
    for row in rows:
        spec = spec_from_row(row)
        viewset = viewsets.get(spec.task)
        fresh = json.loads(
            judge.evaluate(load_image(str(row["image_path"])), spec, viewset).to_json()
        )
        stored = stored_scores(row)
        out.append(
            {
                "uid": row["uid"],
                "legacy_row": is_legacy(row),
                "uid_matches_spec": spec.uid() == row["uid"],
                "raw_max_delta": max(
                    abs(a - b)
                    for a, b in zip(
                        _flat(fresh["scores"]), _flat(stored["scores"]), strict=True
                    )
                ),
                "derived": {
                    "p": max(
                        abs(float(a) - float(b))
                        for a, b in zip(fresh["p"], stored["p"], strict=True)
                    ),
                    "sep": max(
                        abs(float(a) - float(b))
                        for a, b in zip(fresh["sep"], stored["sep"], strict=True)
                    ),
                    "j": abs(float(fresh["j"]) - stored["j"]),
                    "sep_min": abs(float(fresh["sep_min"]) - stored["sep_min"]),
                },
                "diagnosis_stored": _modern_diagnosis(stored["diagnosis"], viewset),
                "diagnosis_now": fresh["diagnosis"],
            }
        )
    return out


def _modern_diagnosis(diagnosis: str, viewset: ViewSet) -> str:
    """Map the two-view diagnosis names onto the N-view ones."""
    legacy = {
        "pair_mismatch": "all_lost",
        "low_loses": f"lost:{viewset.task.slot_names[0]}",
        "high_absent": f"lost:{viewset.task.slot_names[1]}",
    }
    return legacy.get(diagnosis, diagnosis)


def check_generation(
    rows: list[dict[str, Any]],
    work: Path,
    device: str,
    judge: ClipBlipJudge,
    viewsets: ViewSets,
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
    from ava.image.engine import Engine

    engine = Engine(device=device)
    results: list[dict[str, Any]] = []
    control: dict[str, Any] = {}

    for index, row in enumerate(rows):
        spec = spec_from_row(row)
        viewset = viewsets.get(spec.task)
        stored = stored_scores(row)
        image_path = save_sample(
            engine.generate(spec, viewset)[1], work / str(row["uid"])
        )
        fresh = json.loads(
            judge.evaluate(load_image(image_path), spec, viewset).to_json()
        )

        if index == 0:
            again_path = save_sample(
                engine.generate(spec, viewset)[1], work / f"{row['uid']}_again"
            )
            twice = json.loads(
                judge.evaluate(load_image(again_path), spec, viewset).to_json()
            )
            control = {
                "uid": row["uid"],
                "j_delta": abs(float(twice["j"]) - float(fresh["j"])),
                "sep_min_delta": abs(float(twice["sep_min"]) - float(fresh["sep_min"])),
                "diagnosis_unchanged": twice["diagnosis"] == fresh["diagnosis"],
                **compare_images(image_path, again_path),
            }

        stored_diagnosis = _modern_diagnosis(stored["diagnosis"], viewset)
        results.append(
            {
                "uid": row["uid"],
                "uid_matches_spec": spec.uid() == row["uid"],
                "j_stored": stored["j"],
                "j_now": float(fresh["j"]),
                "j_delta": abs(float(fresh["j"]) - stored["j"]),
                "sep_min_delta": abs(float(fresh["sep_min"]) - stored["sep_min"]),
                "diagnosis_stored": stored_diagnosis,
                "diagnosis_now": fresh["diagnosis"],
                "diagnosis_unchanged": fresh["diagnosis"] == stored_diagnosis,
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

    viewsets = ViewSets(view_seed=args.view_seed)
    work = Path(args.out) / args.label
    result: dict[str, Any] = {
        "label": args.label,
        "run": str(args.run),
        "candidates_available": len(rows),
        "legacy_rows": sum(is_legacy(r) for r in rows),
        "versions": _versions(),
        "views": check_views(rows, work, viewsets),
    }

    judge = ClipBlipJudge(device=args.device, use_blip=False)
    result["judge"] = check_judge(rows, judge, viewsets)

    if args.generate > 0:
        result["generation"] = check_generation(
            rows[: args.generate], work, args.device, judge, viewsets
        )

    result["summary"] = {
        "views_all_identical": all(v["same_bytes"] for v in result["views"]),
        "views_worst_max_abs_diff_8bit": worst(result["views"], "max_abs_diff_8bit"),
        "judge_all_uids_match": all(j["uid_matches_spec"] for j in result["judge"]),
        "judge_worst_raw_delta": worst(result["judge"], "raw_max_delta"),
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
        "--view-seed",
        type=int,
        default=0,
        help="the run's view seed (config.yaml: view_seed); randomised views only",
    )
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
    print(
        f"candidates: {result['candidates_available']} "
        f"({result['legacy_rows']} written before the N-view spec)\n"
    )
    for key, value in result["summary"].items():
        print(f"  {key:38s} {value}")


if __name__ == "__main__":
    main()
