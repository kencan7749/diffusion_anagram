"""Step 0 gate: does J actually measure whether the illusion holds?

The loop is not written until this passes. Three checks, per the plan:

  0-1 sign      the known-good example scores p_far > 0.5 and p_near > 0.5
  0-2 separation deliberately broken examples (generation sigma 1.0 / 6.0)
                score clearly below the good one
  0-3 eye       a human compares the rendered sheet against the J ranking

Only 0-1 and 0-2 are decided here; 0-3 needs the figure from render_step0.py.

Why sigma 1.0 and 6.0 break the image (view_hybrid.py's own note: "small sigma
=> low freq is easier to see, larger sigma => high freq easier to see"):

  sigma=1.0  the high-pass band is squeezed to the very finest detail, so
             prompt_high cannot be rendered  -> p_near should collapse
  sigma=6.0  the low-pass band is squeezed to the very coarsest structure, so
             prompt_low cannot be rendered   -> p_far should collapse

The evaluation blur stays fixed at ava.spec.SIGMA in every case: the viewing
distance of the human is a constant, not a function of how the image was made.
Varying sigma is for this gate only; the search loop keeps it at 2.0.

Analysis only. Writes results/step0/scores.jsonl and summary.json; drawing is
render_step0.py's job.

Run from the repository root:
    .venv/bin/python -m scripts.step0_validate
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torchvision.utils import save_image

from ava.engine import Engine, save_sample
from ava.judge import ClipBlipJudge, load_image
from ava.spec import SIGMA, CandidateSpec

# The known-good example already in results/hybrid_smoke, reproduced here so
# the sigma sweep is otherwise identical to it.
REFERENCE_SPEC = CandidateSpec(
    prompt_low="a painting of a panda",
    prompt_high="a painting of a flower arrangement",
    seed=0,
    guidance_scale=10.0,
    num_inference_steps=30,
)

# (label, generation sigma, expectation)
CASES: list[tuple[str, float, str]] = [
    ("sigma_1.0", 1.0, "high-pass band too narrow; expect p_near to collapse"),
    ("sigma_2.0", SIGMA, "known-good setting; expect both above 0.5"),
    ("sigma_6.0", 6.0, "low-pass band too narrow; expect p_far to collapse"),
]

FAR_MODES = ("blur", "resize")


@dataclass(frozen=True)
class Step0Paths:
    root: Path

    @property
    def images(self) -> Path:
        return self.root / "images"

    @property
    def scores(self) -> Path:
        return self.root / "scores.jsonl"

    @property
    def summary(self) -> Path:
        return self.root / "summary.json"


def generate_cases(paths: Step0Paths, device: str) -> dict[str, Path]:
    """Generate one image per sigma, reusing any that already exist."""
    wanted = {
        label: paths.images / label / "sample_256.png" for label, _, _ in CASES
    }
    todo = [(label, sigma) for label, sigma, _ in CASES if not wanted[label].exists()]
    if not todo:
        print("[generate] all images present, skipping generation")
        return wanted

    engine = Engine(device=device)
    for label, sigma in todo:
        print(f"[generate] {label} (generation sigma={sigma})")
        _, img_256 = engine.generate(REFERENCE_SPEC, sigma=sigma)
        save_sample(img_256, paths.images / label)
    return wanted


def evaluate_cases(
    images: dict[str, Path], paths: Step0Paths, device: str
) -> list[dict[str, Any]]:
    """Score every image under every far-view implementation."""
    rows: list[dict[str, Any]] = []
    for far_mode in FAR_MODES:
        judge = ClipBlipJudge(device=device, far_mode=far_mode)
        for label, gen_sigma, expectation in CASES:
            img = load_image(images[label])
            verdict = judge.evaluate(img, REFERENCE_SPEC)

            # Persist the exact views that were scored, so render_step0.py can
            # composite the sheet without recomputing anything.
            far, _ = judge.views(img)
            far_path = paths.images / label / f"far_{far_mode}.png"
            far_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(far, far_path, padding=0)

            row = json.loads(verdict.to_json())
            row["case"] = label
            row["generation_sigma"] = gen_sigma
            row["expectation"] = expectation
            row["far_mode"] = far_mode
            row["image_path"] = str(images[label])
            row["far_image_path"] = str(far_path)
            row["prompt_low"] = REFERENCE_SPEC.full_low
            row["prompt_high"] = REFERENCE_SPEC.full_high
            rows.append(row)
            print(
                f"[{far_mode:6s} {label}] p_far={verdict.p_far:.4f} "
                f"p_near={verdict.p_near:.4f} J={verdict.j:.4f} "
                f"({verdict.diagnose()})"
            )
            print(f"         far : {verdict.caption_far!r}")
            print(f"         near: {verdict.caption_near!r}")
        del judge
        torch.cuda.empty_cache()

    with paths.scores.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def check_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Decide 0-1 and 0-2 from the scores, per far-view implementation."""
    summary: dict[str, Any] = {"far_modes": {}}
    for far_mode in FAR_MODES:
        by_case = {r["case"]: r for r in rows if r["far_mode"] == far_mode}
        good = by_case["sigma_2.0"]
        broken = [by_case["sigma_1.0"], by_case["sigma_6.0"]]

        sign_ok = bool(good["p_far"] > 0.5 and good["p_near"] > 0.5)
        # Separation: the good case must outscore both broken ones on J.
        margin = float(good["j"]) - max(float(b["j"]) for b in broken)
        separation_ok = margin > 0.0

        summary["far_modes"][far_mode] = {
            "sign_ok": sign_ok,
            "separation_ok": separation_ok,
            "j_margin": margin,
            "good": {k: good[k] for k in ("p_far", "p_near", "j")},
            "broken": {
                str(b["case"]): {k: b[k] for k in ("p_far", "p_near", "j")}
                for b in broken
            },
        }

    modes = summary["far_modes"]
    assert isinstance(modes, dict)
    summary["gate_passed"] = all(
        v["sign_ok"] and v["separation_ok"] for v in modes.values()
    )
    # The plan requires that the conclusion not depend on which far view is used.
    summary["far_modes_agree"] = len(
        {(v["sign_ok"], v["separation_ok"]) for v in modes.values()}
    ) == 1
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/step0"))
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    paths = Step0Paths(args.out)
    paths.root.mkdir(parents=True, exist_ok=True)

    images = generate_cases(paths, args.device)
    rows = evaluate_cases(images, paths, args.device)
    summary = check_gate(rows)

    paths.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nGATE: {'PASS' if summary['gate_passed'] else 'FAIL'}")
    print("0-3 (visual check) still requires: .venv/bin/python -m scripts.render_step0")


if __name__ == "__main__":
    main()
