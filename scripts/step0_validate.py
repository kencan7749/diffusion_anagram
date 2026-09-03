"""Step 0 gate: does J actually measure whether the illusion holds?

The loop is not written until this passes. Three checks, per the plan:

  0-1 sign      the known-good example reads as its prompt in both views
  0-2 separation deliberately broken examples (generation sigma 1.0 / 6.0)
                score clearly below the good one
  0-3 eye       a human compares the rendered sheet against the J ranking

Only 0-1 and 0-2 are decided here; 0-3 needs the figure from render_step0.py.

Why sigma 1.0 and 6.0 break the image (view_hybrid.py's own note: "small sigma
=> low freq is easier to see, larger sigma => high freq easier to see"):

  sigma=1.0  the high-pass band is squeezed to the very finest detail, so
             the high prompt cannot be rendered -> p[high] should collapse
  sigma=6.0  the low-pass band is squeezed to the very coarsest structure, so
             the low prompt cannot be rendered  -> p[low] should collapse

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
from visual_anagrams.views.view_hybrid import HybridHighPassView, HybridLowPassView

from ava.image.engine import Engine, save_sample
from ava.image.judge import ClipBlipJudge, build_verdict, load_image, to_pil
from ava.image.perceive import far_view, far_view_resize, near_view
from ava.image.tasks import get_task
from ava.image.views import ViewSet
from ava.spec import KERNEL_SIZE, SIGMA, CandidateSpec

# The known-good example already in results/hybrid_smoke, reproduced here so
# the sigma sweep is otherwise identical to it.
REFERENCE_SPEC = CandidateSpec(
    task="hybrid",
    prompts=("a panda", "a flower arrangement"),
    style="a painting of",
    seed=0,
    guidance_scale=10.0,
    num_inference_steps=30,
)

# (label, generation sigma, expectation)
CASES: list[tuple[str, float, str]] = [
    ("sigma_1.0", 1.0, "high-pass band too narrow; expect p[high] to collapse"),
    ("sigma_2.0", SIGMA, "known-good setting; expect both views to hold"),
    ("sigma_6.0", 6.0, "low-pass band too narrow; expect p[low] to collapse"),
]

# Two far-view implementations, so the gate does not depend on one blur.
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


def hybrid_viewset_with_sigma(sigma: float) -> ViewSet:
    """The hybrid task's views with a deliberately different generation sigma.

    Only this gate does this; `ViewSet.build` always uses SIGMA.
    """
    task = get_task("hybrid")
    views = [
        HybridLowPassView(sigma=sigma, kernel_size=KERNEL_SIZE),
        HybridHighPassView(sigma=sigma, kernel_size=KERNEL_SIZE),
    ]
    return ViewSet(task=task, views=views, seed=0)


def generate_cases(paths: Step0Paths, device: str) -> dict[str, Path]:
    """Generate one image per sigma, reusing any that already exist."""
    wanted = {label: paths.images / label / "sample_256.png" for label, _, _ in CASES}
    todo = [(label, sigma) for label, sigma, _ in CASES if not wanted[label].exists()]
    if not todo:
        print("[generate] all images present, skipping generation")
        return wanted

    engine = Engine(device=device)
    for label, sigma in todo:
        print(f"[generate] {label} (generation sigma={sigma})")
        _, img_256 = engine.generate(REFERENCE_SPEC, hybrid_viewset_with_sigma(sigma))
        save_sample(img_256, paths.images / label)
    return wanted


def score_with_far_mode(
    judge: ClipBlipJudge, img: torch.Tensor, viewset: ViewSet, far_mode: str
) -> tuple[Any, list[torch.Tensor]]:
    """Score the hybrid under one far-view implementation; returns (verdict, views)."""
    far = far_view(img) if far_mode == "blur" else far_view_resize(img)
    views = [far, near_view(img)]
    pils = [to_pil(v) for v in views]
    s = (
        judge.image_embeddings(pils)
        @ judge.text_embeddings(REFERENCE_SPEC.full_prompts).T
    ).cpu()
    captions = [judge.caption(p) for p in pils]
    verdict = build_verdict(
        REFERENCE_SPEC, viewset, s, judge.logit_scale, captions, {"far_mode": far_mode}
    )
    return verdict, views


def evaluate_cases(
    images: dict[str, Path], paths: Step0Paths, device: str
) -> list[dict[str, Any]]:
    """Score every image under every far-view implementation."""
    judge = ClipBlipJudge(device=device)
    viewset = ViewSet.build(get_task("hybrid"))
    rows: list[dict[str, Any]] = []
    for far_mode in FAR_MODES:
        for label, gen_sigma, expectation in CASES:
            img = load_image(images[label])
            verdict, views = score_with_far_mode(judge, img, viewset, far_mode)

            # Persist the exact views that were scored, so render_step0.py can
            # composite the sheet without recomputing anything.
            far_path = paths.images / label / f"far_{far_mode}.png"
            far_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(views[0], far_path, padding=0)

            row = json.loads(verdict.to_json())
            row["case"] = label
            row["generation_sigma"] = gen_sigma
            row["expectation"] = expectation
            row["far_mode"] = far_mode
            row["image_path"] = str(images[label])
            row["far_image_path"] = str(far_path)
            row["prompts"] = REFERENCE_SPEC.full_prompts
            rows.append(row)
            print(
                f"[{far_mode:6s} {label}] p_low={verdict.p[0]:.4f} "
                f"p_high={verdict.p[1]:.4f} J={verdict.j:.4f} ({verdict.diagnose()})"
            )
            for slot, caption in zip(verdict.slots, verdict.captions, strict=True):
                print(f"         {slot:4s}: {caption!r}")

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

        sign_ok = all(good["holds"])
        # Separation: the good case must outscore both broken ones on J.
        margin = float(good["j"]) - max(float(b["j"]) for b in broken)
        separation_ok = margin > 0.0

        summary["far_modes"][far_mode] = {
            "sign_ok": sign_ok,
            "separation_ok": separation_ok,
            "j_margin": margin,
            "good": {k: good[k] for k in ("p", "j")},
            "broken": {str(b["case"]): {k: b[k] for k in ("p", "j")} for b in broken},
        }

    modes = summary["far_modes"]
    assert isinstance(modes, dict)
    summary["gate_passed"] = all(
        v["sign_ok"] and v["separation_ok"] for v in modes.values()
    )
    # The plan requires that the conclusion not depend on which far view is used.
    summary["far_modes_agree"] = (
        len({(v["sign_ok"], v["separation_ok"]) for v in modes.values()}) == 1
    )
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
