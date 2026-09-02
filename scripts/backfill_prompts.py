"""Write prompt.txt into candidate directories of runs made before it existed.

Reads only the persisted scores; generates nothing and modifies nothing that is
already there. Safe to run repeatedly.

    .venv/bin/python -m scripts.backfill_prompts runs/myrun
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ava.loop import write_prompt_card
from ava.report import load_scores
from ava.spec import CandidateSpec


def spec_from_row(row: dict[str, Any]) -> CandidateSpec:
    return CandidateSpec(
        task=str(row["task"]),
        prompts=tuple(str(p) for p in row["prompts"]),
        style=str(row["style"]),
        seed=int(row["seed"]),
        ref_image=row.get("ref_image"),
    )


def backfill(run_dir: Path, overwrite: bool = False) -> tuple[int, int]:
    """Returns (written, skipped)."""
    score_files = sorted(run_dir.glob("round_*/scores.jsonl"))
    score_files += sorted(run_dir.glob("harvest/scores.jsonl"))

    written = skipped = 0
    for path in score_files:
        for row in load_scores(path):
            out_dir = Path(str(row["image_path"])).parent
            if not out_dir.is_dir():
                skipped += 1
                continue
            if (out_dir / "prompt.txt").exists() and not overwrite:
                skipped += 1
                continue
            extra: dict[str, object] = {
                "origin": str(row.get("origin", "?")),
                "J": f"{float(row['j']):.4f}",
                "sep": f"{float(row.get('sep_min', 0.0)):+.4f}",
                "verdict": str(row.get("diagnosis", "?")),
            }
            for slot, caption in zip(
                row["slots"], row.get("captions", []), strict=False
            ):
                extra[f"cap {slot}"] = caption
            write_prompt_card(out_dir, spec_from_row(row), **extra)
            written += 1
    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="e.g. runs/myrun")
    parser.add_argument(
        "--overwrite", action="store_true", help="rewrite cards that already exist"
    )
    args = parser.parse_args()

    written, skipped = backfill(args.run_dir, args.overwrite)
    print(f"wrote {written} prompt.txt, skipped {skipped}")


if __name__ == "__main__":
    main()
