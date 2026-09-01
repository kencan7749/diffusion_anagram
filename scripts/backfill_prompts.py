"""Write prompt.txt into candidate directories of runs made before it existed.

Reads only the persisted scores; generates nothing and modifies nothing that is
already there. Safe to run repeatedly.

    .venv/bin/python -m scripts.backfill_prompts runs_gpt2/search
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ava.loop import write_prompt_card
from ava.report import load_scores
from ava.spec import CandidateSpec


def spec_from_row(row: dict) -> CandidateSpec:
    return CandidateSpec(
        prompt_low=row["prompt_low"],
        prompt_high=row["prompt_high"],
        style=row["style"],
        seed=int(row["seed"]),
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
            write_prompt_card(
                out_dir,
                spec_from_row(row),
                origin=str(row.get("origin", "?")),
                J=f"{float(row['j']):.4f}",
                sep=f"{float(row.get('sep_min', 0.0)):+.4f}",
                verdict=str(row.get("diagnosis", "?")),
                far_cap=str(row.get("caption_far", "")),
                near_cap=str(row.get("caption_near", "")),
            )
            written += 1
    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="e.g. runs_gpt2/search")
    parser.add_argument(
        "--overwrite", action="store_true", help="rewrite cards that already exist"
    )
    args = parser.parse_args()

    written, skipped = backfill(args.run_dir, args.overwrite)
    print(f"wrote {written} prompt.txt, skipped {skipped}")


if __name__ == "__main__":
    main()
