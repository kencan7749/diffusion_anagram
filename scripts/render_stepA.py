"""Draw the Step A contact sheet: one paper example per task, every view shown.

Reads only results/stepA/scores.jsonl and the view images the analysis step
persisted. Never regenerates, never re-judges.

Run from the repository root:
    .venv/bin/python -m scripts.render_stepA
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ava.report import build_contact_sheet, load_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/stepA"))
    parser.add_argument("--columns", type=int, default=6)
    args = parser.parse_args()

    rows = load_scores(args.root / "scores.jsonl")
    out = build_contact_sheet(
        rows,
        args.root / "figures" / "stepA_contact.png",
        columns=args.columns,
        min_j=None,
    )
    print(f"wrote {out} ({len(rows)} tasks)")


if __name__ == "__main__":
    main()
