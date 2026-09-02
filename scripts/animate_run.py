"""Pre-render transition clips for a run's candidates, with the upstream animate.py.

Reads runs/<run>/round_*/scores.jsonl and writes `anim_<slot>.mp4` beside each
candidate's stills (`ava.image.animate`). Nothing is regenerated or re-scored:
the input is the persisted `sample_256.png`, the run's view objects (rebuilt
from `config.yaml` and `views/<task>.pt`) and the prompts. Clips that already
exist are kept unless `--force`.

Run from the repository root:
    .venv/bin/python -m scripts.animate_run --run runs/flip_evo_20260902 --held-only
    .venv/bin/python -m scripts.animate_run --run runs/hybrid_20260902 --top 8
    .venv/bin/python -m scripts.animate_run --run runs/x --uid b26999438da3

About 2.5 s and 250 kB per clip at 256 px on a CPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ava.image import animate
from ava.webui.data import read_jsonl, round_dirs


def load_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rd in round_dirs(root):
        rows.extend(read_jsonl(rd / "scores.jsonl"))
    return rows


def select_rows(
    rows: list[dict[str, Any]],
    *,
    held_only: bool,
    top: int | None,
    uids: list[str],
) -> list[dict[str, Any]]:
    """Which candidates to animate: explicit uids, else held / top-N filters."""
    if uids:
        wanted = set(uids)
        return [r for r in rows if r["uid"] in wanted]
    out = [r for r in rows if not held_only or r.get("diagnosis") == "ok"]
    if top is not None:
        out.sort(key=lambda r: float(r.get("sep_min", float("-inf"))), reverse=True)
        out = out[:top]
    return out


def animate_run(
    root: Path,
    rows: list[dict[str, Any]],
    *,
    force: bool = False,
    timing: animate.Timing = animate.DEFAULT_TIMING,
) -> dict[str, int]:
    """Animate `rows`; return counts by outcome. Unsupported tasks are reported once."""
    viewsets = animate.viewsets_for_run(root)
    counts = {"written": 0, "kept": 0, "unsupported": 0, "missing": 0}
    unsupported: set[str] = set()
    for row in rows:
        task = str(row["task"])
        if task in unsupported:
            counts["unsupported"] += 1
            continue
        try:
            before = {p for p in animate.candidate_dir(root, row).glob("anim_*.mp4")}
            paths = animate.animate_row(root, row, viewsets, timing=timing, force=force)
        except animate.UnsupportedView as e:
            print(f"[unsupported] {e}")
            unsupported.add(task)
            counts["unsupported"] += 1
            continue
        except FileNotFoundError as e:
            print(f"[missing] {row['uid']}: {e}")
            counts["missing"] += 1
            continue
        new = [p for p in paths if force or p not in before]
        counts["written" if new else "kept"] += 1
        print(
            f"[{'written' if new else 'kept'}] {row['uid']} {task}: "
            + ", ".join(p.name for p in paths)
        )
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="runs/<run> directory")
    parser.add_argument(
        "--held-only", action="store_true", help="only candidates whose illusion held"
    )
    parser.add_argument("--top", type=int, default=None, help="best N by sep_min")
    parser.add_argument("--uid", action="append", default=[], help="a candidate uid")
    parser.add_argument("--force", action="store_true", help="rewrite existing clips")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = select_rows(
        load_rows(args.run), held_only=args.held_only, top=args.top, uids=args.uid
    )
    print(f"{len(rows)} candidates in {args.run}")
    counts = animate_run(args.run, rows, force=args.force)
    print(", ".join(f"{k} {v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
