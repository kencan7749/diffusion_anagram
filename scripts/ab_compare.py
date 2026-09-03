"""Compare a v1 (bandit) run with a v2 (evolve) run at the same budget.

Reads only what the two runs persisted: every round's scores.jsonl, the v2
run's clusters.npz (the cell map both arms are measured on, so coverage
means the same thing for both) and its search_state.json. Nothing is
regenerated or re-judged; the only model touched is CLIP's text tower, to
place the v1 run's prompts on the v2 run's cluster map.

The two numbers the design document makes the adoption decision on
(design_search_v2.md Sec. 6):

    yield      pairs that held (diagnosis `ok`) per evaluation
    coverage   distinct (task, cell) combinations holding at least one pair

Writes results/stepB/ab.json (everything) and ab.md (the table).

Run from the repository root, after scripts/run_ab_search.sh:
    .venv/bin/python -m scripts.ab_compare --v1 runs_ab/v1/v1 --v2 runs_ab/v2/v2
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ava.loop import RunPaths, candidate_key
from ava.report import load_scores, rank_score
from ava.search.archive import Archive, Clusterer
from ava.search.embed import CachingEmbedder
from ava.search.evolve import CLUSTERS_FILENAME, STATE_FILENAME
from ava.spec import CandidateSpec


def rows_of(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in RunPaths(root).all_scores():
        rows.extend(load_scores(path))
    return rows


def spec_of(row: dict[str, Any]) -> CandidateSpec:
    return CandidateSpec(
        task=str(row["task"]),
        prompts=tuple(str(p) for p in row["prompts"]),
        style=str(row["style"]),
        seed=int(row["seed"]),
        ref_image=row.get("ref_image"),
    )


def summarise(rows: Sequence[dict[str, Any]], archive: Archive) -> dict[str, Any]:
    """Yield, coverage and the per-task breakdown of one arm."""
    held = [r for r in rows if r.get("diagnosis") == "ok"]
    pairs = {candidate_key(r) for r in rows}
    held_pairs = {candidate_key(r) for r in held}
    cells_all = {(r["task"], *archive.cell_for(spec_of(r))) for r in rows}
    cells_held = {(r["task"], *archive.cell_for(spec_of(r))) for r in held}
    seps = sorted((rank_score(r) for r in held), reverse=True)

    by_task: dict[str, dict[str, Any]] = {}
    for task in sorted({str(r["task"]) for r in rows}):
        t_rows = [r for r in rows if r["task"] == task]
        t_held = [r for r in t_rows if r.get("diagnosis") == "ok"]
        by_task[task] = {
            "evaluations": len(t_rows),
            "held": len(t_held),
            "yield": len(t_held) / len(t_rows) if t_rows else 0.0,
            "cells_held": len({c for c in cells_held if c[0] == task}),
            "cells_filled": len({c for c in cells_all if c[0] == task}),
        }
    return {
        "evaluations": len(rows),
        "held": len(held),
        "yield": len(held) / len(rows) if rows else 0.0,
        "distinct_pairs": len(pairs),
        "distinct_pairs_held": len(held_pairs),
        "cells_filled": len(cells_all),
        "cells_held": len(cells_held),
        "best_sep_min": seps[0] if seps else None,
        "top8_mean_sep_min": float(np.mean(seps[:8])) if seps else None,
        "origins": dict(Counter(str(r["origin"]) for r in rows)),
        "by_task": by_task,
    }


def v2_extras(root: Path) -> dict[str, Any]:
    """What only the evolve run can report: operators, surrogate skill, dedup."""
    path = root / STATE_FILENAME
    if not path.exists():
        return {}
    state = json.loads(path.read_text(encoding="utf-8"))
    return {
        "operators": state.get("operators", {}),
        "surrogate_log": state.get("surrogate_log", []),
        "dedup": state.get("rounds", [{}])[-1].get("dedup", {}),
        "rounds": state.get("rounds", []),
    }


def decide(v1: dict[str, Any], v2: dict[str, Any]) -> dict[str, Any]:
    """The adoption rule from design_search_v2.md Sec. 6, applied literally."""
    yield_ok = v2["yield"] >= v1["yield"]
    coverage_ok = v2["cells_held"] > v1["cells_held"]
    return {
        "yield_v2_at_least_v1": yield_ok,
        "coverage_v2_exceeds_v1": coverage_ok,
        "adopt_v2": yield_ok and coverage_ok,
    }


def write_markdown(result: dict[str, Any], out: Path) -> None:
    v1, v2, d = result["v1"], result["v2"], result["decision"]
    lines = [
        "# A/B: component bandit (v1) vs evolutionary search (v2)",
        "",
        f"Same budget ({v1['evaluations']} vs {v2['evaluations']} evaluations), "
        "same tasks, same seeds, private copies of the same vocabulary.",
        "Coverage is measured on the v2 run's cluster map for both arms.",
        "",
        "| metric | v1 bandit | v2 evolve |",
        "|---|---|---|",
        f"| evaluations | {v1['evaluations']} | {v2['evaluations']} |",
        f"| held (`ok`) | {v1['held']} | {v2['held']} |",
        f"| **yield** | {v1['yield']:.3f} | {v2['yield']:.3f} |",
        f"| distinct pairs | {v1['distinct_pairs']} | {v2['distinct_pairs']} |",
        f"| distinct pairs held | {v1['distinct_pairs_held']} "
        f"| {v2['distinct_pairs_held']} |",
        f"| cells filled | {v1['cells_filled']} | {v2['cells_filled']} |",
        f"| **cells held** | {v1['cells_held']} | {v2['cells_held']} |",
        f"| best sep_min | {_fmt(v1['best_sep_min'])} | {_fmt(v2['best_sep_min'])} |",
        f"| top-8 mean sep_min | {_fmt(v1['top8_mean_sep_min'])} "
        f"| {_fmt(v2['top8_mean_sep_min'])} |",
        "",
        "## Per task",
        "",
        "| task | v1 yield | v2 yield | v1 cells held | v2 cells held |",
        "|---|---|---|---|---|",
    ]
    for task in sorted(set(v1["by_task"]) | set(v2["by_task"])):
        a = v1["by_task"].get(task, {})
        b = v2["by_task"].get(task, {})
        lines.append(
            f"| {task} | {a.get('yield', 0.0):.3f} ({a.get('held', 0)}/"
            f"{a.get('evaluations', 0)}) | {b.get('yield', 0.0):.3f} "
            f"({b.get('held', 0)}/{b.get('evaluations', 0)}) "
            f"| {a.get('cells_held', 0)} | {b.get('cells_held', 0)} |"
        )
    lines += [
        "",
        "## Where v2's candidates came from",
        "",
        "| origin | count |",
        "|---|---|",
    ]
    for origin, n in sorted(v2["origins"].items()):
        lines.append(f"| {origin} | {n} |")
    extras = result.get("v2_extras", {})
    if extras.get("operators"):
        lines += [
            "",
            "## Operator bandit (v2)",
            "",
            "| operator | n | mean reward |",
            "|---|---|---|",
        ]
        ops = extras["operators"]
        for name in sorted(ops.get("counts", {})):
            mean = ops["means"].get(name, 0.0)
            lines.append(f"| {name} | {ops['counts'][name]} | {mean:.3f} |")
    if extras.get("surrogate_log"):
        lines += [
            "",
            "## Surrogate skill per round (v2)",
            "",
            "| round | n | LOO rho | mode |",
            "|---|---|---|---|",
        ]
        for e in extras["surrogate_log"]:
            rho = "—" if e.get("rho") is None else f"{e['rho']:+.3f}"
            lines.append(f"| {e['round']} | {e['n_train']} | {rho} | {e['mode']} |")
    if extras.get("dedup"):
        dd = extras["dedup"]
        lines += [
            "",
            "## Duplicate rejection (v2)",
            "",
            f"eta = {dd.get('eta')}; checked {dd.get('n_checked')}, rejected "
            f"{dd.get('n_rejected')}; max-cosine quantiles "
            f"{json.dumps(dd.get('max_similarity_quantiles', {}))}",
        ]
    lines += [
        "",
        "## Decision (design_search_v2.md Sec. 6)",
        "",
        f"- yield v2 >= v1: **{d['yield_v2_at_least_v1']}**",
        f"- cells held v2 > v1: **{d['coverage_v2_exceeds_v1']}**",
        f"- adopt v2 as default: **{d['adopt_v2']}**",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def _fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:+.3f}"


def compare(v1_root: Path, v2_root: Path, embed: CachingEmbedder) -> dict[str, Any]:
    clusterer = Clusterer.load(v2_root / CLUSTERS_FILENAME)
    archive = Archive(clusterer, embed)
    v1_rows, v2_rows = rows_of(v1_root), rows_of(v2_root)
    v1, v2 = summarise(v1_rows, archive), summarise(v2_rows, archive)
    return {
        "v1_run": str(v1_root),
        "v2_run": str(v2_root),
        "clusters": {"k": clusterer.k, "n_words": len(clusterer.words)},
        "v1": v1,
        "v2": v2,
        "v2_extras": v2_extras(v2_root),
        "decision": decide(v1, v2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, required=True, help="bandit run directory")
    parser.add_argument("--v2", type=Path, required=True, help="evolve run directory")
    parser.add_argument("--out", type=Path, default=Path("results/stepB"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from ava.image.judge import ClipBlipJudge

    judge = ClipBlipJudge(device=args.device, use_blip=False)

    def embed(prompts: Sequence[str]) -> np.ndarray:
        return judge.text_embeddings(list(prompts)).cpu().numpy()

    result = compare(args.v1, args.v2, CachingEmbedder(embed))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ab.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown(result, args.out / "ab.md")
    print((args.out / "ab.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
