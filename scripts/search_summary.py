"""What an evolve run learned, read from its persisted files alone.

Reads runs/<run>/archive.jsonl, search_state.json and clusters.npz and writes
search_summary.md beside them. Nothing is regenerated, re-judged or refitted,
and no model is loaded: the cluster ids are already on every individual.

Sections:

  coverage    per task, how many cells hold an elite that held at least once
  elites      the archive, best first: cell, prompts, style, fitness, seeds
  racing      how many pairs reached each seed count
  surrogate   the leave-one-out rho per round, and whether it ever drove selection
  operators   pulls and mean improvement reward per operator
  dedup       how often near-duplicates were rejected, and at what cosines

Run from the repository root:
    .venv/bin/python -m scripts.search_summary --run runs/flip_evo_20260902
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ava.search.evolve import ARCHIVE_FILENAME, CLUSTERS_FILENAME, STATE_FILENAME


def load_archive(root: Path) -> list[dict[str, Any]]:
    path = root / ARCHIVE_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; is this an evolve run?")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def cluster_examples(root: Path, per_cluster: int = 4) -> dict[int, list[str]]:
    """A few vocabulary words per cluster, so a cell id means something to a reader."""
    path = root / CLUSTERS_FILENAME
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=True) as z:
        words = [str(w) for w in z["words"]]
        labels = [int(x) for x in z["labels"]]
    out: dict[int, list[str]] = {}
    for word, label in zip(words, labels, strict=True):
        if len(out.setdefault(label, [])) < per_cluster:
            out[label].append(word)
    return out


def summarise(root: Path, top: int) -> dict[str, Any]:
    individuals = load_archive(root)
    state_path = root / STATE_FILENAME
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {}
    )
    elites = sorted(
        (i for i in individuals if i["elite"]),
        key=lambda i: (-(i["n_ok"] > 0), -i["fitness"], -i["n_seeds"]),
    )
    coverage: dict[str, dict[str, int]] = {}
    for i in elites:
        entry = coverage.setdefault(i["spec"]["task"], {"filled": 0, "held": 0})
        entry["filled"] += 1
        entry["held"] += int(i["n_ok"] > 0)
    return {
        "run": str(root),
        "pairs": len(individuals),
        "evaluations": sum(i["n_seeds"] for i in individuals),
        "coverage": coverage,
        "elites": elites[:top],
        "seed_counts": dict(Counter(i["n_seeds"] for i in individuals)),
        "held_by_seeds": {
            n: sum(1 for i in individuals if i["n_seeds"] == n and i["n_ok"] == n)
            for n in sorted({i["n_seeds"] for i in individuals})
        },
        "operators": state.get("operators", {}),
        "surrogate_log": state.get("surrogate_log", []),
        "dedup": (state.get("rounds") or [{}])[-1].get("dedup", {}),
        "origins": dict(
            Counter(e["origin"] for i in individuals for e in i["evaluations"])
        ),
        "clusters": cluster_examples(root),
    }


def to_markdown(s: dict[str, Any]) -> str:
    lines = [
        f"# Search summary: `{s['run']}`",
        "",
        f"{s['pairs']} pairs, {s['evaluations']} evaluations "
        f"({', '.join(f'{k}: {v}' for k, v in sorted(s['origins'].items()))}).",
        "",
        "## Coverage",
        "",
        "| task | cells filled | cells held |",
        "|---|---|---|",
    ]
    for task, c in sorted(s["coverage"].items()):
        lines.append(f"| {task} | {c['filled']} | {c['held']} |")
    if s["clusters"]:
        lines += ["", "Cluster ids, with a few of the words that define them:", ""]
        for cid, words in sorted(s["clusters"].items()):
            lines.append(f"- `{cid}`: {', '.join(words)}")
    lines += [
        "",
        "## Elites, best first",
        "",
        "`fitness` is the mean `sep_min` over the seeds the pair was generated "
        "with; `held` counts the seeds on which every view read as its prompt.",
        "",
        "| cell | prompts (slot order) | style | fitness | held / seeds | made by |",
        "|---|---|---|---|---|---|",
    ]
    for i in s["elites"]:
        spec = i["spec"]
        cell = ",".join(str(c) for c in i["cell"])
        made = i["operator"] or i["evaluations"][0]["origin"]
        lines.append(
            f"| {spec['task']}:{cell} | {' / '.join(spec['prompts'])} "
            f"| {spec['style'] or '—'} | {i['fitness']:+.3f} "
            f"| {i['n_ok']} / {i['n_seeds']} | {made} |"
        )
    lines += [
        "",
        "## Racing",
        "",
        "| seeds | pairs with this many | of which held on every seed |",
        "|---|---|---|",
    ]
    for n in sorted(s["seed_counts"]):
        lines.append(
            f"| {n} | {s['seed_counts'][n]} | {s['held_by_seeds'].get(n, 0)} |"
        )
    if s["surrogate_log"]:
        lines += [
            "",
            "## Surrogate",
            "",
            "Leave-one-out Spearman rho of the GP on everything evaluated so far. "
            "`ucb` means it drove selection that round; `uniform` means it did not.",
            "",
            "| round | n | rho | mode |",
            "|---|---|---|---|",
        ]
        for e in s["surrogate_log"]:
            rho = "—" if e.get("rho") is None else f"{e['rho']:+.3f}"
            lines.append(f"| {e['round']} | {e['n_train']} | {rho} | {e['mode']} |")
    ops = s["operators"]
    if ops.get("counts"):
        lines += [
            "",
            "## Operators",
            "",
            "Mean reward is the squashed improvement of the child over its parent, "
            "in [0, 1); zero means no child was ever better.",
            "",
            "| operator | pulls | mean reward |",
            "|---|---|---|",
        ]
        for name in sorted(ops["counts"]):
            mean = ops["means"].get(name, 0.0)
            lines.append(f"| {name} | {ops['counts'][name]} | {mean:.3f} |")
    if s["dedup"]:
        d = s["dedup"]
        lines += [
            "",
            "## Duplicate rejection",
            "",
            f"eta = {d.get('eta')}: {d.get('n_rejected')} of {d.get('n_checked')} "
            f"children rejected. Max-cosine quantiles over the checked children: "
            f"{json.dumps(d.get('max_similarity_quantiles', {}))}",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="runs/<run> directory")
    parser.add_argument("--top", type=int, default=24, help="elites to list")
    args = parser.parse_args()
    text = to_markdown(summarise(args.run, args.top))
    out = args.run / "search_summary.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
