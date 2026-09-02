"""What the web UI shows, read from a run's persisted files alone.

Every function here is a pure function of `runs/<run>/`: `config.yaml`,
`state.json`, `round_*/candidates.jsonl`, `round_*/scores.jsonl`, and for an
evolve run `archive.jsonl`, `search_state.json` and `clusters.npz`. Nothing is
regenerated, re-judged or refitted, no model is loaded, and torch is never
imported. The UI is a figure in the sense of `.claude/rules/coding.md`: it
reads artifacts, it does not compute them.

The functions return plain dicts and lists so they serialise to JSON as they
are; the Flask layer in `ava.webui.api` adds nothing but routing.

Media paths are returned relative to the run root (`round_003/<uid>/view_flip.png`)
rather than as the working-directory-relative strings the loop wrote, so a run
can be served from any location.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ARCHIVE_FILENAME = "archive.jsonl"
STATE_FILENAME = "search_state.json"
CLUSTERS_FILENAME = "clusters.npz"
ROUND_GLOB = "round_*"
WORDS_PER_CLUSTER = 4

# Files whose modification time tells a polling client that a run has moved on.
_VERSION_FILES = ("state.json", ARCHIVE_FILENAME, STATE_FILENAME)


# ---------------------------------------------------------------------------
# Low-level readers
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every complete line of a JSON-lines file; a partial last line is skipped.

    A run that is still writing may leave a half-written last line. That line
    is not an error to a viewer, it is a candidate that has not landed yet.
    """
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def read_config(root: Path) -> dict[str, Any]:
    path = root / "config.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def round_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.glob(ROUND_GLOB) if p.is_dir())


def round_index(round_dir: Path) -> int:
    return int(round_dir.name.split("_", 1)[1])


def is_run_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.yaml").exists()


def run_version(root: Path) -> float:
    """The newest modification time among the files a viewer depends on."""
    stamps = [0.0]
    for name in _VERSION_FILES:
        p = root / name
        if p.exists():
            stamps.append(p.stat().st_mtime)
    for rd in round_dirs(root):
        for name in ("candidates.jsonl", "scores.jsonl"):
            p = rd / name
            if p.exists():
                stamps.append(p.stat().st_mtime)
    return max(stamps)


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


def _media(row: dict[str, Any], round_dir: str) -> list[dict[str, str]]:
    """One entry per view the judge looked at, as a run-relative path.

    The loop recorded working-directory-relative paths. Only the file name is
    taken from them; the directory is `round_XXX/<uid>/` by the run layout, so a
    run copied elsewhere still resolves.
    """
    uid = str(row["uid"])
    media: list[dict[str, str]] = []
    wavs = row.get("wav_paths")
    if isinstance(wavs, dict):
        for slot, p in wavs.items():
            media.append(
                {
                    "slot": str(slot),
                    "kind": "audio",
                    "path": f"{round_dir}/{uid}/{Path(str(p)).name}",
                }
            )
        return media
    views = row.get("view_paths") or []
    slots = row.get("slots") or []
    for slot, p in zip(slots, views, strict=False):
        media.append(
            {
                "slot": str(slot),
                "kind": "image",
                "path": f"{round_dir}/{uid}/{Path(str(p)).name}",
            }
        )
    return media


def _sample_path(row: dict[str, Any], round_dir: str) -> str | None:
    p = row.get("image_path")
    if not p:
        return None
    return f"{round_dir}/{row['uid']}/{Path(str(p)).name}"


def candidate_view(row: dict[str, Any], round_dir: str) -> dict[str, Any]:
    """A scores.jsonl row with media resolved and the noise dropped."""
    extra = row.get("extra") or {}
    out = {
        "uid": row["uid"],
        "round": int(row.get("round", round_index(Path(round_dir)))),
        "task": row["task"],
        "slots": list(row.get("slots", [])),
        "prompts": list(row.get("prompts", [])),
        "style": row.get("style", ""),
        "seed": row.get("seed"),
        "origin": row.get("origin", ""),
        "operator": extra.get("operator"),
        "parents": list(extra.get("parents") or []),
        "selection": extra.get("selection"),
        "surrogate": extra.get("surrogate") or {},
        "ok": bool(row.get("diagnosis") == "ok"),
        "diagnosis": row.get("diagnosis", ""),
        "holds": list(row.get("holds", [])),
        "j": row.get("j"),
        "sep": list(row.get("sep", [])),
        "sep_min": row.get("sep_min", min(row["sep"]) if row.get("sep") else None),
        "alignment": row.get("alignment"),
        "concealment": row.get("concealment"),
        "scores": row.get("scores"),
        "captions": list(row.get("captions", [])),
        "media": _media(row, round_dir),
        "sample": _sample_path(row, round_dir),
        "pair": pair_id(row["task"], row.get("prompts", []), row.get("style", "")),
    }
    if "seconds" in row:
        out["seconds"] = row["seconds"]
    return out


def candidates(root: Path) -> list[dict[str, Any]]:
    """Every scored candidate of the run, in round then file order."""
    out: list[dict[str, Any]] = []
    for rd in round_dirs(root):
        for row in read_jsonl(rd / "scores.jsonl"):
            out.append(candidate_view(row, rd.name))
    return out


# ---------------------------------------------------------------------------
# Rounds
# ---------------------------------------------------------------------------


def _round_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    seps = [float(r["sep_min"]) for r in rows if r.get("sep_min") is not None]
    return {
        "n": len(rows),
        "n_ok": sum(1 for r in rows if r.get("diagnosis") == "ok"),
        "best": max(seps) if seps else None,
        "mean": statistics.fmean(seps) if seps else None,
        "median": statistics.median(seps) if seps else None,
        "origins": dict(Counter(str(r.get("origin", "")) for r in rows)),
        "operators": dict(
            Counter(
                str((r.get("extra") or {}).get("operator"))
                for r in rows
                if (r.get("extra") or {}).get("operator")
            )
        ),
        "tasks": dict(Counter(str(r.get("task", "")) for r in rows)),
    }


def rounds(root: Path) -> list[dict[str, Any]]:
    """Per-round statistics, with the search layer's own log merged in.

    `search_state.json` keeps one entry per completed round under `rounds`
    (coverage, dedup, operator rewards) and one per round under
    `surrogate_log` (rho and whether the GP drove selection). They are matched
    to the round directory by position and by the `round` field respectively.
    """
    search = read_json(root / STATE_FILENAME)
    search_rounds = search.get("rounds") or []
    surrogate = {
        int(e["round"]): e for e in search.get("surrogate_log") or [] if "round" in e
    }
    out: list[dict[str, Any]] = []
    for rd in round_dirs(root):
        idx = round_index(rd)
        rows = read_jsonl(rd / "scores.jsonl")
        planned = len(read_jsonl(rd / "candidates.jsonl"))
        entry = {"index": idx, "planned": planned, **_round_stats(rows)}
        entry["search"] = search_rounds[idx] if idx < len(search_rounds) else None
        entry["surrogate"] = surrogate.get(idx)
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Lineage and archive (evolve runs)
# ---------------------------------------------------------------------------


def pair_id(task: str, prompts: list[str] | tuple[str, ...], style: str) -> str:
    """A short stable id for a pair, the unit the archive stores.

    The archive's own key is a JSON string; this is its sha1 prefix, which is
    friendlier in a URL and in an SVG node id. It is derived, never stored.
    """
    key = json.dumps([task, list(prompts), style])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _pair_id_of_key(key: str) -> str:
    task, prompts, style = json.loads(key)
    return pair_id(task, prompts, style)


def read_archive(root: Path) -> list[dict[str, Any]]:
    return read_jsonl(root / ARCHIVE_FILENAME)


def lineage(root: Path) -> dict[str, Any]:
    """Nodes are pairs, edges are `parents -> child` labelled by operator.

    Bootstrap pairs have no parents and are the roots. A parent key that is
    not itself in the archive (it should not happen; the archive keeps
    everything) still yields an edge, to a node flagged `missing`.
    """
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for row in read_archive(root):
        spec = row["spec"]
        nid = _pair_id_of_key(row["key"])
        nodes[nid] = {
            "id": nid,
            "task": spec["task"],
            "prompts": list(spec["prompts"]),
            "style": spec.get("style", ""),
            "cell": list(row.get("cell", [])),
            "first_round": row.get("first_round"),
            "operator": row.get("operator"),
            "fitness": row.get("fitness"),
            "n_seeds": row.get("n_seeds", len(row.get("evaluations", []))),
            "n_ok": row.get("n_ok"),
            "n_children": row.get("n_children", 0),
            "elite": bool(row.get("elite")),
            "evaluations": list(row.get("evaluations", [])),
            "parents": [_pair_id_of_key(k) for k in row.get("parents", [])],
            "missing": False,
        }
    for node in list(nodes.values()):
        for pid in node["parents"]:
            if pid not in nodes:
                nodes[pid] = {
                    "id": pid,
                    "task": node["task"],
                    "prompts": [],
                    "style": "",
                    "cell": [],
                    "first_round": None,
                    "operator": None,
                    "fitness": None,
                    "n_seeds": 0,
                    "n_ok": 0,
                    "n_children": 0,
                    "elite": False,
                    "evaluations": [],
                    "parents": [],
                    "missing": True,
                }
            edges.append(
                {"source": pid, "target": node["id"], "operator": node["operator"]}
            )
    ordered = sorted(
        nodes.values(),
        key=lambda n: (
            n["first_round"] if n["first_round"] is not None else -1,
            n["id"],
        ),
    )
    return {"nodes": ordered, "edges": edges}


def clusters(root: Path) -> dict[str, Any]:
    """The k-means map the archive's cells refer to: k and a few words per id."""
    path = root / CLUSTERS_FILENAME
    if not path.exists():
        return {"k": 0, "seed": None, "words": {}}
    with np.load(path, allow_pickle=True) as z:
        words = [str(w) for w in z["words"]]
        labels = [int(x) for x in z["labels"]]
        k = int(z["centroids"].shape[0])
        seed = int(z["seed"])
    by_label: dict[int, list[str]] = {i: [] for i in range(k)}
    for w, label in zip(words, labels, strict=True):
        by_label.setdefault(label, []).append(w)
    return {
        "k": k,
        "seed": seed,
        "words": {
            str(i): sorted(ws)[:WORDS_PER_CLUSTER] for i, ws in sorted(by_label.items())
        },
        "sizes": {str(i): len(ws) for i, ws in sorted(by_label.items())},
    }


def archive_grid(root: Path) -> dict[str, Any]:
    """The MAP-Elites archive per task: the elite of every filled cell.

    A cell is `(cluster per searched slot)`; with two slots it is a grid, with
    more it is a list. `dims` tells the client which. Empty cells are the
    client's to draw: it knows `k`.
    """
    cl = clusters(root)
    per_task: dict[str, dict[str, Any]] = {}
    for row in read_archive(root):
        task = row["spec"]["task"]
        cell = [int(c) for c in row.get("cell", [])]
        entry = per_task.setdefault(
            task, {"task": task, "dims": len(cell), "cells": {}}
        )
        key = ",".join(str(c) for c in cell)
        cellinfo = entry["cells"].setdefault(
            key, {"cell": cell, "n_pairs": 0, "n_evaluations": 0, "elite": None}
        )
        cellinfo["n_pairs"] += 1
        cellinfo["n_evaluations"] += len(row.get("evaluations", []))
        if row.get("elite"):
            cellinfo["elite"] = {
                "id": _pair_id_of_key(row["key"]),
                "prompts": list(row["spec"]["prompts"]),
                "style": row["spec"].get("style", ""),
                "fitness": row.get("fitness"),
                "n_ok": row.get("n_ok"),
                "n_seeds": row.get("n_seeds"),
            }
    return {"k": cl["k"], "clusters": cl, "tasks": list(per_task.values())}


# ---------------------------------------------------------------------------
# Run summaries
# ---------------------------------------------------------------------------


def _track(rows: list[dict[str, Any]]) -> str:
    """`audio` if any candidate carries wav files, else `image`."""
    for r in rows:
        if r.get("wav_paths"):
            return "audio"
        if r.get("view_paths"):
            return "image"
    return "image"


def run_summary(root: Path) -> dict[str, Any]:
    """Everything the dashboard card and the poller need, in one read."""
    config = read_config(root)
    state = read_json(root / "state.json")
    dirs = round_dirs(root)
    rows: list[dict[str, Any]] = []
    for rd in dirs:
        rows.extend(read_jsonl(rd / "scores.jsonl"))
    seps = [(float(r["sep_min"]), r) for r in rows if r.get("sep_min") is not None]
    best = max(seps, key=lambda t: t[0])[1] if seps else None
    next_round = int(state.get("round_index", 0))
    planned = int(config.get("rounds", 0))
    current = root / f"round_{next_round:03d}"
    in_progress = None
    if next_round < planned and current.is_dir():
        in_progress = {
            "index": next_round,
            "planned": len(read_jsonl(current / "candidates.jsonl")),
            "scored": len(read_jsonl(current / "scores.jsonl")),
        }
    return {
        "run_id": str(config.get("run_id", root.name)),
        "name": root.name,
        "track": _track(rows),
        "proposer": str(config.get("proposer", "bandit")),
        "tasks": list(config.get("tasks", [])),
        "rounds_planned": planned,
        "rounds_done": len([rd for rd in dirs if (rd / "scores.jsonl").exists()]),
        "next_round": next_round,
        "finished": next_round >= planned and planned > 0,
        "in_progress": in_progress,
        "n_candidates": len(rows),
        "n_held": sum(1 for r in rows if r.get("diagnosis") == "ok"),
        "best": candidate_view(best, f"round_{int(best['round']):03d}")
        if best
        else None,
        "has_archive": (root / ARCHIVE_FILENAME).exists(),
        "config": config,
        "version": run_version(root),
    }


def list_runs(runs_dir: Path) -> list[dict[str, Any]]:
    """Every run directory under `runs_dir`, newest version first."""
    if not runs_dir.is_dir():
        return []
    out = [run_summary(p) for p in sorted(runs_dir.iterdir()) if is_run_dir(p)]
    out.sort(key=lambda s: s["version"], reverse=True)
    return out
