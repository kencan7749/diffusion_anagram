"""The web UI's data layer reads persisted runs and nothing else.

An evolve run is produced with the same fakes `test_search_evolve` uses, so
the archive, search state and clusters are the real thing. A bandit-style run
without an archive and an audio run are written by hand: the layer has to
cope with both, and with a round that is still being scored.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ava.webui import data
from tests.test_search_evolve import run as run_evolve

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def evolve_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("webui")
    _, paths, _, _ = run_evolve(tmp, "evo", rounds=3, k=8)
    return paths.root


def _write_manual_run(root: Path, *, audio: bool, partial: bool) -> None:
    root.mkdir(parents=True)
    (root / "config.yaml").write_text(
        "run_id: manual\nproposer: bandit\nrounds: 2\ntasks:\n- flip\n"
    )
    (root / "state.json").write_text(
        json.dumps({"run_id": "manual", "round_index": 1, "evaluated_uids": ["u1"]})
    )
    r0 = root / "round_000" / "u1"
    r0.mkdir(parents=True)
    row = {
        "uid": "u1",
        "task": "flip",
        "slots": ["identity", "flip"],
        "prompts": ["a duck", "a rabbit"],
        "style": "a photo of",
        "seed": 0,
        "round": 0,
        "origin": "bootstrap",
        "extra": {"operator": None, "parents": []},
        "sep": [0.02, 0.05],
        "sep_min": 0.02,
        "j": 0.9,
        "holds": [True, True],
        "diagnosis": "ok",
        "scores": [[0.3, 0.2], [0.1, 0.3]],
        "captions": ["a duck", "a rabbit"],
    }
    if audio:
        row["wav_paths"] = {
            "forward": "elsewhere/runs/manual/round_000/u1/forward.wav",
            "reverse": "elsewhere/runs/manual/round_000/u1/reverse.wav",
        }
        (r0 / "forward.wav").write_bytes(b"RIFF")
    else:
        row["image_path"] = "runs/manual/round_000/u1/sample_256.png"
        row["view_paths"] = [
            "runs/manual/round_000/u1/view_identity.png",
            "runs/manual/round_000/u1/view_flip.png",
        ]
    (root / "round_000" / "candidates.jsonl").write_text(
        json.dumps({"uid": "u1", "spec": {}}) + "\n"
    )
    (root / "round_000" / "scores.jsonl").write_text(json.dumps(row) + "\n")
    if partial:
        r1 = root / "round_001"
        r1.mkdir()
        (r1 / "candidates.jsonl").write_text(
            "\n".join(json.dumps({"uid": f"c{i}", "spec": {}}) for i in range(4)) + "\n"
        )
        # One landed, one is half written: the viewer must not choke.
        (r1 / "scores.jsonl").write_text(
            json.dumps({**row, "uid": "c0", "round": 1}) + '\n{"uid": "c1", "tas'
        )


# -- evolve run -------------------------------------------------------------


def test_summary_describes_a_finished_evolve_run(evolve_root: Path) -> None:
    s = data.run_summary(evolve_root)
    assert s["proposer"] == "evolve"
    assert s["has_archive"] is True
    assert s["track"] == "image"
    assert s["rounds_done"] == 3 and s["rounds_planned"] == 3
    assert s["finished"] is True and s["in_progress"] is None
    assert s["n_candidates"] == 24
    assert s["best"] is not None and s["best"]["sep_min"] is not None
    assert s["version"] > 0


def test_candidates_resolve_media_relative_to_the_run(evolve_root: Path) -> None:
    rows = data.candidates(evolve_root)
    assert len(rows) == 24
    for row in rows:
        assert row["media"], row["uid"]
        for m in row["media"]:
            assert m["kind"] == "image"
            assert m["path"].startswith(f"round_{row['round']:03d}/{row['uid']}/")
            assert (evolve_root / m["path"]).exists(), m["path"]
        assert row["pair"] == data.pair_id(row["task"], row["prompts"], row["style"])


def test_rounds_merge_the_search_log(evolve_root: Path) -> None:
    rs = data.rounds(evolve_root)
    assert [r["index"] for r in rs] == [0, 1, 2]
    for r in rs:
        assert r["planned"] == r["n"] == 8
        assert r["n_ok"] <= r["n"]
        assert r["search"] is not None and "coverage" in r["search"]
        assert r["surrogate"] is not None and r["surrogate"]["round"] == r["index"]


def test_lineage_links_children_to_archived_parents(evolve_root: Path) -> None:
    g = data.lineage(evolve_root)
    ids = {n["id"] for n in g["nodes"]}
    assert len(ids) == len(g["nodes"])
    assert not any(n["missing"] for n in g["nodes"])
    for e in g["edges"]:
        assert e["source"] in ids and e["target"] in ids
        assert e["operator"]
    roots = [n for n in g["nodes"] if not n["parents"]]
    assert roots and all(n["operator"] is None for n in roots)
    # Every candidate's pair is a node: the archive keeps everything.
    pairs = {c["pair"] for c in data.candidates(evolve_root)}
    assert pairs <= ids


def test_archive_grid_has_one_elite_per_filled_cell(evolve_root: Path) -> None:
    grid = data.archive_grid(evolve_root)
    assert grid["k"] == 4
    assert set(grid["clusters"]["words"]) == {"0", "1", "2", "3"}
    assert grid["tasks"], "no task in the archive"
    for t in grid["tasks"]:
        assert t["dims"] == 2
        for cell in t["cells"].values():
            assert cell["n_pairs"] >= 1
            assert cell["elite"] is not None
            assert cell["elite"]["id"]


# -- manual runs ------------------------------------------------------------


def test_run_without_an_archive_still_summarises(tmp_path: Path) -> None:
    root = tmp_path / "manual"
    _write_manual_run(root, audio=False, partial=False)
    s = data.run_summary(root)
    assert s["has_archive"] is False and s["proposer"] == "bandit"
    assert s["rounds_done"] == 1 and s["rounds_planned"] == 2
    assert s["finished"] is False
    assert data.lineage(root) == {"nodes": [], "edges": []}
    assert data.archive_grid(root)["tasks"] == []
    row = data.candidates(root)[0]
    assert [m["path"] for m in row["media"]] == [
        "round_000/u1/view_identity.png",
        "round_000/u1/view_flip.png",
    ]
    assert row["sample"] == "round_000/u1/sample_256.png"


def test_audio_run_exposes_wavs_as_media(tmp_path: Path) -> None:
    root = tmp_path / "manual"
    _write_manual_run(root, audio=True, partial=False)
    assert data.run_summary(root)["track"] == "audio"
    row = data.candidates(root)[0]
    assert row["sample"] is None
    assert row["media"] == [
        {"slot": "forward", "kind": "audio", "path": "round_000/u1/forward.wav"},
        {"slot": "reverse", "kind": "audio", "path": "round_000/u1/reverse.wav"},
    ]


def test_a_round_still_being_scored_is_reported_as_progress(tmp_path: Path) -> None:
    root = tmp_path / "manual"
    _write_manual_run(root, audio=False, partial=True)
    s = data.run_summary(root)
    assert s["in_progress"] == {"index": 1, "planned": 4, "scored": 1}
    assert s["n_candidates"] == 2
    rs = data.rounds(root)
    assert rs[1]["planned"] == 4 and rs[1]["n"] == 1


def test_list_runs_skips_directories_without_a_config(tmp_path: Path) -> None:
    _write_manual_run(tmp_path / "runs" / "a", audio=False, partial=False)
    (tmp_path / "runs" / "not_a_run").mkdir()
    (tmp_path / "runs" / "vocab.db").write_bytes(b"")
    names = [s["name"] for s in data.list_runs(tmp_path / "runs")]
    assert names == ["a"]
    assert data.list_runs(tmp_path / "nowhere") == []
