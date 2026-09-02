"""The clip queue renders one job at a time and the API drives it.

The renderer is a fake that writes a file where a clip would go, so neither
torch nor ffmpeg is involved; a blocking fake holds a job in `running` to
test the duplicate rule.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
from flask import Flask

from ava.webui import data
from ava.webui.app import create_app
from ava.webui.jobs import AnimationQueue
from tests.test_search_evolve import run as run_evolve

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def fake_render(root: Path, uid: str, cache: dict[str, Any]) -> list[str]:
    row = data.find_row(root, uid)
    assert row is not None
    cdir = root / f"round_{int(row['round']):03d}" / uid
    (cdir / "anim_flip.mp4").write_bytes(b"\x00")
    cache.setdefault("calls", []).append(uid)
    return ["anim_flip.mp4"]


def failing_render(root: Path, uid: str, cache: dict[str, Any]) -> list[str]:
    raise RuntimeError("no make_frame")


@pytest.fixture(scope="module")
def runs_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("webui_jobs")
    run_evolve(tmp, "evo", rounds=1, k=4)
    return tmp / "evo"


# -- the queue ---------------------------------------------------------------


def test_jobs_run_in_order_and_are_reported(runs_dir: Path) -> None:
    root = runs_dir / "evo"
    uids = [c["uid"] for c in data.candidates(root)][:2]
    q = AnimationQueue(render=fake_render)
    assert q.submit(root, uids[0]) and q.submit(root, uids[1])
    assert q.wait(10)
    s = q.status(root)
    assert s["pending"] == [] and s["running"] is None and s["failed"] == {}
    assert set(s["done"]) == set(uids)
    assert q._cache["calls"] == uids
    assert q.status(Path("/elsewhere"))["done"] == {}


def test_a_failure_is_recorded_and_the_worker_survives(runs_dir: Path) -> None:
    root = runs_dir / "evo"
    uid = data.candidates(root)[0]["uid"]
    q = AnimationQueue(render=failing_render)
    q.submit(root, uid)
    assert q.wait(10)
    assert q.status(root)["failed"] == {uid: "RuntimeError: no make_frame"}
    # Resubmitting clears the old verdict and runs again.
    q.submit(root, uid)
    assert q.wait(10)
    assert uid in q.status(root)["failed"]


def test_duplicates_are_refused_while_queued_or_running(runs_dir: Path) -> None:
    root = runs_dir / "evo"
    uid = data.candidates(root)[0]["uid"]
    gate = threading.Event()

    def blocking(root: Path, uid: str, cache: dict[str, Any]) -> list[str]:
        gate.wait(10)
        return []

    q = AnimationQueue(render=blocking)
    assert q.submit(root, uid)
    for _ in range(100):  # until the worker has picked it up
        if q.status(root)["running"] == uid:
            break
        threading.Event().wait(0.01)
    assert q.status(root)["running"] == uid
    assert not q.submit(root, uid)
    gate.set()
    assert q.wait(10)
    assert q.submit(root, uid)


# -- the API -----------------------------------------------------------------


@pytest.fixture()
def app(runs_dir: Path) -> Flask:
    app = create_app(runs_dir, animations=AnimationQueue(render=fake_render))
    app.config["TESTING"] = True
    return app


def test_post_queues_a_clip_and_the_candidate_then_carries_it(app: Flask) -> None:
    client = app.test_client()
    cands = client.get("/api/runs/evo/candidates").get_json()
    # The queue tests above share this run and already gave the first ones a clip.
    uid = next(c for c in cands if not c["animations"])["uid"]
    r = client.post(f"/api/runs/evo/animate/{uid}")
    assert r.status_code == 202
    assert uid in r.get_json()["pending"] or r.get_json()["running"] == uid
    assert app.extensions["animations"].wait(10)
    jobs = client.get("/api/runs/evo/jobs").get_json()
    assert jobs["done"] == {uid: ["anim_flip.mp4"]}
    cand = next(
        c for c in client.get("/api/runs/evo/candidates").get_json() if c["uid"] == uid
    )
    assert cand["animations"] == [
        {"slot": "flip", "path": f"round_000/{uid}/anim_flip.mp4"}
    ]
    assert client.get(f"/files/evo/{cand['animations'][0]['path']}").status_code == 200


def test_post_for_an_unknown_candidate_is_404(app: Flask) -> None:
    client = app.test_client()
    assert client.post("/api/runs/evo/animate/nope").status_code == 404
    assert client.post("/api/runs/nope/animate/x").status_code == 404


def test_post_while_queued_is_409(runs_dir: Path) -> None:
    gate = threading.Event()

    def blocking(root: Path, uid: str, cache: dict[str, Any]) -> list[str]:
        gate.wait(10)
        return []

    app = create_app(runs_dir, animations=AnimationQueue(render=blocking))
    client = app.test_client()
    uid = data.candidates(runs_dir / "evo")[0]["uid"]
    assert client.post(f"/api/runs/evo/animate/{uid}").status_code == 202
    assert client.post(f"/api/runs/evo/animate/{uid}").status_code == 409
    gate.set()
    assert app.extensions["animations"].wait(10)
