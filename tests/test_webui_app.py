"""The Flask layer routes to the data layer and serves a run's files safely."""

from __future__ import annotations

from pathlib import Path

import pytest
from flask import Flask
from flask.testing import FlaskClient

from ava.webui.app import create_app
from tests.test_search_evolve import run as run_evolve

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def runs_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("webui_app")
    run_evolve(tmp, "evo", rounds=2, k=8)
    # The runs directory the loop uses is <tmp>/evo/, holding vocab.db + evo/.
    (tmp / "evo" / "secret.txt").write_text("not served")
    return tmp / "evo"


@pytest.fixture(scope="module")
def app(runs_dir: Path) -> Flask:
    app = create_app(runs_dir)
    app.config["TESTING"] = True
    return app


@pytest.fixture()
def client(app: Flask) -> FlaskClient:
    return app.test_client()


def test_index_is_the_single_page(client: FlaskClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert b"<title>" in r.data
    assert b"/static/app.js" in r.data


def test_runs_are_listed_with_their_summary(client: FlaskClient) -> None:
    r = client.get("/api/runs")
    assert r.status_code == 200
    runs = r.get_json()
    assert [x["name"] for x in runs] == ["evo"]
    assert runs[0]["has_archive"] is True
    assert runs[0]["n_candidates"] == 16


@pytest.mark.parametrize("path", ["", "/rounds", "/candidates", "/lineage", "/grid"])
def test_every_run_endpoint_answers(client: FlaskClient, path: str) -> None:
    r = client.get(f"/api/runs/evo{path}")
    assert r.status_code == 200
    assert r.mimetype == "application/json"
    assert r.get_json() is not None


def test_unknown_or_malformed_runs_are_404(client: FlaskClient) -> None:
    assert client.get("/api/runs/nope").status_code == 404
    assert client.get("/api/runs/..").status_code == 404
    assert client.get("/api/runs/.hidden").status_code == 404
    assert client.get("/api/runs/evo%2F..%2Fx").status_code == 404


def test_run_files_are_served_from_inside_the_run(client: FlaskClient) -> None:
    cand = client.get("/api/runs/evo/candidates").get_json()[0]
    media = cand["media"][0]["path"]
    r = client.get(f"/files/evo/{media}")
    assert r.status_code == 200
    assert r.mimetype == "image/png"
    assert client.get("/files/evo/config.yaml").status_code == 200


def test_run_files_cannot_escape_the_run(client: FlaskClient) -> None:
    assert client.get("/files/evo/../secret.txt").status_code == 404
    assert client.get("/files/evo/%2e%2e/secret.txt").status_code == 404
    assert client.get("/files/evo/missing.png").status_code == 404
    assert client.get("/files/nope/config.yaml").status_code == 404
