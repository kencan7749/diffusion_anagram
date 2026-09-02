"""JSON endpoints: one per function of `ava.webui.data`, nothing more.

  GET /api/runs                        every run under the runs directory
  GET /api/runs/<run>                  summary (also the poll target: `version`)
  GET /api/runs/<run>/rounds           per-round statistics and the search log
  GET /api/runs/<run>/candidates       every scored candidate, media resolved
  GET /api/runs/<run>/lineage          pairs and parent -> child edges (evolve)
  GET /api/runs/<run>/grid             the MAP-Elites archive per task (evolve)
  POST /api/runs/<run>/animate/<uid>   queue the candidate's transition clips
  GET /api/runs/<run>/jobs             the clip queue for this run

A run name is a single path component; anything else is a 404, never a path.
"""

from __future__ import annotations

import re
from pathlib import Path

from flask import Blueprint, Response, abort, current_app, jsonify

from ava.webui import data

api = Blueprint("api", __name__, url_prefix="/api")

RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def runs_dir() -> Path:
    return Path(current_app.config["RUNS_DIR"])


def run_root(name: str) -> Path:
    """The directory of a named run, or a 404 if the name is not a run."""
    if not RUN_NAME.match(name) or ".." in name:
        abort(404)
    root = runs_dir() / name
    if not data.is_run_dir(root):
        abort(404)
    return root


@api.get("/runs")
def list_runs() -> Response:
    return jsonify(data.list_runs(runs_dir()))


@api.get("/runs/<name>")
def summary(name: str) -> Response:
    return jsonify(data.run_summary(run_root(name)))


@api.get("/runs/<name>/rounds")
def rounds(name: str) -> Response:
    return jsonify(data.rounds(run_root(name)))


@api.get("/runs/<name>/candidates")
def candidates(name: str) -> Response:
    return jsonify(data.candidates(run_root(name)))


@api.get("/runs/<name>/lineage")
def lineage(name: str) -> Response:
    return jsonify(data.lineage(run_root(name)))


@api.get("/runs/<name>/grid")
def grid(name: str) -> Response:
    return jsonify(data.archive_grid(run_root(name)))


@api.post("/runs/<name>/animate/<uid>")
def animate(name: str, uid: str) -> tuple[Response, int]:
    """202 with the queue state; 404 for an unknown uid; 409 if already queued."""
    root = run_root(name)
    if data.find_row(root, uid) is None:
        abort(404)
    queue = current_app.extensions["animations"]
    if not queue.submit(root, uid):
        abort(409)
    return jsonify(queue.status(root)), 202


@api.get("/runs/<name>/jobs")
def jobs(name: str) -> Response:
    return jsonify(current_app.extensions["animations"].status(run_root(name)))
