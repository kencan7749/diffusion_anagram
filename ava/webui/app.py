"""Application factory for the run viewer.

Flask's factory pattern: nothing is created at import time, the runs
directory is configuration rather than a global, and tests build an app per
temporary directory. The two blueprints are the whole surface:

  ava.webui.api    /api/...        JSON, straight from `ava.webui.data`
  ava.webui.pages  /  /files/...   the single page and the run's own files

Templates and static files live next to this module.
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask
from flask.json.provider import DefaultJSONProvider

from ava.webui.api import api
from ava.webui.jobs import AnimationQueue
from ava.webui.pages import pages

DEFAULT_RUNS_DIR = "runs"


def create_app(
    runs_dir: str | Path | None = None, animations: AnimationQueue | None = None
) -> Flask:
    """Build the viewer for the runs under `runs_dir` (default: `./runs`).

    `animations` is the clip queue; tests pass one with a fake renderer.
    """
    app = Flask(__name__)
    app.config["RUNS_DIR"] = str(Path(runs_dir or DEFAULT_RUNS_DIR).resolve())
    app.extensions["animations"] = animations or AnimationQueue()
    # Key order is meaningful to a reader of the API (slots before scores).
    provider = DefaultJSONProvider(app)
    provider.sort_keys = False
    app.json = provider
    app.register_blueprint(api)
    app.register_blueprint(pages)
    return app
