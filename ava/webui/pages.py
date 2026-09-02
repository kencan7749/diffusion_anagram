"""The page, and the run's own files (view images, wavs, prompt cards).

`/files/<run>/<path>` serves from inside that run's directory only:
`send_from_directory` joins safely and turns any escape into a 404.
"""

from __future__ import annotations

from flask import Blueprint, Response, render_template, send_from_directory

from ava.webui.api import run_root

pages = Blueprint("pages", __name__)


@pages.get("/")
def index() -> str:
    return render_template("index.html")


@pages.get("/files/<name>/<path:subpath>")
def run_file(name: str, subpath: str) -> Response:
    return send_from_directory(run_root(name), subpath)
