"""Watch runs in the browser: lineage, archive coverage, per-round metrics, media.

Serves everything under `runs/` from the persisted files alone. It can run
beside a search that is still going; the page polls and follows new rounds.

Run from the repository root:
    .venv/bin/python -m scripts.webui --runs runs --port 8765

then open http://127.0.0.1:8765/ (over ssh: `ssh -L 8765:127.0.0.1:8765 host`).
The same app is available to the Flask CLI as `flask --app ava.webui run`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ava.webui.app import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", type=Path, default=Path("runs"), help="directory holding the runs"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--debug", action="store_true", help="Flask debug mode (auto-reload)"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    app = create_app(args.runs)
    print(f"serving {args.runs.resolve()} at http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
