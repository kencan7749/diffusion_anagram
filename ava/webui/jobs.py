"""On-demand rendering of transition clips for the viewer, one at a time.

A viewer asks for a clip it does not have yet; the request is queued and a
single worker thread renders clips in order with `ava.image.animate`, which is
imported inside the job so the Flask process does not load torch until a clip
is actually asked for. Results land on disk beside the candidate's stills; the
page notices them the way it notices any other new file, through the run's
`version`.

The renderer is injected so the queue can be tested without torch or ffmpeg.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ava.webui import data

Renderer = Callable[[Path, str, dict[str, Any]], list[str]]
JobKey = tuple[str, str]  # (run root, uid)


def render_clips(root: Path, uid: str, cache: dict[str, Any]) -> list[str]:
    """The real renderer: the candidate's clips, file names returned.

    `cache` keeps one set of rebuilt view objects per run root across jobs.
    """
    from ava.image import animate  # torch and the vendored checkout, lazily

    row = data.find_row(root, uid)
    if row is None:
        raise KeyError(f"no candidate {uid} in {root}")
    key = str(root)
    if key not in cache:
        cache[key] = animate.viewsets_for_run(root)
    return [p.name for p in animate.animate_row(root, row, cache[key])]


class AnimationQueue:
    """A FIFO of (run, uid) jobs served by one daemon thread."""

    def __init__(self, render: Renderer = render_clips) -> None:
        self._render = render
        self._cache: dict[str, Any] = {}
        self._cv = threading.Condition()
        self._pending: list[JobKey] = []
        self._running: JobKey | None = None
        self._done: dict[JobKey, list[str]] = {}
        self._failed: dict[JobKey, str] = {}
        self._thread: threading.Thread | None = None

    # -- public --------------------------------------------------------------

    def submit(self, root: Path, uid: str) -> bool:
        """Queue a job; False if it is already queued or running."""
        key = (str(root), uid)
        with self._cv:
            if key in self._pending or key == self._running:
                return False
            self._done.pop(key, None)
            self._failed.pop(key, None)
            self._pending.append(key)
            self._ensure_worker()
            self._cv.notify()
        return True

    def status(self, root: Path) -> dict[str, Any]:
        """What the queue holds for one run, keyed by uid."""
        r = str(root)
        with self._cv:
            return {
                "pending": [uid for (root_, uid) in self._pending if root_ == r],
                "running": self._running[1]
                if self._running and self._running[0] == r
                else None,
                "done": {
                    uid: files
                    for (root_, uid), files in self._done.items()
                    if root_ == r
                },
                "failed": {
                    uid: msg for (root_, uid), msg in self._failed.items() if root_ == r
                },
            }

    def wait(self, timeout: float = 30.0) -> bool:
        """Block until idle (for tests and scripts); False on timeout."""
        with self._cv:
            return self._cv.wait_for(
                lambda: not self._pending and self._running is None, timeout=timeout
            )

    # -- worker --------------------------------------------------------------

    def _ensure_worker(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._run, name="ava-animations", daemon=True
            )
            self._thread.start()

    def _run(self) -> None:
        while True:
            with self._cv:
                self._cv.wait_for(lambda: bool(self._pending))
                key = self._pending.pop(0)
                self._running = key
            root, uid = key
            try:
                files = self._render(Path(root), uid, self._cache)
            except Exception as e:  # a failed clip must not kill the worker
                with self._cv:
                    self._failed[key] = f"{type(e).__name__}: {e}"
            else:
                with self._cv:
                    self._done[key] = list(files)
            finally:
                with self._cv:
                    self._running = None
                    self._cv.notify_all()
