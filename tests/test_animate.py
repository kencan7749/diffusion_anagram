"""Transition clips are a pure function of a persisted sample and the run's views.

Real upstream views, real ffmpeg (the binary imageio-ffmpeg ships), tiny
images and a few frames per clip, so a test costs well under a second.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("imageio_ffmpeg")  # noqa: E402  (needs pkg_resources)

from ava.image import animate  # noqa: E402
from ava.image.tasks import get_task  # noqa: E402
from ava.image.views import ViewSet  # noqa: E402

FAST = animate.Timing(
    hold=4, text_fade=2, transition=3, motion_hold=4, motion_transition=200
)
SIZE = 64  # the motion frame needs im_size / 64 >= 1


def sample(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8))


def frames_of(path: Path) -> tuple[int, tuple[int, int]]:
    """Frame count and (height, width), straight from ffmpeg."""
    import imageio_ffmpeg

    n, _ = imageio_ffmpeg.count_frames_and_secs(str(path))
    gen = imageio_ffmpeg.read_frames(str(path))
    meta = next(gen)
    gen.close()
    w, h = meta["size"]
    return int(n), (int(h), int(w))


# -- which views ---------------------------------------------------------------


def test_every_registered_view_but_the_triple_hybrid_can_animate() -> None:
    unsupported = []
    for name in (
        "flip", "rotate_cw", "rotate_ccw", "rotate_180", "skew", "jigsaw",
        "inner_circle", "negate", "patch_permute", "pixel_permute",
        "square_hinge", "three_view", "four_view", "hybrid", "color_hybrid",
        "motion_hybrid", "inverse_hybrid", "triple_hybrid",
    ):  # fmt: skip
        vs = ViewSet.build(get_task(name))
        try:
            animate.animatable_slots(vs.task, vs.views)
        except animate.UnsupportedView:
            unsupported.append(name)
    assert unsupported == ["triple_hybrid"]


def test_plain_slot_is_never_animated() -> None:
    flip = ViewSet.build(get_task("flip"))
    assert animate.animatable_slots(flip.task, flip.views) == [1]
    hybrid = ViewSet.build(get_task("hybrid"))
    assert animate.animatable_slots(hybrid.task, hybrid.views) == [0]  # low, not high
    four = ViewSet.build(get_task("four_view"))
    assert animate.animatable_slots(four.task, four.views) == [1, 2, 3]


# -- clips -----------------------------------------------------------------


def test_flip_writes_one_clip_at_the_upstream_frame_size(tmp_path: Path) -> None:
    vs = ViewSet.build(get_task("flip"))
    paths = animate.animate_candidate(
        sample(),
        vs,
        ["a photo of a duck", "a photo of a rabbit"],
        tmp_path,
        timing=FAST,
    )
    assert [p.name for p in paths] == ["anim_flip.mp4"]
    n, (h, w) = frames_of(paths[0])
    assert (h, w) == (int(SIZE * 1.5), int(SIZE * 1.5))
    # hold/2 + fade + transition + fade + hold/2, boomeranged.
    assert n == 2 * (2 + 2 + 3 + 2 + 2)
    assert not list(tmp_path.glob("*.part.mp4"))


def test_four_view_writes_a_clip_per_rotation(tmp_path: Path) -> None:
    vs = ViewSet.build(get_task("four_view"))
    paths = animate.animate_candidate(
        sample(), vs, ["a", "b", "c", "d"], tmp_path, timing=FAST
    )
    assert [p.name for p in paths] == [
        "anim_rotate_cw.mp4",
        "anim_rotate_180.mp4",
        "anim_rotate_ccw.mp4",
    ]


def test_motion_hybrid_uses_the_motion_blur_animator(tmp_path: Path) -> None:
    vs = ViewSet.build(get_task("motion_hybrid"))
    paths = animate.animate_candidate(
        sample(), vs, ["a car", "a canyon"], tmp_path, timing=FAST
    )
    assert [p.name for p in paths] == ["anim_moving.mp4"]
    n, _ = frames_of(paths[0])
    assert n > 0


def test_triple_hybrid_is_refused_explicitly(tmp_path: Path) -> None:
    vs = ViewSet.build(get_task("triple_hybrid"))
    with pytest.raises(animate.UnsupportedView, match="triple_low_pass"):
        animate.animate_candidate(sample(), vs, ["a", "b", "c"], tmp_path, timing=FAST)
    assert not list(tmp_path.iterdir())


def test_motion_transition_must_suit_the_upstream_blur_window() -> None:
    with pytest.raises(ValueError, match="200"):
        animate.Timing(motion_transition=40)


def test_prompt_count_must_match_views(tmp_path: Path) -> None:
    vs = ViewSet.build(get_task("flip"))
    with pytest.raises(ValueError):
        animate.animate_candidate(sample(), vs, ["only one"], tmp_path, timing=FAST)


def test_existing_clips_are_kept_unless_forced(tmp_path: Path) -> None:
    vs = ViewSet.build(get_task("flip"))
    args = (sample(), vs, ["a", "b"], tmp_path)
    (first,) = animate.animate_candidate(*args, timing=FAST)
    stamp = first.stat().st_mtime_ns
    (again,) = animate.animate_candidate(*args, timing=FAST)
    assert again == first and again.stat().st_mtime_ns == stamp
    (forced,) = animate.animate_candidate(*args, timing=FAST, force=True)
    assert forced == first and forced.stat().st_mtime_ns != stamp


def test_animation_file_names_round_trip() -> None:
    p = animate.animation_path(Path("x"), "rotate_cw")
    assert p == Path("x/anim_rotate_cw.mp4")
    assert animate.slot_of_animation(p) == "rotate_cw"
    with pytest.raises(ValueError):
        animate.slot_of_animation(Path("view_flip.png"))


# -- from a run directory ----------------------------------------------------


def _write_run(root: Path) -> dict:
    root.mkdir(parents=True)
    (root / "config.yaml").write_text("run_id: r\ntasks:\n- flip\nview_seed: 3\n")
    cdir = root / "round_002" / "abc"
    cdir.mkdir(parents=True)
    sample().save(cdir / "sample_256.png")
    row = {
        "uid": "abc",
        "round": 2,
        "task": "flip",
        "prompts": ["a duck", "a rabbit"],
        "style": "a photo of",
        "image_path": "runs/r/round_002/abc/sample_256.png",
    }
    (root / "round_002" / "scores.jsonl").write_text(json.dumps(row) + "\n")
    return row


def test_viewsets_for_run_follow_the_config(tmp_path: Path) -> None:
    root = tmp_path / "r"
    _write_run(root)
    vs = animate.viewsets_for_run(root)
    assert list(vs) == ["flip"] and vs["flip"].seed == 3


def test_animate_row_finds_the_sample_by_the_run_layout(tmp_path: Path) -> None:
    root = tmp_path / "r"
    row = _write_run(root)
    paths = animate.animate_row(root, row, animate.viewsets_for_run(root), timing=FAST)
    assert paths == [root / "round_002" / "abc" / "anim_flip.mp4"]
    with pytest.raises(KeyError):
        animate.animate_row(root, {**row, "task": "hybrid"}, {}, timing=FAST)


# -- the CLI -----------------------------------------------------------------


def test_select_rows_filters_and_ranks() -> None:
    from scripts.animate_run import select_rows

    rows = [
        {"uid": "a", "diagnosis": "ok", "sep_min": 0.1},
        {"uid": "b", "diagnosis": "lost:flip", "sep_min": 0.3},
        {"uid": "c", "diagnosis": "ok", "sep_min": 0.2},
    ]
    assert [r["uid"] for r in select_rows(rows, held_only=True, top=None, uids=[])] == [
        "a",
        "c",
    ]
    assert [r["uid"] for r in select_rows(rows, held_only=False, top=2, uids=[])] == [
        "b",
        "c",
    ]
    assert [r["uid"] for r in select_rows(rows, held_only=True, top=1, uids=["b"])] == [
        "b"
    ]


def test_cli_writes_keeps_and_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from scripts.animate_run import animate_run, load_rows

    root = tmp_path / "r"
    _write_run(root)
    rows = load_rows(root)
    counts = animate_run(root, rows, timing=FAST)
    assert counts == {"written": 1, "kept": 0, "unsupported": 0, "missing": 0}
    counts = animate_run(root, rows, timing=FAST)
    assert counts["kept"] == 1 and counts["written"] == 0
    # A candidate whose sample is gone is reported, not fatal.
    (root / "round_002" / "abc" / "sample_256.png").unlink()
    (root / "round_002" / "abc" / "anim_flip.mp4").unlink()
    assert animate_run(root, rows, timing=FAST)["missing"] == 1
    assert "[missing]" in capsys.readouterr().out
