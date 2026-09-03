"""The two harvests: a contact sheet for the eye, a component table for the record.

Both are pure functions of what a run already persisted. Nothing here
regenerates an image, re-runs the judge, or refits anything: the loop saved the
scored view images next to the scores, so changing a layout, a colour or a
label never triggers a recompute.

  contact_sheet.png  where a human picks the interesting images. Every cell
                     carries each view the judge looked at, every prompt and the
                     score, because the screening score is explicitly not a
                     measure of interestingness and the person looking has to
                     be able to disagree with it.
  components.md      where the knowledge accumulates. The Beta posterior of
                     each arm IS the answer to "which prompts make good
                     illusions in which view", and it keeps improving across
                     runs.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from visual_anagrams.utils import get_courier_font_path

from ava.vocab import STYLE, Arm, list_arms, list_tasks

CELL = 192
PAD = 8
HEADER = 22
LINE = 13
BG = (255, 255, 255)
FG = (20, 20, 20)
MUTED = (110, 110, 110)


def rank_score(row: dict[str, Any]) -> float:
    """Order candidates by the weakest raw CLIP margin, not by J.

    For two views this is the same order J gives -- a two-way softmax is a
    sigmoid of the margin, so J == sigmoid(logit_scale * sep_min) -- but it is
    readable and it does not run out of precision. A sweep's top 13 all printed
    as J >= 0.99 while their margins spanned +0.048 to +0.114, and past a margin
    of roughly 0.37 the sigmoid saturates to exactly 1.0 in float64, at which
    point a top-N selection really would be picking among ties.

    J still decides whether a candidate is shown at all. Screening wants the
    saturated signal; the ordering does not.
    """
    if "sep_min" in row:
        return float(row["sep_min"])
    # A row without the derived field still ranks by its weakest view.
    return min(float(x) for x in row["sep"])


def load_scores(path: Path) -> list[dict[str, Any]]:
    """Read every scored candidate. Nothing is filtered out here."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(get_courier_font_path()), size)


def _shorten(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def _caption_lines(row: dict[str, Any], width: int) -> list[str]:
    slots = [str(s) for s in row["slots"]]
    prompts = [str(p) for p in row["prompts"]]
    lines = [f"{row['task']}"]
    for slot, prompt in zip(slots, prompts, strict=True):
        lines.append(_shorten(f"{slot}: {prompt}", width))
    lines.append(_shorten(f"style: {row['style'] or '(none)'}", width))
    return lines


def build_contact_sheet(
    rows: list[dict[str, Any]],
    out: Path,
    columns: int = 6,
    cell: int = CELL,
    min_j: float | None = None,
) -> Path:
    """Lay out the candidates, best first.

    Ordered by `rank_score` (the weakest raw CLIP margin), gated by J. Each
    cell stacks every view the judge scored, in slot order, so a flip shows
    the image and its flipped copy and a triple hybrid shows three blur levels.

    `min_j` hides the tail that the visual check found not worth looking at.
    It only affects this drawing: every candidate stays in scores.jsonl, so
    the threshold can be moved and the sheet redrawn without regenerating a
    single image. How many were hidden is stamped on the sheet rather than
    left implicit, because a sheet that silently omits work reads as complete.
    """
    if not rows:
        raise ValueError("no scored candidates to draw")

    ranked = sorted(rows, key=rank_score, reverse=True)
    hidden = 0
    if min_j is not None:
        shown = [r for r in ranked if float(r["j"]) >= min_j]
        hidden = len(ranked) - len(shown)
        if not shown:
            raise ValueError(
                f"every candidate scored below min_j={min_j}; nothing to draw"
            )
        ranked = shown

    # Each grid row is as tall as its tallest cell (a four-view candidate
    # stacks four images), so a sheet mixing tasks does not pad every cell to
    # the largest task.
    grid_rows = [ranked[i : i + columns] for i in range(0, len(ranked), columns)]
    row_heights = []
    for group in grid_rows:
        views = max(len(r["view_paths"]) for r in group)
        lines = max(len(r["slots"]) for r in group) + 3  # task, score, style
        row_heights.append(cell * views + LINE * lines + 6)
    row_tops = [0] * len(grid_rows)
    for i in range(1, len(grid_rows)):
        row_tops[i] = row_tops[i - 1] + row_heights[i - 1] + PAD
    banner = HEADER if hidden else 0
    sheet = Image.new(
        "RGB",
        (
            columns * (cell + PAD) + PAD,
            banner + row_tops[-1] + row_heights[-1] + 2 * PAD,
        ),
        BG,
    )
    draw = ImageDraw.Draw(sheet)
    if hidden:
        draw.text(
            (PAD, 5),
            f"showing {len(ranked)} of {len(ranked) + hidden} candidates "
            f"(J >= {min_j}); {hidden} hidden, all of them in scores.jsonl",
            font=_font(12),
            fill=MUTED,
        )
    f_body = _font(11)
    f_small = _font(10)
    width_chars = max(8, cell // 6)

    for i, row in enumerate(ranked):
        x = PAD + (i % columns) * (cell + PAD)
        y = banner + PAD + row_tops[i // columns]

        for v, path in enumerate(row["view_paths"]):
            image = Image.open(str(path)).convert("RGB").resize((cell, cell))
            sheet.paste(image, (x, y + v * cell))

        text_y = y + len(row["view_paths"]) * cell + 3
        draw.text(
            (x, text_y),
            f"sep={rank_score(row):+.3f}  J={float(row['j']):.3f}",
            font=f_body,
            fill=FG,
        )
        draw.text(
            (x, text_y + LINE),
            "\n".join(_caption_lines(row, width_chars)),
            font=f_small,
            fill=MUTED,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def role_signal(role: str) -> str:
    """What each role's posterior was built from, so the numbers cannot be misread."""
    if role == STYLE:
        return "J (the candidate worked at all)"
    return f"p (the {role} view read as its prompt)"


def _rank_role(
    conn: sqlite3.Connection, task: str, role: str, rng: np.random.Generator
) -> list[str]:
    arms = sorted(list_arms(conn, task, role), key=lambda a: a.mean, reverse=True)
    lines = [
        f"### task = `{task}`, role = `{role}` — ranked by {role_signal(role)}",
        "",
        "| word | mean | 90% CI | n | source | citation |",
        "|---|---|---|---|---|---|",
    ]
    for arm in arms:
        lo, hi = arm.credible_interval(rng)
        word = f"`{arm.word}`" if arm.word else "_(no style prefix)_"
        lines.append(
            f"| {word} | {arm.mean:.3f} | {lo:.3f}–{hi:.3f} "
            f"| {arm.n_trials} | {arm.source} | {arm.citation} |"
        )
    lines.append("")
    return lines


def write_components(
    conn: sqlite3.Connection,
    out: Path,
    rng: np.random.Generator | None = None,
) -> Path:
    """Write the component ranking table, one section per (task, role).

    `rng` is injected so the credible intervals, which are estimated by
    sampling, are reproducible from a run's seed.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    arms: list[Arm] = list_arms(conn)
    untried = sum(1 for a in arms if a.n_trials == 0)

    lines = [
        "# Component ranking",
        "",
        "Each row is a bandit arm: a `(word, task, role)` triple, not a word. A",
        "word can rank high as the low-frequency subject of a hybrid and low as",
        "the high-frequency one, or work in a flip and not in a jigsaw, and that",
        "difference is the point of the table.",
        "",
        "`mean` is the Beta posterior mean, built from the signal named in each",
        "section heading. `n` is how many candidates contributed evidence; a high",
        "mean at `n = 0` is only the prior talking. The interval is an equal-tailed",
        "90% credible interval estimated by sampling.",
        "",
        "A posterior is not a component's standing alone: a favoured arm is paired",
        "with favoured partners, so its mean runs above what a uniform sweep",
        "would measure. Use `--uniform` runs to estimate standing.",
        "",
        f"Arms: {len(arms)} total, {untried} never tried.",
        "",
    ]
    for task in list_tasks(conn):
        roles = sorted({a.role for a in arms if a.task == task}, key=_role_order)
        for role in roles:
            lines += _rank_role(conn, task, role, rng)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _role_order(role: str) -> tuple[int, str]:
    return (1 if role == STYLE else 0, role)


def write_round_report(rows: list[dict[str, Any]], out: Path, round_index: int) -> Path:
    """Per-round summary: what was proposed, what held up, and how things failed."""
    by_diagnosis: dict[str, int] = {}
    by_origin: dict[str, int] = {}
    by_task: dict[str, int] = {}
    for row in rows:
        for table, key in (
            (by_diagnosis, "diagnosis"),
            (by_origin, "origin"),
            (by_task, "task"),
        ):
            name = str(row.get(key, "?"))
            table[name] = table.get(name, 0) + 1

    ranked = sorted(rows, key=rank_score, reverse=True)
    j_values = [float(r["j"]) for r in rows]
    held = sum(1 for r in rows if r.get("diagnosis") == "ok")

    lines = [
        f"# Round {round_index}",
        "",
        f"- candidates: {len(rows)}",
        f"- illusion held (`ok`): {held} / {len(rows)}",
        f"- J: max {max(j_values):.3f}, median {float(np.median(j_values)):.3f}, "
        f"min {min(j_values):.3f}",
        "",
        "## Tasks",
        "",
    ]
    for name, count in sorted(by_task.items()):
        lines.append(f"- `{name}`: {count}")
    lines += ["", "## Diagnoses", ""]
    for name, count in sorted(by_diagnosis.items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{name}`: {count}")
    lines += ["", "## Where the candidates came from", ""]
    for name, count in sorted(by_origin.items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{name}`: {count}")

    lines += [
        "",
        "## Candidates, best first",
        "",
        "Ordered by `sep` = the weakest view's raw CLIP margin over its runner-up",
        "prompt. J is not the sort key: its softmax runs at CLIP's logit_scale of",
        "100, so a margin of 0.05 already reads as 0.99 and the best candidates",
        "come out tied. J still decides what is shown at all.",
        "",
        "| sep | J | A | C | diagnosis | task | prompts (slot order) | style |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in ranked:
        style = str(r["style"]) or "—"
        prompts = " / ".join(str(p) for p in r["prompts"])
        lines.append(
            f"| {rank_score(r):+.3f} "
            f"| {float(r['j']):.3f} | {float(r['alignment']):.3f} "
            f"| {float(r['concealment']):.3f} "
            f"| {r['diagnosis']} | {r['task']} | {prompts} | {style} |"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
