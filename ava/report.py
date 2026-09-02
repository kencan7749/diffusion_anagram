"""The two harvests: a contact sheet for the eye, a component table for the record.

Both are pure functions of what a run already persisted. Nothing here
regenerates an image, re-runs the judge, or refits anything: the loop saved the
scored near/far views next to the scores, so changing a layout, a colour or a
label never triggers a recompute.

  contact_sheet.png  where a human picks the interesting images. Every cell
                     carries the original, its far view, both prompts and J,
                     because the screening score is explicitly not a measure of
                     interestingness and the person looking has to be able to
                     disagree with it.
  components.md      where the knowledge accumulates. The Beta posterior of
                     each arm IS the answer to "which prompts make good
                     illusions", and it keeps improving across runs.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from visual_anagrams.utils import get_courier_font_path

from ava.vocab import ROLES, Arm, Role, list_arms

CELL = 192
PAD = 8
HEADER = 22
CAPTION = 58
BG = (255, 255, 255)
FG = (20, 20, 20)
MUTED = (110, 110, 110)

# What each role's posterior was built from, shown in the table header so the
# numbers cannot be misread as a single undifferentiated "score".
ROLE_SIGNAL: dict[Role, str] = {
    "low": "p_far (survives blurring)",
    "high": "p_near (readable up close)",
    "style": "J (the pair worked at all)",
}


def rank_score(row: dict[str, Any]) -> float:
    """Order candidates by the weaker raw CLIP margin, not by J.

    Ordering by this is equivalent to ordering by J -- a two-way softmax is a
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
    # Rows written before sep_min was persisted still rank correctly.
    return min(float(row["sep_far"]), float(row["sep_near"]))


def load_scores(path: Path) -> list[dict[str, Any]]:
    """Read every scored candidate. Nothing is filtered out here."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(get_courier_font_path()), size)


def _shorten(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def build_contact_sheet(
    rows: list[dict[str, Any]],
    out: Path,
    columns: int = 6,
    cell: int = CELL,
    min_j: float | None = None,
) -> Path:
    """Lay out the candidates, best J first.

    Ordered by `rank_score` (the weaker raw CLIP margin), gated by J. The
    order is the same one J gives; the margin is simply legible where J has
    saturated.

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
    n_rows = (len(ranked) + columns - 1) // columns
    banner = HEADER if hidden else 0
    # Two images per cell, stacked: the original and its far view.
    cell_h = cell * 2 + CAPTION
    sheet = Image.new(
        "RGB",
        (columns * (cell + PAD) + PAD, banner + n_rows * (cell_h + PAD) + PAD),
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
        y = banner + PAD + (i // columns) * (cell_h + PAD)

        near = Image.open(str(row["image_path"])).convert("RGB").resize((cell, cell))
        far = Image.open(str(row["far_image_path"])).convert("RGB").resize((cell, cell))
        sheet.paste(near, (x, y))
        sheet.paste(far, (x, y + cell))

        draw.text(
            (x, y + 2 * cell + 3),
            f"sep={rank_score(row):+.3f}  J={float(row['j']):.3f}",
            font=f_body,
            fill=FG,
        )
        draw.text(
            (x, y + 2 * cell + 17),
            f"lo {_shorten(str(row['prompt_low']), width_chars)}\n"
            f"hi {_shorten(str(row['prompt_high']), width_chars)}\n"
            f"st {_shorten(str(row['style']) or '(none)', width_chars)}",
            font=f_small,
            fill=MUTED,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def _rank_role(
    conn: sqlite3.Connection, role: Role, rng: np.random.Generator
) -> list[str]:
    arms = sorted(list_arms(conn, role), key=lambda a: a.mean, reverse=True)
    lines = [
        f"### role = `{role}` — ranked by {ROLE_SIGNAL[role]}",
        "",
        "| word | mean | 90% CI | n | source |",
        "|---|---|---|---|---|",
    ]
    for arm in arms:
        lo, hi = arm.credible_interval(rng)
        word = f"`{arm.word}`" if arm.word else "_(no style prefix)_"
        lines.append(
            f"| {word} | {arm.mean:.3f} | {lo:.3f}–{hi:.3f} "
            f"| {arm.n_trials} | {arm.source} |"
        )
    lines.append("")
    return lines


def write_components(
    conn: sqlite3.Connection,
    out: Path,
    rng: np.random.Generator | None = None,
) -> Path:
    """Write the component ranking table.

    `rng` is injected so the credible intervals, which are estimated by
    sampling, are reproducible from a run's seed.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    arms: list[Arm] = list_arms(conn)
    untried = sum(1 for a in arms if a.n_trials == 0)

    lines = [
        "# Component ranking",
        "",
        "Each row is a bandit arm: a `(word, role)` pair, not a word. A word can",
        "rank high as the low-frequency subject and low as the high-frequency one,",
        "and that difference is the point of the table.",
        "",
        "`mean` is the Beta posterior mean, built from the signal named in each",
        "section heading. `n` is how many candidates contributed evidence; a high",
        "mean at `n = 0` is only the prior talking. The interval is an equal-tailed",
        "90% credible interval estimated by sampling.",
        "",
        f"Arms: {len(arms)} total, {untried} never tried.",
        "",
    ]
    for role in ROLES:
        lines += _rank_role(conn, role, rng)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_round_report(rows: list[dict[str, Any]], out: Path, round_index: int) -> Path:
    """Per-round summary: what was proposed, what held up, and how things failed."""
    by_diagnosis: dict[str, int] = {}
    by_origin: dict[str, int] = {}
    for row in rows:
        d = str(row.get("diagnosis", "?"))
        o = str(row.get("origin", "?"))
        by_diagnosis[d] = by_diagnosis.get(d, 0) + 1
        by_origin[o] = by_origin.get(o, 0) + 1

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
        "## Diagnoses",
        "",
    ]
    for name, count in sorted(by_diagnosis.items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{name}`: {count}")
    lines += ["", "## Where the candidates came from", ""]
    for name, count in sorted(by_origin.items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{name}`: {count}")

    lines += [
        "",
        "## Candidates, best first",
        "",
        "Ordered by `sep` = min(`sep_far`, `sep_near`), the weaker of the two raw",
        "CLIP cosine margins. J is not the sort key: its softmax runs at CLIP's",
        "logit_scale of 100, so a margin of 0.05 already reads as 0.99 and the",
        "best candidates come out tied. J still decides what is shown at all.",
        "",
        "| sep | J | p_far | p_near | sep_far | sep_near | diagnosis "
        "| low | high | style |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in ranked:
        style = str(r["style"]) or "—"
        lines.append(
            f"| {rank_score(r):+.3f} "
            f"| {float(r['j']):.3f} | {float(r['p_far']):.3f} "
            f"| {float(r['p_near']):.3f} "
            f"| {float(r['sep_far']):+.3f} | {float(r['sep_near']):+.3f} "
            f"| {r['diagnosis']} | {r['prompt_low']} | {r['prompt_high']} "
            f"| {style} |"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
