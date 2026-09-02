"""Draw the Step B figures from the persisted results. Never recomputes anything.

  fidelity_<reading>.png   scatter of a cheap reading's sep_min against the
                           full generation's, coloured by whether the full
                           generation held; AUC in the title
  ab_bars.png              yield and cells-held bars for v1 and v2

Reads results/stepB/fidelity/{scores.jsonl,fidelity.json} and
results/stepB/ab.json, whichever exist. PIL only, like the other renderers.

Run from the repository root:
    .venv/bin/python -m scripts.render_stepB
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from visual_anagrams.utils import get_courier_font_path

from ava.report import load_scores

W, H = 560, 560
MARGIN = 64
BG = (255, 255, 255)
AXIS = (60, 60, 60)
HELD = (30, 120, 200)
LOST = (200, 70, 60)
GRID = (225, 225, 225)


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(get_courier_font_path()), size)


def _scale(v: float, lo: float, hi: float, a: int, b: int) -> int:
    if hi <= lo:
        return (a + b) // 2
    return int(a + (v - lo) / (hi - lo) * (b - a))


def scatter(
    x: list[float],
    y: list[float],
    held: list[bool],
    title: str,
    xlabel: str,
    ylabel: str,
    out: Path,
) -> Path:
    lo = min(min(x), min(y), 0.0) - 0.02
    hi = max(max(x), max(y), 0.0) + 0.02
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    left, right, top, bottom = MARGIN, W - 24, 48, H - MARGIN

    zero_x = _scale(0.0, lo, hi, left, right)
    zero_y = _scale(0.0, lo, hi, bottom, top)
    d.line([(zero_x, top), (zero_x, bottom)], fill=GRID)
    d.line([(left, zero_y), (right, zero_y)], fill=GRID)
    # The diagonal: where a cheap reading would agree exactly.
    d.line(
        [(left, bottom), (right, top)],
        fill=GRID,
    )
    d.rectangle([left, top, right, bottom], outline=AXIS)
    for xi, yi, h in zip(x, y, held, strict=True):
        px = _scale(xi, lo, hi, left, right)
        py = _scale(yi, lo, hi, bottom, top)
        d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=HELD if h else LOST)
    f = _font(12)
    d.text((left, 14), title, font=_font(13), fill=AXIS)
    d.text((left, bottom + 8), f"{lo:+.2f}", font=f, fill=AXIS)
    d.text((right - 40, bottom + 8), f"{hi:+.2f}", font=f, fill=AXIS)
    d.text(((left + right) // 2 - 60, bottom + 28), xlabel, font=f, fill=AXIS)
    d.text((6, top), f"{hi:+.2f}", font=f, fill=AXIS)
    d.text((6, bottom - 12), f"{lo:+.2f}", font=f, fill=AXIS)
    d.text((6, (top + bottom) // 2), ylabel[:10], font=f, fill=AXIS)
    d.text((right - 200, top + 6), "blue: full generation held", font=f, fill=HELD)
    d.text((right - 200, top + 22), "red: it did not", font=f, fill=LOST)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def render_fidelity(root: Path, figures: Path) -> list[Path]:
    scores = root / "scores.jsonl"
    summary_path = root / "fidelity.json"
    if not scores.exists() or not summary_path.exists():
        return []
    rows = load_scores(scores)
    summary = json.loads(summary_path.read_text())
    reference = summary["reference"]
    by_fid: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        by_fid.setdefault(str(r["fidelity"]), {})[str(r["candidate"])] = r
    ref = by_fid[reference]
    written = []
    for label, stats in summary["readings"].items():
        table = by_fid[label]
        shared = sorted(set(table) & set(ref))
        x = [float(table[c]["sep_min"]) for c in shared]
        y = [float(ref[c]["sep_min"]) for c in shared]
        held = [ref[c]["diagnosis"] == "ok" for c in shared]
        a = stats["auc_vs_reference_pass"]
        rho = stats["spearman_vs_reference_sep_min"]
        title = (
            f"{label} vs {reference}: AUC={a:.2f} rho={rho:+.2f}"
            if a is not None
            else f"{label} vs {reference}: AUC undefined"
        )
        written.append(
            scatter(
                x,
                y,
                held,
                title,
                f"sep_min at {label}",
                f"sep_min at {reference}",
                figures / f"fidelity_{label.replace('@', '_')}.png",
            )
        )
    return written


def render_ab(path: Path, figures: Path) -> Path | None:
    if not path.exists():
        return None
    ab = json.loads(path.read_text())
    v1, v2 = ab["v1"], ab["v2"]
    img = Image.new("RGB", (W, 320), BG)
    d = ImageDraw.Draw(img)
    f = _font(12)
    d.text(
        (MARGIN, 14),
        "A/B at equal budget: v1 bandit vs v2 evolve",
        font=_font(13),
        fill=AXIS,
    )
    panels = [
        ("yield (held / evaluations)", v1["yield"], v2["yield"], 1.0),
        ("cells held", v1["cells_held"], v2["cells_held"], None),
    ]
    for p, (label, a, b, cap) in enumerate(panels):
        x0 = MARGIN + p * 250
        top, bottom = 60, 260
        hi = cap if cap is not None else max(a, b, 1) * 1.15
        for j, (val, colour, name) in enumerate(((a, LOST, "v1"), (b, HELD, "v2"))):
            bx = x0 + j * 90
            by = _scale(float(val), 0.0, hi, bottom, top)
            d.rectangle([bx, by, bx + 60, bottom], fill=colour)
            text = f"{val:.3f}" if cap is not None else str(val)
            d.text((bx, by - 16), text, font=f, fill=AXIS)
            d.text((bx + 20, bottom + 6), name, font=f, fill=AXIS)
        d.text((x0, bottom + 26), label, font=f, fill=AXIS)
    figures.mkdir(parents=True, exist_ok=True)
    out = figures / "ab_bars.png"
    img.save(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/stepB"))
    args = parser.parse_args()
    figures = args.root / "figures"
    for path in render_fidelity(args.root / "fidelity", figures):
        print(f"wrote {path}")
    ab = render_ab(args.root / "ab.json", figures)
    if ab is not None:
        print(f"wrote {ab}")


if __name__ == "__main__":
    main()
