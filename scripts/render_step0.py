"""Render the Step 0 contact sheet for the human check (0-3).

Reads only results/step0/scores.jsonl. It never regenerates an image, never
re-runs the judge, and never recomputes a view: the analysis step persisted the
exact near/far images that were scored. Changing this sheet's layout must never
trigger a recompute.

Run from the repository root:
    .venv/bin/python -m scripts.render_step0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from visual_anagrams.utils import get_courier_font_path

CELL = 256
PAD = 12
HEADER = 26  # column titles
CAPTION = 74  # per-cell text under the image
LEFT = 120  # row label gutter

BG = (255, 255, 255)
FG = (20, 20, 20)
MUTED = (110, 110, 110)


def load_rows(scores: Path) -> list[dict[str, Any]]:
    with scores.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(get_courier_font_path()), size)


def build_sheet(rows: list[dict[str, Any]], out: Path) -> Path:
    """One row per generation sigma; columns are near, far(blur), far(resize)."""
    cases = sorted({str(r["case"]) for r in rows})
    far_modes = sorted({str(r["far_mode"]) for r in rows})
    columns = ["near"] + [f"far ({m})" for m in far_modes]

    width = LEFT + len(columns) * (CELL + PAD) + PAD
    height = HEADER + len(cases) * (CELL + CAPTION + PAD) + PAD
    sheet = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(sheet)
    f_title = get_font(15)
    f_body = get_font(12)
    f_small = get_font(11)

    prompts = rows[0]
    draw.text(
        (PAD, 6),
        f"low={prompts['prompt_low']}   high={prompts['prompt_high']}",
        font=f_small,
        fill=MUTED,
    )

    for ci, name in enumerate(columns):
        x = LEFT + ci * (CELL + PAD)
        draw.text((x, HEADER - 18), name, font=f_title, fill=FG)

    for ri, case in enumerate(cases):
        y = HEADER + ri * (CELL + CAPTION + PAD)
        case_rows = {str(r["far_mode"]): r for r in rows if r["case"] == case}
        any_row = next(iter(case_rows.values()))

        draw.text((PAD, y + CELL // 2 - 20), case, font=f_title, fill=FG)
        draw.text(
            (PAD, y + CELL // 2),
            f"gen sigma\n{any_row['generation_sigma']}",
            font=f_small,
            fill=MUTED,
        )

        # near view: identical across far_modes, so draw it once
        near = (
            Image.open(str(any_row["image_path"])).convert("RGB").resize((CELL, CELL))
        )
        sheet.paste(near, (LEFT, y))
        draw.text(
            (LEFT, y + CELL + 4),
            f"caption: {str(any_row['caption_near'])[:34]}",
            font=f_small,
            fill=MUTED,
        )

        for ci, mode in enumerate(far_modes, start=1):
            r = case_rows[mode]
            x = LEFT + ci * (CELL + PAD)
            far = (
                Image.open(str(r["far_image_path"])).convert("RGB").resize((CELL, CELL))
            )
            sheet.paste(far, (x, y))
            draw.text(
                (x, y + CELL + 4),
                f"J={float(r['j']):.3f}  {r['diagnosis']}\n"
                f"p_far={float(r['p_far']):.3f} p_near={float(r['p_near']):.3f}",
                font=f_body,
                fill=FG,
            )
            draw.text(
                (x, y + CELL + 40),
                f"caption: {str(r['caption_far'])[:34]}",
                font=f_small,
                fill=MUTED,
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/step0"))
    args = parser.parse_args()

    rows = load_rows(args.root / "scores.jsonl")
    out = build_sheet(rows, args.root / "figures" / "step0_contact.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
