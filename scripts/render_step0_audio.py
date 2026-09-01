"""Draw Step 0a from its persisted result. Reads scores.json, nothing else.

Deliberately unable to recompute anything: no CLAP, no signal synthesis, no
reopening of the WAVs. Changing a colour or a label here must never cost a
model load, and a figure must always be a pure function of the numbers that
were actually recorded.

    .venv/bin/python -m scripts.render_step0_audio
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from visual_anagrams.utils import get_courier_font_path

PANEL_W, PANEL_H = 520, 130
MARGIN, GAP = 60, 34
FORWARD_RGB, REVERSE_RGB = (30, 90, 200), (210, 70, 40)


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(get_courier_font_path()), size)


def _polyline(
    profile: list[float], peak: float, x0: int, y0: int
) -> list[tuple[int, int]]:
    """Map an RMS profile onto a panel, sharing one scale across all panels."""
    step = PANEL_W / max(1, len(profile) - 1)
    return [
        (int(x0 + i * step), int(y0 + PANEL_H - (v / peak) * PANEL_H))
        for i, v in enumerate(profile)
    ]


def draw_envelopes(result: dict[str, Any], out: Path) -> Path:
    """Level over time, forward against reversed, one panel per signal.

    This is the one genuinely visual claim in Step 0a: that the control's
    contour is unchanged by reversal while the percussive burst's is turned
    inside out. The self-similarity numbers say the same thing, but a reader
    should be able to see it without trusting them.
    """
    signals = result["signals"]
    names = sorted(signals)
    peak = max(
        max(max(s["rms_profile"]["forward"]), max(s["rms_profile"]["reverse"]))
        for s in signals.values()
    )

    width = PANEL_W + 2 * MARGIN
    height = MARGIN + len(names) * (PANEL_H + GAP + 26) + MARGIN
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    f_title, f_small = _font(13), _font(11)

    y = MARGIN
    for name in names:
        data = signals[name]
        sim = data["self_similarity"]
        draw.text((MARGIN, y - 20), f"{name}", fill="black", font=f_title)
        draw.text(
            (MARGIN + 260, y - 19),
            f"cos(fwd, rev) = {sim:+.4f}",
            fill="black",
            font=f_small,
        )
        draw.rectangle(
            [MARGIN, y, MARGIN + PANEL_W, y + PANEL_H], outline=(200, 200, 200)
        )
        draw.line(
            _polyline(data["rms_profile"]["forward"], peak, MARGIN, y),
            fill=FORWARD_RGB,
            width=2,
        )
        draw.line(
            _polyline(data["rms_profile"]["reverse"], peak, MARGIN, y),
            fill=REVERSE_RGB,
            width=2,
        )
        y += PANEL_H + GAP + 26

    draw.text((MARGIN, height - MARGIN + 8), "forward", fill=FORWARD_RGB, font=f_small)
    draw.text(
        (MARGIN + 80, height - MARGIN + 8), "reversed", fill=REVERSE_RGB, font=f_small
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def write_report(result: dict[str, Any], out: Path) -> Path:
    """The numbers, in the order the argument runs."""
    signals = result["signals"]
    summary = result["summary"]
    control = summary["control_self_similarity"]

    lines = [
        "# Step 0a - is CLAP sensitive to the direction of time?",
        "",
        f"model: `{result['config']['clap_id']}`  ",
        f"logit_scale: {result['config']['logit_scale']:.2f}  ",
        f"signals: {result['config']['duration_s']} s @ "
        f"{result['config']['sample_rate']} Hz, noise seed "
        f"{result['config']['noise_seed']}",
        "",
        "## 1. Does the embedding move at all?",
        "",
        "`cos(emb(x), emb(reverse(x)))`. The control is a stationary carrier under",
        "a symmetric envelope: reversing it changes every sample and nothing",
        "audible, so its value is what *no perceptual change* looks like here.",
        "",
        "| signal | self-similarity | drop vs control |",
        "|---|---|---|",
    ]
    for name in sorted(signals):
        sim = signals[name]["self_similarity"]
        lines.append(f"| `{name}` | {sim:+.4f} | {control - sim:+.4f} |")

    lines += [
        "",
        "## 2. Does it move towards the right words?",
        "",
        "`S[view][prompt]`, both axes ordered (forward, reverse). The anagram",
        "needs *both* margins positive: each view preferring its own prompt.",
        "One positive margin alone is a prompt bias, not direction sensitivity.",
        "",
        "| signal | family | phrasing | sep_fwd | sep_rev | J | both>0 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in sorted(signals):
        for pair in signals[name]["pairs"]:
            lines.append(
                f"| `{name}` | {pair['family']} | {pair['phrasing']} "
                f"| {pair['sep_forward']:+.4f} | {pair['sep_reverse']:+.4f} "
                f"| {pair['j']:.4f} | {'**yes**' if pair['direction_ok'] else 'no'} |"
            )

    lines += ["", "## Prompt pairs", ""]
    for pair in result["prompt_pairs"]:
        lines.append(
            f"- **{pair['family']} / {pair['phrasing']}** - "
            f"forward: _{pair['forward']}_ | reverse: _{pair['reverse']}_"
        )
    lines += ["", "Audio for every panel is under `audio/`.", ""]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results/step0_audio"))
    args = parser.parse_args()

    result = json.loads((args.run / "scores.json").read_text(encoding="utf-8"))
    figure = draw_envelopes(result, args.run / "figures" / "envelopes.png")
    report = write_report(result, args.run / "report.md")
    print(f"wrote {figure}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
