"""Step 0a: is CLAP sensitive to the direction of time?

The gate for the whole audio-anagram idea. The reversal view can only work if
the judge scores a sound and its reverse differently, and scores them
differently *in the direction the prompts point*. CLAP's audio tower pools
over time, so this is not a given.

Two questions are asked, in order, because they fail differently:

  1. self-similarity   cos(emb(x), emb(reverse(x)))
     Prompt-free. If a sound and its reverse land in the same place, no
     wording can recover the score and the encoder is the problem.

  2. score matrix      S[view][prompt], axes ordered (forward, reverse)
     Whether the movement is towards the right words. Every prompt pair is
     run against every signal, so a negative result can be attributed to a
     signal or to a phrasing rather than to the idea.

The control signal carries the reading. Reversing stationary noise under a
symmetric envelope changes every sample and nothing audible, so its
self-similarity is what "no perceptual change" looks like on this scale. A
chirp scoring near the control means CLAP is deaf to direction; a chirp
scoring well below it means the sensitivity is real.

Nothing is judged pass or fail here beyond the arithmetic. Every raw cosine is
written out, so a threshold can be moved and the figures redrawn without
loading the model again.

    .venv/bin/python -m scripts.step0_audio_judge
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ava.audio.features import describe, rms_profile
from ava.audio.judge import CLAP_ID, ClapJudge, scores_to_probs
from ava.audio.perceive import forward_view, reverse_view
from ava.audio.signals import DURATION, NOISE_SEED, PROBES, SAMPLE_RATE
from ava.audio.wavfile import write_wav

# Two phrasings per contrast, and every pair is run against every signal.
# One wording deciding the answer would be a finding about that wording.
PROMPT_PAIRS: list[dict[str, str]] = [
    {
        "family": "trajectory",
        "phrasing": "sweeping",
        "forward": "a sound sweeping upward in pitch",
        "reverse": "a sound sweeping downward in pitch",
    },
    {
        "family": "trajectory",
        "phrasing": "plain",
        "forward": "a rising tone",
        "reverse": "a falling tone",
    },
    {
        "family": "envelope",
        "phrasing": "descriptive",
        "forward": "a sharp hit that fades away",
        "reverse": "a sound swelling louder out of silence",
    },
    {
        "family": "envelope",
        "phrasing": "plain",
        "forward": "a percussive attack with a decay",
        "reverse": "a reversed sound building up",
    },
]


def evaluate_pair(
    judge: ClapJudge, wave: np.ndarray, pair: dict[str, str], logit_scale: float
) -> dict[str, Any]:
    """One signal against one prompt pair, with every intermediate kept."""
    s = judge.score_matrix(wave, pair["forward"], pair["reverse"])
    p_forward, p_reverse, j = scores_to_probs(s, logit_scale)

    # Margins, not just probabilities: a two-way softmax at CLAP's logit_scale
    # saturates, and the image pipeline already lost resolution that way once.
    sep_forward = float(s[0, 0] - s[0, 1])
    sep_reverse = float(s[1, 1] - s[1, 0])

    return {
        **pair,
        "s": [[float(v) for v in row] for row in s],
        "p_forward": p_forward,
        "p_reverse": p_reverse,
        "j": j,
        "sep_forward": sep_forward,
        "sep_reverse": sep_reverse,
        "sep_min": min(sep_forward, sep_reverse),
        # Both views preferring their own prompt is the only outcome the
        # anagram can be built on.
        "direction_ok": sep_forward > 0.0 and sep_reverse > 0.0,
    }


def evaluate_signal(
    judge: ClapJudge, name: str, wave: np.ndarray, audio_dir: Path, logit_scale: float
) -> dict[str, Any]:
    """Everything measured about one probe signal."""
    reversed_wave = reverse_view(wave)
    return {
        "self_similarity": judge.self_similarity(wave),
        "descriptors": {
            "forward": describe(forward_view(wave), SAMPLE_RATE),
            "reverse": describe(reversed_wave, SAMPLE_RATE),
        },
        # Persisted so the figures never have to reopen the audio: a plot
        # tweak must not be able to trigger a recompute.
        "rms_profile": {
            "forward": [float(v) for v in rms_profile(forward_view(wave))],
            "reverse": [float(v) for v in rms_profile(reversed_wave)],
        },
        "wav": {
            "forward": str(
                write_wav(audio_dir / f"{name}_forward.wav", wave, SAMPLE_RATE)
            ),
            "reverse": str(
                write_wav(audio_dir / f"{name}_reverse.wav", reversed_wave, SAMPLE_RATE)
            ),
        },
        "pairs": [
            evaluate_pair(judge, wave, pair, logit_scale) for pair in PROMPT_PAIRS
        ],
    }


def summarise(signals: dict[str, Any]) -> dict[str, Any]:
    """The two numbers the decision actually turns on.

    `direction_drop` is the control's self-similarity minus a signal's. It is
    the honest form of the question: not "did the embedding move" but "did it
    move more than it does for a sound whose reversal is inaudible".
    """
    control = float(signals["symmetric_burst"]["self_similarity"])
    return {
        "control_self_similarity": control,
        "direction_drop": {
            name: control - float(data["self_similarity"])
            for name, data in signals.items()
        },
        "pairs_with_correct_direction": {
            name: [
                p["phrasing"] + "/" + p["family"]
                for p in data["pairs"]
                if p["direction_ok"]
            ]
            for name, data in signals.items()
        },
    }


def run(out_dir: Path, device: str, clap_id: str) -> Path:
    judge = ClapJudge(device=device, clap_id=clap_id, sample_rate=SAMPLE_RATE)
    logit_scale = judge.logit_scale

    audio_dir = out_dir / "audio"
    signals = {
        name: evaluate_signal(judge, name, make(), audio_dir, logit_scale)
        for name, make in sorted(PROBES.items())
    }

    result = {
        "config": {
            "clap_id": clap_id,
            "device": device,
            "sample_rate": SAMPLE_RATE,
            "duration_s": DURATION,
            "noise_seed": NOISE_SEED,
            "logit_scale": logit_scale,
        },
        "prompt_pairs": PROMPT_PAIRS,
        "signals": signals,
        "summary": summarise(signals),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "scores.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/step0_audio"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clap-id", default=CLAP_ID)
    args = parser.parse_args()

    path = run(args.out, args.device, args.clap_id)
    result = json.loads(path.read_text(encoding="utf-8"))

    print(f"wrote {path}")
    print("\nself-similarity  cos(emb(x), emb(reverse(x)))")
    control = result["summary"]["control_self_similarity"]
    for name, drop in result["summary"]["direction_drop"].items():
        sim = result["signals"][name]["self_similarity"]
        print(f"  {name:18s} {sim:+.4f}   drop vs control {drop:+.4f}")
    print(f"  (control = {control:+.4f}: what an inaudible reversal looks like)")

    print("\nscore matrix  sep_forward / sep_reverse  (both > 0 is what we need)")
    for name, data in result["signals"].items():
        print(f"  {name}")
        for pair in data["pairs"]:
            mark = "ok " if pair["direction_ok"] else "-- "
            print(
                f"    {mark}{pair['family']:11s} {pair['phrasing']:11s}"
                f" {pair['sep_forward']:+.4f} / {pair['sep_reverse']:+.4f}"
                f"   J={pair['j']:.4f}"
            )


if __name__ == "__main__":
    main()
