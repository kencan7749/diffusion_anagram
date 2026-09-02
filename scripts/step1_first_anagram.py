"""Step 1: generate reversal anagrams and score them.

The first thing that produces the artefact the project is actually after: one
sound that reads as two, depending on which way it is played. Everything
before this measured whether the pieces could work; this asks whether they do.

The prompt pairs oppose *temporal envelope*, not pitch. That is not a stylistic
choice -- Step 0a found CLAP separates a sound from its reverse along the
envelope (a percussive burst drops to 0.525 self-similarity against a control
at 0.998) and essentially not at all along pitch direction (a rising chirp only
drops to 0.921, and no trajectory phrasing got both margins positive). A pair
built on rising versus falling pitch would be unscoreable by this judge however
well the sampler worked.

Scoring reuses the image pipeline's arithmetic unchanged: a 2x2 matrix
S[view][prompt] with both axes ordered (forward, reverse), and J = min of the
two, because a candidate that only works one way round is not a near miss.

Nothing is discarded. Every candidate is written with its full score
breakdown, so a threshold can be moved and the audio re-auditioned without
generating anything again.

    .venv/bin/python -m scripts.step1_first_anagram --seeds 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ava.audio.engine import AudioAnagramEngine
from ava.audio.features import describe
from ava.audio.judge import ClapJudge
from ava.audio.perceive import forward_view, reverse_view
from ava.audio.spec import AudioCandidateSpec
from ava.audio.wavfile import write_wav
from ava.metric import scores_to_probs

# Same timbre, opposite temporal shape -- the only axis Step 0a showed to be
# usable. Each pair also has to be physically plausible in both directions, or
# the model is being asked for a sound that does not exist.
PROMPT_PAIRS: list[tuple[str, str]] = [
    (
        "a match being struck, sharp attack then a slow decay",
        "a fire dying down into silence, then stopping",
    ),
    (
        "a hammer hitting an anvil once, ringing out",
        "a metallic sound swelling out of silence to a sudden stop",
    ),
    (
        "air rushing out of a balloon, fading away",
        "a balloon being inflated, building up to a stop",
    ),
]


def evaluate(
    engine: AudioAnagramEngine, judge: ClapJudge, spec: AudioCandidateSpec, out: Path
) -> dict[str, Any]:
    """Generate one candidate, score both views, persist everything."""
    wave = engine.generate(spec)
    rate = engine.sample_rate

    matrix = judge.score_matrix(wave, spec.prompt_forward, spec.prompt_reverse)
    p_forward, p_reverse, j = scores_to_probs(matrix, judge.logit_scale)
    sep_forward = float(matrix[0, 0] - matrix[0, 1])
    sep_reverse = float(matrix[1, 1] - matrix[1, 0])

    directory = out / spec.uid()
    write_wav(directory / "forward.wav", forward_view(wave), rate)
    write_wav(directory / "reverse.wav", reverse_view(wave), rate)
    (directory / "prompt.txt").write_text(
        f"forward: {spec.prompt_forward}\n"
        f"reverse: {spec.prompt_reverse}\n"
        f"seed   : {spec.seed}\n"
        f"J      : {j:.4f}\n",
        encoding="utf-8",
    )

    return {
        "uid": spec.uid(),
        "prompt_forward": spec.prompt_forward,
        "prompt_reverse": spec.prompt_reverse,
        "seed": spec.seed,
        "s": [[float(v) for v in row] for row in matrix],
        "p_forward": p_forward,
        "p_reverse": p_reverse,
        "j": j,
        "sep_forward": sep_forward,
        "sep_reverse": sep_reverse,
        "sep_min": min(sep_forward, sep_reverse),
        "holds": sep_forward > 0.0 and sep_reverse > 0.0,
        "self_similarity": judge.self_similarity(wave),
        "descriptors": {
            "forward": describe(forward_view(wave), rate),
            "reverse": describe(reverse_view(wave), rate),
        },
        "wav": {
            "forward": str(directory / "forward.wav"),
            "reverse": str(directory / "reverse.wav"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/step1"))
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    engine = AudioAnagramEngine(device=args.device)
    judge = ClapJudge(device=args.device, sample_rate=engine.sample_rate)

    rows: list[dict[str, Any]] = []
    for prompt_forward, prompt_reverse in PROMPT_PAIRS:
        for seed in range(args.seeds):
            spec = AudioCandidateSpec(
                prompt_forward=prompt_forward,
                prompt_reverse=prompt_reverse,
                seed=seed,
                duration_s=args.duration,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.steps,
            )
            row = evaluate(engine, judge, spec, args.out)
            rows.append(row)
            mark = "HOLDS" if row["holds"] else "  -  "
            print(
                f"  {mark} J={row['j']:.4f}"
                f" sep={row['sep_forward']:+.4f}/{row['sep_reverse']:+.4f}"
                f" self_sim={row['self_similarity']:+.4f}"
                f"  seed={seed}  {prompt_forward[:40]}"
            )

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "scores.json"
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    held = sum(1 for r in rows if r["holds"])
    print(f"\n{held}/{len(rows)} candidates hold in both directions")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
