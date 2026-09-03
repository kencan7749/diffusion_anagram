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

from ava.audio.features import describe
from ava.audio.judge import ClapJudge
from ava.audio.loop import AUDIO_TASK_NAMES, BACKENDS, AudioGenerator, build_engine
from ava.audio.perceive import perceive
from ava.audio.spec import AudioCandidateSpec
from ava.audio.tasks import (
    FREQ_HYBRID_750,
    TIME_JIGSAW_4,
    TIME_MOSAIC_40MS,
    TIME_REVERSE,
)
from ava.audio.wavfile import write_wav
from ava.image.tasks import IllusionTask, get_task
from ava.metric import scores_to_probs

# Same timbre, opposite temporal shape -- the only axis Step 0a showed to be
# usable. Each pair also has to be physically plausible in both directions, or
# the model is being asked for a sound that does not exist.
PROMPT_PAIRS: dict[str, list[tuple[str, str]]] = {
    TIME_REVERSE.name: [
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
    ],
    # (near, far): a texture heard in the room, and something whose identity
    # sits below 750 Hz, heard through the wall. Factorized Diffusion's advice
    # -- at least one flexible subject -- is followed on the near side.
    FREQ_HYBRID_750.name: [
        ("rain falling steadily on a roof", "distant thunder rumbling"),
        ("a crowd applauding in a hall", "a bass drum beating slowly"),
        ("food sizzling in a frying pan", "a truck engine idling"),
    ],
    # (whole, shuffled): two orders of events, so that cutting the first into
    # four blocks and re-splicing them can plausibly be the second.
    TIME_JIGSAW_4.name: [
        (
            "footsteps approaching, then a door closing",
            "a door closing, then footsteps walking away",
        ),
        (
            "a drum roll building up to a cymbal crash",
            "a cymbal crash followed by a drum roll",
        ),
        (
            "a car starting up and driving off",
            "a car arriving and the engine switching off",
        ),
    ],
    # (whole, mosaic): a structured sound, and the texture its 40 ms grains
    # could pass for once shuffled.
    TIME_MOSAIC_40MS.name: [
        ("a man giving a speech", "a crowd murmuring in a restaurant"),
        ("a piano melody", "wind chimes in a breeze"),
        ("morse code being tapped out", "rain falling steadily on a roof"),
    ],
}


def evaluate(
    engine: AudioGenerator,
    judge: ClapJudge,
    spec: AudioCandidateSpec,
    out: Path,
    task: IllusionTask = TIME_REVERSE,
) -> dict[str, Any]:
    """Generate one candidate, score every view, persist everything.

    The row keeps the reversal track's field names (`p_forward`, `sep_reverse`,
    ...) for every task: slot 0 is "forward" and slot 1 "reverse" in the
    sense of the first and second view, so earlier results stay comparable.
    """
    wave = engine.generate(spec)
    rate = engine.sample_rate
    views = perceive(wave, task, rate)

    matrix = judge.score_views(views, spec.prompts)
    p_forward, p_reverse, j = scores_to_probs(matrix, judge.logit_scale)
    sep_forward = float(matrix[0, 0] - matrix[0, 1])
    sep_reverse = float(matrix[1, 1] - matrix[1, 0])

    directory = out / spec.uid()
    wav_paths = {
        name: str(write_wav(directory / f"{name}.wav", view, rate))
        for name, view in zip(task.view_names, views, strict=True)
    }
    (directory / "prompt.txt").write_text(
        f"{task.slot_names[0]}: {spec.prompt_forward}\n"
        f"{task.slot_names[1]}: {spec.prompt_reverse}\n"
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
        "self_similarity": judge.view_similarity(views),
        "descriptors": {
            name: describe(view, rate)
            for name, view in zip(task.view_names, views, strict=True)
        },
        "wav": wav_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default results/step1 (stable_audio) or results/step1_<backend>",
    )
    parser.add_argument("--backend", choices=BACKENDS, default="stable_audio")
    parser.add_argument("--task", choices=AUDIO_TASK_NAMES, default=TIME_REVERSE.name)
    parser.add_argument("--projector", type=Path, default=None)
    parser.add_argument("--model-id", default=None, help="default: the backend's own")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    task = get_task(args.task)
    suffix = "" if args.backend == "stable_audio" else f"_{args.backend}"
    if task is not TIME_REVERSE:
        suffix += f"_{task.name}"
    out_dir: Path = args.out or Path(f"results/step1{suffix}")
    if task.name not in PROMPT_PAIRS:
        raise SystemExit(f"no first-anagram prompt pairs authored for {task.name}")

    engine = build_engine(
        args.backend, args.device, args.model_id, task.name, args.projector
    )
    judge = ClapJudge(device=args.device, sample_rate=engine.sample_rate)

    rows: list[dict[str, Any]] = []
    for prompt_forward, prompt_reverse in PROMPT_PAIRS[task.name]:
        for seed in range(args.seeds):
            spec = AudioCandidateSpec(
                prompt_forward=prompt_forward,
                prompt_reverse=prompt_reverse,
                seed=seed,
                duration_s=args.duration,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.steps,
            )
            row = evaluate(engine, judge, spec, out_dir, task)
            rows.append(row)
            mark = "HOLDS" if row["holds"] else "  -  "
            print(
                f"  {mark} J={row['j']:.4f}"
                f" sep={row['sep_forward']:+.4f}/{row['sep_reverse']:+.4f}"
                f" self_sim={row['self_similarity']:+.4f}"
                f"  seed={seed}  {prompt_forward[:40]}"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "scores.json"
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    held = sum(1 for r in rows if r["holds"])
    print(f"\n{held}/{len(rows)} candidates hold in both directions")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
