"""Does Stable Audio Open load and generate here?

A smoke test, not an experiment. It answers the one question that has to be
settled before Step 0b is worth writing: whether the model runs on this
machine and produces audio a person can listen to. Nothing here is scored.

The prompts are chosen to match what Step 0a found usable. CLAP separated a
sound and its reverse along the envelope and not along pitch direction, so
the sounds worth checking the generator on are ones with a definite temporal
shape -- a strike that decays, a swell that arrives.

    .venv/bin/python -m scripts.smoke_stable_audio
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ava.audio.wavfile import write_wav

MODEL_ID = "stabilityai/stable-audio-open-1.0"
PROMPTS = [
    "a match being struck, sharp attack then a slow decay",
    "a sound swelling out of silence into a sudden stop",
    "a hammer hitting an anvil once, ringing out",
]
NEGATIVE_PROMPT = "low quality"


def generate(
    out_dir: Path,
    model_id: str,
    steps: int,
    seconds: float,
    guidance_scale: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    from diffusers import StableAudioPipeline

    # See ava/audio/codec.py: Oobleck wraps its convs in the deprecated
    # torch.nn.utils.weight_norm, which computes the weight at construction and
    # so cannot be built on the meta device -- aten::_weight_norm_interface has
    # no Meta kernel in any torch through 2.5.1, and diffusers still uses that
    # spelling through 0.35. Turning off the low-memory path is the fix, not a
    # workaround for a version we happen to be pinned to.
    pipe = StableAudioPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=False
    )
    pipe = pipe.to(device)

    sample_rate = int(pipe.vae.sampling_rate)
    written: list[dict[str, Any]] = []
    for index, prompt in enumerate(PROMPTS):
        # Seeded per prompt so any one clip can be regenerated on its own.
        generator = torch.Generator(device).manual_seed(seed + index)
        audio = pipe(
            prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=steps,
            audio_end_in_s=seconds,
            num_waveforms_per_prompt=1,
            guidance_scale=guidance_scale,
            generator=generator,
        ).audios[0]

        wave = audio.float().cpu().numpy()

        # The model is not bounded to [-1, 1] and does overshoot it. Writing
        # such a clip straight to 16-bit PCM would clamp the peaks, so scale it
        # down instead and record the gain -- a silently clipped waveform is
        # not the audio that was generated, and these clips become inputs to
        # the view-validity measurement.
        peak = float(abs(wave).max())
        gain = 1.0 / peak if peak > 1.0 else 1.0
        path = write_wav(out_dir / f"{index:02d}.wav", wave * gain, sample_rate)

        written.append(
            {
                "prompt": prompt,
                "seed": seed + index,
                "path": str(path),
                "shape": list(wave.shape),
                "peak": peak,
                "gain_applied": gain,
                "silent": bool(peak < 1e-4),
            }
        )
        print(f"  {path}  peak={peak:.3f} gain={gain:.3f}  {prompt}")

    return {
        "model_id": model_id,
        "sample_rate": sample_rate,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "audio_end_in_s": seconds,
        "negative_prompt": NEGATIVE_PROMPT,
        "dtype": "float16",
        "device": device,
        "clips": written,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/smoke_stable_audio"))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    result = generate(
        args.out,
        args.model_id,
        args.steps,
        args.seconds,
        args.guidance_scale,
        args.seed,
        args.device,
    )
    path = args.out / "run.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
