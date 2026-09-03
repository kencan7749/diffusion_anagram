"""Does AudioLDM 2 load and generate here, and is the mel front end right?

The counterpart of `scripts.smoke_stable_audio`, with one extra question. The
Stable Audio codec works on waveforms, so the only way to be wrong about its
input was the sample rate. AudioLDM 2's autoencoder takes a log-mel
spectrogram that diffusers never computes, and `ava.audio.mel` rebuilds the
recipe from the upstream training code. If that recipe is off -- a power
spectrum where a magnitude was expected, a different mel normalisation, the
wrong level -- the VAE still encodes something and Step 0b still prints
numbers, only they are numbers about the wrong spectrogram.

HiFi-GAN was trained on exactly the spectrogram the VAE was, so it is the
check: `vocode(log_mel(x))` must play back as x. The distance is read against
the same signal compared with a different probe (`mismatch_db`), as in
Step 0b, so that it can be called small.

    .venv/bin/python -m scripts.smoke_audioldm2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ava.audio.codec import AudioLDM2Codec
from ava.audio.features import log_mel_distance
from ava.audio.mel import log_mel_spectrogram, peak_normalize
from ava.audio.signals import percussive_burst, rising_chirp, symmetric_burst
from ava.audio.wavfile import write_wav
from scripts.smoke_stable_audio import NEGATIVE_PROMPT, PROMPTS

MODEL_ID = "cvssp/audioldm2"


def check_front_end(out_dir: Path, codec: AudioLDM2Codec) -> dict[str, Any]:
    """Vocoder round trip of the front end on the three probes."""
    rate = codec.sample_rate
    probes = {
        "rising_chirp": rising_chirp(sample_rate=rate),
        "percussive_burst": percussive_burst(sample_rate=rate),
        "symmetric_burst": symmetric_burst(sample_rate=rate),
    }
    names = list(probes)
    results: dict[str, Any] = {}
    for index, name in enumerate(names):
        # The front end normalises the level, so compare at that level.
        reference = peak_normalize(probes[name])
        played = codec.vocode(log_mel_spectrogram(probes[name]))
        mismatch = peak_normalize(probes[names[(index + 1) % len(names)]])
        write_wav(out_dir / "front_end" / f"{name}_input.wav", reference, rate)
        write_wav(out_dir / "front_end" / f"{name}_vocoded.wav", played, rate)
        results[name] = {
            "roundtrip_db": log_mel_distance(played, reference, rate),
            "mismatch_db": log_mel_distance(mismatch, reference, rate),
            "mismatch_against": names[(index + 1) % len(names)],
        }
        print(
            f"  front end {name:18s} roundtrip={results[name]['roundtrip_db']:.3f} dB"
            f"  mismatch={results[name]['mismatch_db']:.3f} dB"
        )
    return results


def generate(
    out_dir: Path,
    model_id: str,
    steps: int,
    seconds: float,
    guidance_scale: float,
    seed: int,
    device: str,
) -> list[dict[str, Any]]:
    from diffusers import AudioLDM2Pipeline

    pipe = AudioLDM2Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    sample_rate = int(pipe.vocoder.config.sampling_rate)

    written: list[dict[str, Any]] = []
    for index, prompt in enumerate(PROMPTS):
        generator = torch.Generator(device).manual_seed(seed + index)
        audio = pipe(
            prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=steps,
            audio_length_in_s=seconds,
            num_waveforms_per_prompt=1,
            guidance_scale=guidance_scale,
            generator=generator,
        ).audios[0]
        wave = audio.reshape(1, -1)
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
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/smoke_audioldm2"))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    codec = AudioLDM2Codec(model_id=args.model_id, device=args.device)
    front_end = check_front_end(args.out, codec)
    del codec
    torch.cuda.empty_cache()

    clips = generate(
        args.out,
        args.model_id,
        args.steps,
        args.seconds,
        args.guidance_scale,
        args.seed,
        args.device,
    )
    result = {
        "model_id": args.model_id,
        "sample_rate": 16_000,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "audio_length_in_s": args.seconds,
        "negative_prompt": NEGATIVE_PROMPT,
        "dtype": "float16",
        "device": args.device,
        "front_end": front_end,
        "clips": clips,
    }
    path = args.out / "run.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
