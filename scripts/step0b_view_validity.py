"""Step 0b: is flipping the latent the same operation as reversing the audio?

The reversal anagram works in latent space. Its sampler computes

    eps = 0.5 * (eps(z, p_fwd) + flip(eps(flip(z), p_rev)))

and the second branch only means "make it sound like p_rev when played
backwards" if flipping the latent's time axis decodes to the reversed
waveform. If it does not, the sampler still runs, still converges, and still
produces audio -- it just optimises a quantity with no perceptual reading.
Nothing raises. That is why this is measured before the sampler is written.

Four numbers per signal, and only the relationships between them mean anything:

    floor_forward   Dec(Enc(x))            vs x           the codec's own error
    floor_reverse   Dec(Enc(reverse(x)))   vs reverse(x)  the same, on the target
    measured        Dec(flip(Enc(x)))      vs reverse(x)  the question
    null            x                      vs reverse(x)  if flip did nothing
    mismatch        reverse(x)             vs another signal   the far end

`measured` is read against `floor_reverse`, not against zero: a codec that
loses 3 dB on any round trip cannot be expected to do better than 3 dB here.
`null` is the one that rules out the trivial explanation. If flipping the
latent were a no-op, Dec(flip(z)) would come back as x rather than reverse(x),
and `measured` would land on `null`. A low `measured` only means something when
`null` is high -- and for a time-symmetric signal `null` is near zero by
construction, which is why the symmetric control cannot answer this question
and the percussive burst is the one that can.

`mismatch` fixes the other end of the scale, so that a number can be called
small or large rather than just quoted. Without it, "4.1 dB" is unreadable --
the same trap Step 0a hit, where a self-similarity of 0.92 only became legible
once a control pinned what 'unchanged' looked like.

`latent_relative_error` compares flip(Enc(x)) with Enc(reverse(x)) directly.
It separates the two ways this can fail: the encoder not being equivariant,
versus the decoder undoing it.

    .venv/bin/python -m scripts.step0b_view_validity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ava.audio.codec import LatentCodec, StableAudioCodec
from ava.audio.features import log_mel_distance
from ava.audio.perceive import reverse_view
from ava.audio.signals import percussive_burst, rising_chirp, symmetric_burst
from ava.audio.wavfile import write_wav


def probe_signals(sample_rate: int) -> dict[str, np.ndarray]:
    """Synthetic probes, rendered at the codec's own rate.

    Deliberately the same three shapes Step 0a used, so the two results can be
    read side by side. They are out of distribution for an autoencoder trained
    on real recordings, but that inflates every number here equally and the
    verdict is a comparison between them, not an absolute.
    """
    return {
        "rising_chirp": rising_chirp(sample_rate=sample_rate),
        "percussive_burst": percussive_burst(sample_rate=sample_rate),
        "symmetric_burst": symmetric_burst(sample_rate=sample_rate),
    }


def load_generated(directory: Path) -> dict[str, np.ndarray]:
    """Clips written by scripts.smoke_stable_audio, if they are there.

    In-distribution audio matters: a codec asked to reconstruct a synthetic
    chirp is being asked something it was never trained for.
    """
    manifest = directory / "run.json"
    if not manifest.is_file():
        return {}

    import wave as wave_module

    out: dict[str, np.ndarray] = {}
    for index, clip in enumerate(json.loads(manifest.read_text())["clips"]):
        with wave_module.open(clip["path"], "rb") as handle:
            channels = handle.getnchannels()
            raw = handle.readframes(handle.getnframes())
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32767.0
        out[f"generated_{index:02d}"] = data.reshape(-1, channels).T
    return out


def measure(
    codec: LatentCodec, wave: np.ndarray, mismatch: np.ndarray
) -> dict[str, Any]:
    """All four distances, plus the latent-space diagnostic, for one signal."""
    rate = codec.sample_rate
    reversed_wave = reverse_view(wave) if wave.ndim == 1 else wave[:, ::-1].copy()

    z = codec.encode(wave)
    z_reversed = codec.encode(reversed_wave)
    flipped = codec.flip_time(z)

    decoded = {
        "floor_forward": codec.decode(z),
        "floor_reverse": codec.decode(z_reversed),
        "measured": codec.decode(flipped),
    }
    denominator = float(torch.linalg.vector_norm(z_reversed))
    return {
        "latent_shape": list(z.shape),
        "latent_relative_error": float(
            torch.linalg.vector_norm(flipped - z_reversed) / denominator
        ),
        "latent_cosine": float(
            torch.nn.functional.cosine_similarity(
                flipped.flatten(), z_reversed.flatten(), dim=0
            )
        ),
        "floor_forward_db": log_mel_distance(decoded["floor_forward"], wave, rate),
        "floor_reverse_db": log_mel_distance(
            decoded["floor_reverse"], reversed_wave, rate
        ),
        "measured_db": log_mel_distance(decoded["measured"], reversed_wave, rate),
        "null_db": log_mel_distance(wave, reversed_wave, rate),
        "mismatch_db": log_mel_distance(reversed_wave, mismatch, rate),
        "_decoded": decoded,
        "_reversed": reversed_wave,
    }


def run(out_dir: Path, codec: LatentCodec, signals: dict[str, np.ndarray]) -> Path:
    rate = codec.sample_rate
    names = list(signals)
    results: dict[str, Any] = {}

    for index, name in enumerate(names):
        # The far end of the scale: a different signal from the same set.
        mismatch = signals[names[(index + 1) % len(names)]]
        entry = measure(codec, signals[name], mismatch)

        audio_dir = out_dir / "audio" / name
        write_wav(audio_dir / "input.wav", signals[name], rate)
        write_wav(audio_dir / "target_reversed.wav", entry.pop("_reversed"), rate)
        for label, decoded in entry.pop("_decoded").items():
            write_wav(audio_dir / f"{label}.wav", decoded, rate)

        entry["mismatch_against"] = names[(index + 1) % len(names)]
        entry["headroom_db"] = entry["measured_db"] - entry["floor_reverse_db"]
        entry["fraction_of_mismatch"] = (
            entry["measured_db"] / entry["mismatch_db"] if entry["mismatch_db"] else 0.0
        )
        results[name] = entry

    payload = {
        "codec": type(codec).__name__,
        "model_id": getattr(codec, "model_id", "?"),
        "sample_rate": rate,
        "signals": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "scores.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/step0b"))
    parser.add_argument(
        "--generated", type=Path, default=Path("results/smoke_stable_audio")
    )
    parser.add_argument("--model-id", default="stabilityai/stable-audio-open-1.0")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    codec = StableAudioCodec(model_id=args.model_id, device=args.device)
    signals = {**probe_signals(codec.sample_rate), **load_generated(args.generated)}
    path = run(args.out, codec, signals)

    result = json.loads(path.read_text(encoding="utf-8"))
    print(f"wrote {path}\n")
    print(
        f"{'signal':20s} {'floor_rev':>10s} {'measured':>10s} {'headroom':>9s}"
        f" {'null':>8s} {'mismatch':>9s} {'lat_cos':>8s}"
    )
    for name, data in result["signals"].items():
        print(
            f"{name:20s} {data['floor_reverse_db']:10.3f} {data['measured_db']:10.3f}"
            f" {data['headroom_db']:+9.3f} {data['null_db']:8.3f}"
            f" {data['mismatch_db']:9.3f} {data['latent_cosine']:8.4f}"
        )
    print("\nheadroom = measured - floor_reverse: near 0 means flipping the latent")
    print("reverses the audio. null = what measured would be if the flip did")
    print("nothing; it must be large for headroom to mean anything.")


if __name__ == "__main__":
    main()
