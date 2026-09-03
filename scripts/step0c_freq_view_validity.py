"""Step 0c: fit the low-pass projector on AudioLDM 2's latent, and check it holds.

The frequency hybrid needs a map on the latent that means "the low band"
(`ava.audio.bands`). It is fitted here by least squares from clips encoded as
they are and after low-pass filtering, then judged on clips that took no part
in the fit. As in Step 0b, only relationships between numbers mean anything:

    measured   Dec(low(Enc x))        vs lowpass(Dec(Enc x))   the question
    null       Dec(Enc x)             vs lowpass(Dec(Enc x))   if the map did nothing
    floor      Dec(Enc(lowpass x))    vs lowpass x             the codec's own error
    hi_drop    energy above the cutoff, Dec(low(Enc x)) minus Dec(Enc x), in dB
    lo_keep    the same below the cutoff (0 means the low band is intact)

`measured` is read against `floor`; `null` says how much there was to remove.

The projector is an analysis artefact: it depends on the clips, so they are
listed with their digests in the sidecar JSON, and every run that samples
with it names it explicitly (`--projector`).

    .venv/bin/python -m scripts.step0c_freq_view_validity --cutoff 750
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import wave as wave_module
from pathlib import Path
from typing import Any

import numpy as np

from ava.audio.bands import LowpassProjector, fit_lowpass_projector, to_frames
from ava.audio.codec import AudioLDM2Codec
from ava.audio.features import log_mel_distance, to_mono
from ava.audio.perceive import lowpass_view
from ava.audio.resample import resample
from ava.audio.signals import percussive_burst, rising_chirp
from ava.audio.wavfile import write_wav

# Every generated clip on disk from the reversal track, on both backends. The
# smoke clips are held out on purpose: they are the test set.
DEFAULT_TRAIN = [
    "results/step1/*/forward.wav",
    "results/step1_audioldm2*/*/forward.wav",
    "runs_audio_smoke/audio_smoke_audioldm2/round_000/*/forward.wav",
]
DEFAULT_TEST = ["results/smoke_audioldm2/*.wav", "results/smoke_stable_audio/*.wav"]


def read_mono(path: Path, sample_rate: int) -> np.ndarray:
    with wave_module.open(str(path), "rb") as handle:
        channels, rate = handle.getnchannels(), handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32767.0
    return resample(to_mono(data.reshape(-1, channels).T), rate, sample_rate)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def band_energy_db(
    wave: np.ndarray, cutoff_hz: float, rate: int
) -> tuple[float, float]:
    """(below, above) the cutoff, in dB, from the power spectrum."""
    mono = to_mono(wave)
    power = np.abs(np.fft.rfft(mono)) ** 2
    freqs = np.fft.rfftfreq(mono.size, 1.0 / rate)
    below = 10.0 * np.log10(power[freqs < cutoff_hz].sum() + 1e-12)
    above = 10.0 * np.log10(power[freqs >= cutoff_hz].sum() + 1e-12)
    return float(below), float(above)


def frames_of(codec: AudioLDM2Codec, wave: np.ndarray) -> np.ndarray:
    return to_frames(codec.encode(wave)).cpu().double().numpy()


def evaluate(
    codec: AudioLDM2Codec,
    projector: LowpassProjector,
    wave: np.ndarray,
    audio_dir: Path | None,
) -> dict[str, float]:
    rate = codec.sample_rate
    lowpass = lowpass_view(projector.cutoff_hz)
    z = codec.encode(wave)
    full = codec.decode(z)
    low = codec.decode(projector.low(z))
    target = lowpass(full, rate)
    lowpassed_input = lowpass(wave, rate)
    floor = log_mel_distance(
        codec.decode(codec.encode(lowpassed_input)), lowpassed_input, rate
    )
    below_full, above_full = band_energy_db(full, projector.cutoff_hz, rate)
    below_low, above_low = band_energy_db(low, projector.cutoff_hz, rate)
    if audio_dir is not None:
        write_wav(audio_dir / "input.wav", wave, rate)
        write_wav(audio_dir / "decoded.wav", full, rate)
        write_wav(audio_dir / "target_lowpassed.wav", target, rate)
        write_wav(audio_dir / "measured_low_component.wav", low, rate)
    return {
        "measured_db": log_mel_distance(low, target, rate),
        "null_db": log_mel_distance(full, target, rate),
        "floor_db": floor,
        "hi_drop_db": above_low - above_full,
        "lo_keep_db": below_low - below_full,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=float, default=750.0)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--train", nargs="*", default=DEFAULT_TRAIN, help="globs")
    parser.add_argument("--test", nargs="*", default=DEFAULT_TEST, help="globs")
    parser.add_argument("--out", type=Path, default=Path("results/step0c_freq_view"))
    parser.add_argument("--model-id", default="cvssp/audioldm2")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    codec = AudioLDM2Codec(model_id=args.model_id, device=args.device)
    rate = codec.sample_rate
    lowpass = lowpass_view(args.cutoff)

    train_paths = sorted(Path(p) for pattern in args.train for p in glob.glob(pattern))
    test_paths = sorted(Path(p) for pattern in args.test for p in glob.glob(pattern))
    if not train_paths:
        raise SystemExit(
            "no training clips found; run the reversal track's Step 1 first"
        )
    overlap = set(train_paths) & set(test_paths)
    if overlap:
        raise SystemExit(f"test clips must be held out of the fit: {sorted(overlap)}")

    latents, lowpassed = [], []
    for path in train_paths:
        wave = read_mono(path, rate)
        latents.append(frames_of(codec, wave))
        lowpassed.append(frames_of(codec, lowpass(wave, rate)))
    projector = fit_lowpass_projector(
        latents,
        lowpassed,
        cutoff_hz=args.cutoff,
        ridge=args.ridge,
        meta={
            "codec": type(codec).__name__,
            "model_id": args.model_id,
            "sample_rate": rate,
            "filter": "butterworth order 8, zero-phase (ava.audio.perceive)",
            "train_clips": [{"path": str(p), "md5": md5(p)} for p in train_paths],
        },
    )
    stem = args.out / f"projector_{int(args.cutoff)}hz"
    saved = projector.save(stem)

    tests: dict[str, np.ndarray] = {
        "percussive_burst": percussive_burst(sample_rate=rate),
        "rising_chirp": rising_chirp(sample_rate=rate),
    }
    for path in test_paths:
        tests[f"{path.parent.name}/{path.stem}"] = read_mono(path, rate)

    results: dict[str, Any] = {}
    print(f"fitted on {len(train_paths)} clips, {projector.meta['frames']} frames,")
    print(f"train relative residual {projector.meta['train_relative_residual']:.3f}\n")
    print(
        f"{'held-out signal':34s} {'measured':>8s} {'null':>6s} {'floor':>6s}"
        f" {'hi_drop':>8s} {'lo_keep':>8s}"
    )
    for name, wave in tests.items():
        entry = evaluate(
            codec, projector, wave, args.out / "audio" / name.replace("/", "_")
        )
        results[name] = entry
        print(
            f"{name:34s} {entry['measured_db']:8.2f} {entry['null_db']:6.2f}"
            f" {entry['floor_db']:6.2f} {entry['hi_drop_db']:+8.1f}"
            f" {entry['lo_keep_db']:+8.1f}"
        )

    payload = {
        "projector": str(saved),
        "projector_md5": md5(saved),
        "cutoff_hz": args.cutoff,
        "ridge": args.ridge,
        "train_clips": len(train_paths),
        "train_frames": projector.meta["frames"],
        "train_relative_residual": projector.meta["train_relative_residual"],
        "held_out": results,
    }
    path = args.out / "scores.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {saved} and {path}")
    print("measured is read against floor; null is what measured would be if the")
    print("map did nothing. hi_drop should be well below 0 dB, lo_keep near 0 dB.")


if __name__ == "__main__":
    main()
