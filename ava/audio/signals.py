"""Deterministic probe signals for validating the judge.

These exist to answer one question before any generator is chosen: does CLAP
hear the difference between a sound and the same sound played backwards?

The three signals are chosen to separate *why* an answer comes out the way it
does, because "CLAP scored them differently" on its own does not say what it
responded to:

  rising_chirp     flat envelope, moving spectrum  -> isolates trajectory
  percussive_burst flat spectrum, moving envelope  -> isolates envelope
  symmetric_burst  flat spectrum, symmetric envelope -> control

The control is the load-bearing one. Reversing stationary noise under a
symmetric envelope changes every sample but changes nothing a listener can
hear, so its embedding *should* survive reversal. Without it, a high
similarity on the chirp is ambiguous: it could mean CLAP is deaf to direction,
or it could mean the signal was direction-free to begin with. With it, the two
readings separate.

Everything here is a pure function of its arguments plus a fixed seed, so a
result can be regenerated exactly.
"""

from collections.abc import Callable

import numpy as np

SAMPLE_RATE = 48_000  # laion/clap-htsat-unfused is trained at 48 kHz
DURATION = 5.0  # CLAP truncates at 10 s; 5 s leaves margin and halves the cost
NOISE_SEED = 0

# A fade this short is inaudible but removes the click that a hard start or
# stop would put at the edge. It is applied symmetrically, so it cannot itself
# become the direction cue the experiment is trying to measure.
FADE_SECONDS = 0.005


def _time_axis(sample_rate: int, duration: float) -> np.ndarray:
    """Sample times in seconds, shape (n,)."""
    return np.arange(int(round(sample_rate * duration)), dtype=np.float64) / sample_rate


def _symmetric_fade(n: int, sample_rate: int) -> np.ndarray:
    """Raised-cosine fade in and out, identical at both ends."""
    fade = int(round(FADE_SECONDS * sample_rate))
    window = np.ones(n, dtype=np.float64)
    if fade > 0 and 2 * fade <= n:
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade) / fade))
        window[:fade] = ramp
        window[n - fade :] = ramp[::-1]
    return window


def _normalize(x: np.ndarray) -> np.ndarray:
    """Scale to a peak of 0.95, so nothing clips and level is not a cue."""
    peak = float(np.max(np.abs(x)))
    return x if peak == 0.0 else 0.95 * x / peak


def _stationary_noise(n: int, seed: int) -> np.ndarray:
    """Band-limited noise whose statistics do not change over time.

    Stationary is the point: any direction the judge reports must come from
    the envelope that gets multiplied in, not from the carrier.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    # A short moving average tilts the spectrum down without giving the
    # carrier any temporal structure of its own.
    kernel = np.ones(8) / 8.0
    return np.convolve(noise, kernel, mode="same")


def rising_chirp(
    sample_rate: int = SAMPLE_RATE,
    duration: float = DURATION,
    f_start: float = 200.0,
    f_end: float = 4000.0,
) -> np.ndarray:
    """Linear frequency sweep upward at constant amplitude.

    Reversed, this is exactly a downward sweep -- the cleanest possible
    instance of "same material, opposite trajectory", and the case the
    reversal view is supposed to be able to exploit.
    """
    t = _time_axis(sample_rate, duration)
    # Integrate the linear frequency ramp to get phase.
    phase = 2 * np.pi * (f_start * t + (f_end - f_start) * t**2 / (2 * duration))
    return _normalize(np.sin(phase) * _symmetric_fade(t.size, sample_rate))


def percussive_burst(
    sample_rate: int = SAMPLE_RATE,
    duration: float = DURATION,
    tau: float = 0.35,
    seed: int = NOISE_SEED,
) -> np.ndarray:
    """Sharp attack, exponential decay, over a stationary carrier.

    Reversed, this swells out of silence. The spectrum is the same throughout
    in both directions, so anything the judge notices is the envelope.
    """
    t = _time_axis(sample_rate, duration)
    envelope = np.exp(-t / tau)
    carrier = _stationary_noise(t.size, seed)
    return _normalize(carrier * envelope * _symmetric_fade(t.size, sample_rate))


def symmetric_burst(
    sample_rate: int = SAMPLE_RATE,
    duration: float = DURATION,
    seed: int = NOISE_SEED,
) -> np.ndarray:
    """Control: stationary carrier under a time-symmetric envelope.

    Reversal changes every sample and changes nothing audible. If the judge
    reports this as different, its sensitivity is not perceptual and the
    numbers from the other two signals cannot be read as evidence.
    """
    t = _time_axis(sample_rate, duration)
    centre = duration / 2.0
    envelope = np.exp(-(((t - centre) / (duration / 6.0)) ** 2))
    carrier = _stationary_noise(t.size, seed)
    return _normalize(carrier * envelope * _symmetric_fade(t.size, sample_rate))


# Registered in one place so the analysis script and its tests cannot drift.
PROBES: dict[str, Callable[[], np.ndarray]] = {
    "rising_chirp": rising_chirp,
    "percussive_burst": percussive_burst,
    "symmetric_burst": symmetric_burst,
}
