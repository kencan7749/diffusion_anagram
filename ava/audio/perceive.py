"""The two ways a listener hears one signal: forwards and backwards.

Deliberately separate from whatever decomposition a sampler eventually uses,
for the same reason `ava/perceive.py` is separate from the Factorized
Diffusion view classes: the thing being scored has to be the thing a person
would hear, not the algebra that produced it.

Time reversal is an involution -- `reverse_view(reverse_view(x)) is x` up to
copying -- which is what makes it the cheapest possible anagram view. There is
no parameter to tune and no inverse to derive.
"""

from collections.abc import Callable

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ava.audio.permute import (
    JIGSAW_BLOCKS,
    JIGSAW_PERM,
    JIGSAW_SEED,
    MOSAIC_BLOCK_S,
    MOSAIC_FADE_S,
    MOSAIC_SEED,
    block_slices,
    derangement,
)
from ava.image.tasks import IllusionTask


def forward_view(wave: np.ndarray) -> np.ndarray:
    """The signal as it is. Named rather than inlined so the pair reads as a pair."""
    return wave


def reverse_view(wave: np.ndarray) -> np.ndarray:
    """The signal played backwards.

    Indexes the last axis, not the first. Generators return (channels, n), and
    `wave[::-1]` on one of those swaps left and right while leaving time alone
    -- a wrong answer that raises nothing and, on a near-symmetric stereo
    image, sounds almost right.

    Returns a contiguous copy rather than a reversed view, because the CLAP
    feature extractor is given raw buffers and negative strides are a source
    of silent breakage downstream.
    """
    return np.ascontiguousarray(wave[..., ::-1])


# A view takes the signal and its sample rate: reversal ignores the rate, a
# filter cannot.
View = Callable[[np.ndarray, int], np.ndarray]

LOWPASS_ORDER = 8


def lowpass_view(cutoff_hz: float) -> View:
    """The signal heard through a wall or from far away: a low-pass filter.

    Butterworth of order 8, applied forwards and backwards so the phase is
    unchanged and the envelope is not smeared in one direction -- a filter
    with group delay would add a small time shift to exactly the axis the
    reversal track is about. This is also the filter the latent projector
    (`ava.audio.bands`) is fitted against, so the judge and the sampler mean
    the same thing by "low".
    """

    def view(wave: np.ndarray, sample_rate: int) -> np.ndarray:
        sos = butter(
            LOWPASS_ORDER, cutoff_hz, btype="low", fs=sample_rate, output="sos"
        )
        return np.ascontiguousarray(sosfiltfilt(sos, wave, axis=-1))

    return view


def permute_view(perm: tuple[int, ...]) -> View:
    """The recording cut into equal blocks and spliced in another order.

    Hard cuts, no crossfade: the judge must hear what the latent view did,
    and a crossfade would be a second, unmeasured operation. Position i of
    the result is source block `perm[i]`, with the same remainder rule as
    the latent side, so the two cut at the same instants when the waveform
    is exactly the decoded span of the latent (which the engines return).
    """

    def view(wave: np.ndarray, sample_rate: int) -> np.ndarray:
        slices = block_slices(wave.shape[-1], len(perm))
        out = wave.copy()
        for position, source in enumerate(perm):
            out[..., slices[position]] = wave[..., slices[source]]
        return out

    return view


def mosaic_view(
    block_s: float = MOSAIC_BLOCK_S,
    seed: int = MOSAIC_SEED,
    fade_s: float = MOSAIC_FADE_S,
) -> View:
    """The recording cut into `block_s` blocks, every one of them moved.

    The permutation is drawn from (number of blocks, seed), exactly as the
    latent side (`ava.audio.engine.FramePermute`) draws it from (frames,
    seed); the two agree when a block is one latent frame, which is what the
    mosaic task requires of its codec. Each block gets a raised-cosine fade
    of `fade_s` at both ends -- not a crossfade, blocks do not overlap -- so
    the cuts do not click. The latent side cannot fade, but its decoder
    smooths the cuts on its own, and the fade brought the decoded mosaic
    within 1 dB of the codec's floor where hard cuts left it at 4 dB.
    """

    def view(wave: np.ndarray, sample_rate: int) -> np.ndarray:
        size = int(round(block_s * sample_rate))
        blocks = wave.shape[-1] // size
        if blocks < 2:
            raise ValueError(f"need at least two {block_s} s blocks, got {blocks}")
        perm = derangement(blocks, seed)
        slices = block_slices(wave.shape[-1], blocks)
        fade = int(round(fade_s * sample_rate))
        ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(fade) + 0.5) / max(fade, 1))
        out = wave.copy()
        for position, source in enumerate(perm):
            block = wave[..., slices[source]].copy()
            if fade and block.shape[-1] > 2 * fade:
                block[..., :fade] *= ramp
                block[..., -fade:] *= ramp[::-1]
            out[..., slices[position]] = block
        return out

    return view


# Parameters of the views that have any, for a run's config.yaml.
VIEW_PARAMS: dict[str, dict[str, object]] = {
    f"jigsaw_{JIGSAW_BLOCKS}": {
        "blocks": JIGSAW_BLOCKS,
        "seed": JIGSAW_SEED,
        "perm": list(JIGSAW_PERM),
    },
    "mosaic_40ms": {
        "block_s": MOSAIC_BLOCK_S,
        "seed": MOSAIC_SEED,
        "fade_s": MOSAIC_FADE_S,
        "perm": "derangement(blocks, seed), blocks = length // 0.04 s",
    },
}


def _identity(wave: np.ndarray, sample_rate: int) -> np.ndarray:
    return forward_view(wave)


def _reversed(wave: np.ndarray, sample_rate: int) -> np.ndarray:
    return reverse_view(wave)


# Keyed by the names `IllusionTask.view_names` uses for audio tasks.
VIEWS: dict[str, View] = {
    "forward": _identity,
    "reverse": _reversed,
    "near": _identity,
    "far_750": lowpass_view(750.0),
    "whole": _identity,
    f"jigsaw_{JIGSAW_BLOCKS}": permute_view(JIGSAW_PERM),
    "mosaic_40ms": mosaic_view(),
}


def views_of(task: IllusionTask) -> list[View]:
    """The perceptual views of a task, in slot order."""
    missing = [name for name in task.view_names if name not in VIEWS]
    if missing:
        raise KeyError(f"{task.name}: no audio view named {missing}")
    return [VIEWS[name] for name in task.view_names]


def perceive(
    wave: np.ndarray, task: IllusionTask, sample_rate: int
) -> list[np.ndarray]:
    """What a listener hears of one signal under each of the task's views."""
    return [view(wave, sample_rate) for view in views_of(task)]
