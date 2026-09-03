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
