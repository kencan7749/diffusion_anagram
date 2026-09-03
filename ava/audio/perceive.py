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


View = Callable[[np.ndarray], np.ndarray]

# Keyed by the names `IllusionTask.view_names` uses for audio tasks. Other
# modules add theirs here at import time (a registry, like the task table).
VIEWS: dict[str, View] = {"forward": forward_view, "reverse": reverse_view}


def views_of(task: IllusionTask) -> list[View]:
    """The perceptual views of a task, in slot order."""
    missing = [name for name in task.view_names if name not in VIEWS]
    if missing:
        raise KeyError(f"{task.name}: no audio view named {missing}")
    return [VIEWS[name] for name in task.view_names]


def perceive(wave: np.ndarray, task: IllusionTask) -> list[np.ndarray]:
    """What a listener hears of one signal under each of the task's views."""
    return [view(wave) for view in views_of(task)]
