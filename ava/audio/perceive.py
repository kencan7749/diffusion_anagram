"""The two ways a listener hears one signal: forwards and backwards.

Deliberately separate from whatever decomposition a sampler eventually uses,
for the same reason `ava/perceive.py` is separate from the Factorized
Diffusion view classes: the thing being scored has to be the thing a person
would hear, not the algebra that produced it.

Time reversal is an involution -- `reverse_view(reverse_view(x)) is x` up to
copying -- which is what makes it the cheapest possible anagram view. There is
no parameter to tune and no inverse to derive.
"""

import numpy as np


def forward_view(wave: np.ndarray) -> np.ndarray:
    """The signal as it is. Named rather than inlined so the pair reads as a pair."""
    return wave


def reverse_view(wave: np.ndarray) -> np.ndarray:
    """The signal played backwards.

    Returns a contiguous copy rather than a reversed view, because the CLAP
    feature extractor is given raw buffers and negative strides are a source
    of silent breakage downstream.
    """
    return np.ascontiguousarray(wave[::-1])


VIEWS = {"forward": forward_view, "reverse": reverse_view}
