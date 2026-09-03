"""Block permutations of time, shared by the latent view and the listener's view.

The time jigsaw cuts a sound into equal blocks and plays them in another
order -- Visual Anagrams' jigsaw with time in place of the plane. The same
permutation has to be applied to the latent inside the sampler and to the
waveform a listener hears, so the permutation itself lives here, as plain
integers with no numpy or torch, and the two sides import it.

A permutation is drawn once from a seed and never changed: it is a fixed
parameter of the task, like the patch permutations of the image track, and
runs record which one they used. A derangement (no block stays in place) is
required, otherwise part of the "shuffled" view is just the original.
"""

from __future__ import annotations

import random

JIGSAW_BLOCKS = 4
JIGSAW_SEED = 0


def derangement(blocks: int, seed: int) -> tuple[int, ...]:
    """A permutation of `range(blocks)` with no fixed point, drawn from `seed`.

    `perm[i]` is the index of the source block that lands in position i.
    Rejection sampling; for four blocks about a third of draws are accepted.
    """
    if blocks < 2:
        raise ValueError("a derangement needs at least two blocks")
    rng = random.Random(seed)
    while True:
        perm = list(range(blocks))
        rng.shuffle(perm)
        if all(source != position for position, source in enumerate(perm)):
            return tuple(perm)


def inverse_permutation(perm: tuple[int, ...]) -> tuple[int, ...]:
    """The permutation that undoes `perm`."""
    inverse = [0] * len(perm)
    for position, source in enumerate(perm):
        inverse[source] = position
    return tuple(inverse)


def block_slices(length: int, blocks: int) -> list[slice]:
    """Equal blocks over the first `blocks * (length // blocks)` positions.

    Whatever does not divide evenly is left at the end, untouched, so the
    same rule on a latent of T frames and a waveform of T * hop samples cuts
    at the same instants.
    """
    size = length // blocks
    return [slice(i * size, (i + 1) * size) for i in range(blocks)]


JIGSAW_PERM: tuple[int, ...] = derangement(JIGSAW_BLOCKS, JIGSAW_SEED)
