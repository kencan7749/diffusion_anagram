"""The illusion score, shared by both tracks.

An anagram is scored the same way whatever it is made of: render it into the
views a person can see, ask a contrastive model which prompt each view
matches, and take the *weakest* of the answers. Nothing in that depends on the
views being spatial frequencies, rotations or time directions, so it lives here
rather than in either track's judge.

Kept free of torch's model machinery so it can be tested on a hand-written
matrix, without loading CLIP or CLAP.
"""

import torch


def _check_square(s: torch.Tensor) -> int:
    if s.ndim != 2 or s.shape[0] != s.shape[1] or s.shape[0] < 2:
        raise ValueError(
            f"expected an (N, N) score matrix with N >= 2, got {tuple(s.shape)}"
        )
    return int(s.shape[0])


def multiway_probs(s: torch.Tensor, logit_scale: float) -> tuple[list[float], float]:
    """Turn an NxN score matrix into (p per view, J).

    `s` is S[view][prompt] with both axes in the same order, so the diagonal is
    each view scored against its own prompt. Each row is an N-way choice, so
    chance level is 1/N, and the softmax temperature is the model's own
    logit_scale.

    J is the minimum, not the mean. A mean would reward a candidate that
    renders one view perfectly and the others not at all; an illusion only
    exists when every view holds.

    Note that J saturates: at a logit_scale of 100 a cosine margin of 0.05
    already gives 0.993. J is therefore a pass/fail signal, and the raw margins
    are what carry an ordering. See `Verdict.sep_min`.
    """
    n = _check_square(s)
    p = (logit_scale * s.float()).softmax(dim=-1)
    diag = [float(p[i, i]) for i in range(n)]
    return diag, min(diag)


def scores_to_probs(s: torch.Tensor, logit_scale: float) -> tuple[float, float, float]:
    """Two-view form of `multiway_probs`: (p_first, p_second, J).

    Kept for the two-view callers (the audio track, the Step 0 scripts) that
    read the two probabilities positionally.
    """
    if s.shape[-2:] != (2, 2):
        raise ValueError(f"expected a (2, 2) score matrix, got {tuple(s.shape)}")
    p, j = multiway_probs(s, logit_scale)
    return p[0], p[1], j


def alignment(s: torch.Tensor) -> float:
    """Visual Anagrams' alignment score A: the worst view's own-prompt score."""
    n = _check_square(s)
    return float(min(float(s[i, i]) for i in range(n)))


def concealment(s: torch.Tensor, logit_scale: float) -> float:
    """Visual Anagrams' concealment score C (Eq. 9).

    The mean of the diagonal of softmax(S / tau), averaged over both softmax
    directions (over prompts for each view, and over views for each prompt).
    Chance level is 1/N. Unlike J this is a mean, so it is reported for
    comparison with the paper and never used to rank or screen.
    """
    _check_square(s)
    logits = logit_scale * s.float()
    rows = logits.softmax(dim=-1).diagonal().mean()
    cols = logits.softmax(dim=-2).diagonal().mean()
    return float((rows + cols) / 2.0)
