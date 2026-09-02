"""The illusion score, shared by both tracks.

An anagram is scored the same way whatever it is made of: render it into the
two views a person can see, ask a contrastive model which prompt each view
matches, and take the *weaker* of the two answers. Nothing in that depends on
the views being spatial frequencies rather than time directions, so it lives
here rather than in either track's judge.

Kept free of torch's model machinery so it can be tested on a hand-written
matrix, without loading CLIP or CLAP.
"""

import torch


def scores_to_probs(s: torch.Tensor, logit_scale: float) -> tuple[float, float, float]:
    """Turn a 2x2 score matrix into (p_first, p_second, J).

    `s` is S[view][prompt] with both axes in the same order, so the diagonal
    is each view scored against its own prompt: (low, high) for hybrid images,
    (forward, reverse) for audio anagrams. Each row is a two-way choice, so
    chance level is 0.5, and the softmax temperature is the model's own
    logit_scale.

    J is the minimum, not the sum. A sum would reward a candidate that renders
    one view perfectly and the other not at all; an illusion only exists when
    both views hold.

    Note that J saturates: at a logit_scale of 100 a cosine margin of 0.05
    already gives 0.993. J is therefore a pass/fail signal, and the raw
    margins are what carry an ordering. See `Verdict.sep_min`.
    """
    if s.shape[-2:] != (2, 2):
        raise ValueError(f"expected a (2, 2) score matrix, got {tuple(s.shape)}")
    p = (logit_scale * s.float()).softmax(dim=-1)
    p_first = float(p[0, 0])
    p_second = float(p[1, 1])
    return p_first, p_second, min(p_first, p_second)
