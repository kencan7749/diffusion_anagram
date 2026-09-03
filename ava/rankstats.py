"""Rank statistics used to check predictions against outcomes.

Two questions come up repeatedly: does a predicted score *order* candidates
the way the measured score does (Spearman's rho), and does a cheap score
*separate* the candidates that held from those that did not (the area under
the ROC curve)? Both are rank-based, so a saturated or rescaled predictor is
judged on its ordering alone. Numpy only.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def ranks(x: ArrayLike) -> np.ndarray:
    """1-based ranks, ties given their average rank."""
    a = np.asarray(x, dtype=np.float64)
    order = np.argsort(a, kind="stable")
    out = np.empty(len(a), dtype=np.float64)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        out[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return out


def spearman(a: ArrayLike, b: ArrayLike) -> float:
    """Spearman's rho; 0.0 when either side is constant."""
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("need two sequences of equal length >= 2")
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0.0 or ry.std() == 0.0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def auc(scores: ArrayLike, positive: ArrayLike) -> float | None:
    """Area under the ROC curve of `scores` for the boolean `positive` labels.

    Computed as the Mann-Whitney statistic: the probability that a random
    positive outscores a random negative, ties counting half. None when one
    class is empty, because the question has no answer then rather than an
    answer of 0.5.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(positive, dtype=bool)
    if len(s) != len(y):
        raise ValueError("scores and labels must have the same length")
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    r = ranks(s)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
