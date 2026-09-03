"""Spearman's rho and the ROC AUC on hand-checkable inputs."""

import numpy as np
import pytest

from ava.rankstats import auc, ranks, spearman


def test_ranks_average_ties() -> None:
    assert list(ranks([10, 20, 20, 30])) == [1.0, 2.5, 2.5, 4.0]


def test_spearman_hand_examples() -> None:
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    # Ranks (1, 2, 3, 4) vs (2, 1, 4, 3): rho = 1 - 6 * 4 / (4 * 15) = 0.6
    assert spearman([1, 2, 3, 4], [2, 1, 4, 3]) == pytest.approx(0.6)
    assert spearman([1, 2, 3], [5, 5, 5]) == 0.0
    with pytest.raises(ValueError):
        spearman([1], [2])


def test_auc_hand_examples() -> None:
    # Perfect separation, reversed, and chance.
    assert auc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert auc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == 0.0
    assert auc([0.5, 0.5, 0.5, 0.5], [False, True, False, True]) == 0.5
    # One positive above one negative and below another: 0.5.
    assert auc([0.1, 0.5, 0.9], [False, True, False]) == 0.5
    # Two positives, two negatives, one pair inverted out of four: 0.75.
    assert auc([0.1, 0.2, 0.3, 0.9], [False, True, False, True]) == 0.75


def test_auc_is_undefined_without_both_classes() -> None:
    assert auc([0.1, 0.2], [True, True]) is None
    assert auc([0.1, 0.2], [False, False]) is None
    with pytest.raises(ValueError):
        auc([0.1], [True, False])


def test_auc_matches_a_brute_force_pair_count() -> None:
    rng = np.random.default_rng(0)
    s = rng.normal(size=60)
    y = rng.random(60) < 0.4
    pos, neg = s[y], s[~y]
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    assert auc(s, y) == pytest.approx(wins / (len(pos) * len(neg)))
