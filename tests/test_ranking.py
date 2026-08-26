"""Ranking uses the raw CLIP margins; screening still uses J.

The two orderings are the same one: a two-way softmax is a sigmoid of the
margin and both views share CLIP's logit_scale, so J == sigmoid(scale *
sep_min). The margin is used for ordering because it stays legible and precise
where J does not. In a real 40-candidate sweep the top 13 all printed as
J >= 0.99 while their margins spanned +0.048 to +0.114, and past a margin of
about 0.37 the sigmoid reaches exactly 1.0 in float64 and ties become real.
"""

from __future__ import annotations

import pytest

from ava.judge import scores_to_probs
from ava.report import rank_score
from ava.spec import Verdict

LOGIT_SCALE = 100.0


def verdict(sep_far: float, sep_near: float) -> Verdict:
    """A verdict with the requested margins, and the J those margins imply."""
    import torch

    s = torch.tensor([[0.30, 0.30 - sep_far], [0.30 - sep_near, 0.30]])
    p_far, p_near, j = scores_to_probs(s, LOGIT_SCALE)
    return Verdict(
        uid="x",
        s_far_low=0.30,
        s_far_high=0.30 - sep_far,
        s_near_low=0.30 - sep_near,
        s_near_high=0.30,
        p_far=p_far,
        p_near=p_near,
        j=j,
    )


def test_sep_min_is_the_weaker_margin() -> None:
    v = verdict(sep_far=0.12, sep_near=0.04)
    assert v.sep_min == pytest.approx(0.04)
    assert v.sep_min == min(v.sep_far, v.sep_near)


def test_sep_min_is_a_min_not_a_mean() -> None:
    """Same reason J is a min: one dead view means no illusion."""
    lopsided = verdict(sep_far=0.30, sep_near=-0.02)
    balanced = verdict(sep_far=0.06, sep_near=0.06)
    assert (lopsided.sep_far + lopsided.sep_near) / 2 > (
        balanced.sep_far + balanced.sep_near
    ) / 2
    assert lopsided.sep_min < balanced.sep_min


def test_sep_and_j_induce_the_same_order() -> None:
    """J is a monotone transform of sep_min, so ranking by either agrees.

    Pinned as a test because the switch to margins was first justified by a
    claim that J ordered the top candidates arbitrarily. It does not.
    """
    import math

    for sep_far in (-0.10, -0.02, 0.0, 0.01, 0.05, 0.20):
        for sep_near in (-0.10, -0.02, 0.0, 0.01, 0.05, 0.20):
            v = verdict(sep_far, sep_near)
            expected = 1 / (1 + math.exp(-LOGIT_SCALE * v.sep_min))
            assert v.j == pytest.approx(expected, abs=1e-6)


def test_sep_stays_legible_where_j_has_saturated() -> None:
    """What the switch actually buys: resolution, not a different order."""
    clear = verdict(sep_far=0.114, sep_near=0.114)
    marginal = verdict(sep_far=0.048, sep_near=0.048)

    # J puts both in the same "essentially 1.0" band a reader cannot rank by eye.
    assert clear.j > 0.99 and marginal.j > 0.99
    assert clear.j - marginal.j < 0.01
    # The margins underneath differ by more than a factor of two.
    assert clear.sep_min > 2 * marginal.sep_min


def test_j_ties_for_real_at_large_margins() -> None:
    """Past roughly 0.37 the sigmoid hits 1.0 in float64 and J stops ordering."""
    a = verdict(sep_far=0.40, sep_near=0.40)
    b = verdict(sep_far=0.60, sep_near=0.60)
    assert a.j == b.j == 1.0  # J genuinely cannot separate them
    assert b.sep_min > a.sep_min  # the margin still can


def test_rank_score_reads_the_persisted_field() -> None:
    import json

    row = json.loads(verdict(sep_far=0.11, sep_near=0.05).to_json())
    assert "sep_min" in row
    assert rank_score(row) == pytest.approx(0.05)


def test_rank_score_falls_back_for_rows_written_before_sep_min() -> None:
    """Older runs must still rank correctly without being regenerated."""
    legacy = {"sep_far": 0.11, "sep_near": 0.05, "j": 0.99}
    assert rank_score(legacy) == pytest.approx(0.05)


def test_ranking_and_screening_disagree_on_purpose() -> None:
    """A candidate can pass the J gate and still rank last. That is the design."""
    barely = verdict(sep_far=0.06, sep_near=0.06)
    strongly = verdict(sep_far=0.25, sep_near=0.25)
    assert barely.j > 0.5 and strongly.j > 0.5  # both pass screening
    assert strongly.sep_min > barely.sep_min  # but not the same rank


def test_a_failed_view_gives_a_negative_margin() -> None:
    """Sign is meaningful: below zero the view picked the wrong prompt."""
    v = verdict(sep_far=-0.05, sep_near=0.20)
    assert v.sep_min < 0
    assert v.p_far < 0.5
    assert v.diagnose() == "low_loses"
