"""Bandit bookkeeping: posterior updates, sampling, and the untried-arm quota."""

from pathlib import Path

import numpy as np
import pytest

from ava.vocab import (
    add_arm,
    connect,
    list_arms,
    seed_author_vocab,
    thompson_sample,
    untried_arms,
    update_arm,
)

SEED = 0


@pytest.fixture()
def conn(tmp_path: Path):
    c = connect(tmp_path / "vocab.db")
    yield c
    c.close()


def test_update_moves_the_posterior(conn) -> None:
    add_arm(conn, "a duck", "low", "test")
    before = next(a for a in list_arms(conn, "low") if a.word == "a duck")
    update_arm(conn, "a duck", "low", 0.9)
    after = next(a for a in list_arms(conn, "low") if a.word == "a duck")

    assert after.alpha == pytest.approx(before.alpha + 0.9)
    assert after.beta == pytest.approx(before.beta + 0.1)
    assert after.n_trials == before.n_trials + 1
    assert after.mean > before.mean


def test_update_rejects_non_probability(conn) -> None:
    add_arm(conn, "a duck", "low", "test")
    with pytest.raises(ValueError):
        update_arm(conn, "a duck", "low", 1.5)


def test_update_unknown_arm_raises(conn) -> None:
    with pytest.raises(KeyError):
        update_arm(conn, "not seeded", "low", 0.5)


def test_roles_are_independent_arms(conn) -> None:
    """Keying on (word, role): evidence must not leak from one role to another."""
    add_arm(conn, "albert einstein", "low", "test")
    add_arm(conn, "albert einstein", "high", "test")
    update_arm(conn, "albert einstein", "low", 1.0)

    low = next(a for a in list_arms(conn, "low") if a.word == "albert einstein")
    high = next(a for a in list_arms(conn, "high") if a.word == "albert einstein")
    assert low.n_trials == 1
    assert high.n_trials == 0
    assert low.mean > high.mean


def test_add_arm_never_overwrites(conn) -> None:
    add_arm(conn, "a duck", "low", "author", prior=(3.0, 1.0))
    update_arm(conn, "a duck", "low", 1.0)
    assert add_arm(conn, "a duck", "low", "mined", prior=(1.0, 1.0)) is False

    arm = next(a for a in list_arms(conn, "low") if a.word == "a duck")
    assert arm.alpha == pytest.approx(4.0)
    assert arm.source == "author"
    assert arm.n_trials == 1


def test_untried_arms_shrink_as_they_are_tried(conn) -> None:
    seed_author_vocab(conn)
    before = untried_arms(conn, "low")
    assert before, "a freshly seeded database has untried arms"
    update_arm(conn, before[0].word, "low", 0.5)
    after = {a.word for a in untried_arms(conn, "low")}
    assert before[0].word not in after
    assert len(after) == len(before) - 1


def test_thompson_sampling_is_reproducible(conn) -> None:
    seed_author_vocab(conn)
    a = thompson_sample(conn, "low", np.random.default_rng(SEED))
    b = thompson_sample(conn, "low", np.random.default_rng(SEED))
    assert a == b


def test_thompson_sampling_favours_the_better_arm(conn) -> None:
    add_arm(conn, "good", "low", "test", prior=(50.0, 1.0))
    add_arm(conn, "bad", "low", "test", prior=(1.0, 50.0))
    rng = np.random.default_rng(SEED)
    picks = [thompson_sample(conn, "low", rng).word for _ in range(200)]
    assert picks.count("good") > 190


def test_thompson_sampling_respects_exclusions(conn) -> None:
    add_arm(conn, "good", "low", "test", prior=(50.0, 1.0))
    add_arm(conn, "bad", "low", "test", prior=(1.0, 50.0))
    rng = np.random.default_rng(SEED)
    arm = thompson_sample(conn, "low", rng, exclude=frozenset({"good"}))
    assert arm.word == "bad"


def test_thompson_sampling_with_no_arms_raises(conn) -> None:
    with pytest.raises(LookupError):
        thompson_sample(conn, "low", np.random.default_rng(SEED))


def test_credible_interval_brackets_the_mean(conn) -> None:
    add_arm(conn, "a duck", "low", "test", prior=(8.0, 2.0))
    arm = next(a for a in list_arms(conn, "low") if a.word == "a duck")
    lo, hi = arm.credible_interval(np.random.default_rng(SEED))
    assert 0.0 <= lo < arm.mean < hi <= 1.0
