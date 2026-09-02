"""Bandit bookkeeping: posterior updates, sampling, and the untried-arm quota."""

from pathlib import Path

import numpy as np
import pytest

from ava.vocab import (
    add_arm,
    connect,
    list_arms,
    list_tasks,
    seed_author_vocab,
    thompson_sample,
    untried_arms,
    update_arm,
)

SEED = 0
T = "hybrid"


@pytest.fixture()
def conn(tmp_path: Path):
    c = connect(tmp_path / "vocab.db")
    yield c
    c.close()


def arm(conn, word: str, task: str = T, role: str = "low"):
    return next(a for a in list_arms(conn, task, role) if a.word == word)


def test_update_moves_the_posterior(conn) -> None:
    add_arm(conn, "a duck", T, "low", "test")
    before = arm(conn, "a duck")
    update_arm(conn, "a duck", T, "low", 0.9)
    after = arm(conn, "a duck")

    assert after.alpha == pytest.approx(before.alpha + 0.9)
    assert after.beta == pytest.approx(before.beta + 0.1)
    assert after.n_trials == before.n_trials + 1
    assert after.mean > before.mean


def test_update_rejects_non_probability(conn) -> None:
    add_arm(conn, "a duck", T, "low", "test")
    with pytest.raises(ValueError):
        update_arm(conn, "a duck", T, "low", 1.5)


def test_update_unknown_arm_raises(conn) -> None:
    with pytest.raises(KeyError):
        update_arm(conn, "not seeded", T, "low", 0.5)


def test_roles_are_independent_arms(conn) -> None:
    """Keying on (word, task, role): evidence must not leak between roles."""
    add_arm(conn, "albert einstein", T, "low", "test")
    add_arm(conn, "albert einstein", T, "high", "test")
    update_arm(conn, "albert einstein", T, "low", 1.0)

    assert arm(conn, "albert einstein", T, "low").n_trials == 1
    assert arm(conn, "albert einstein", T, "high").n_trials == 0


def test_tasks_are_independent_arms(conn) -> None:
    """A word that works in a flip has said nothing about working in a hybrid."""
    add_arm(conn, "a duck", "flip", "subject", "test")
    add_arm(conn, "a duck", "jigsaw", "subject", "test")
    update_arm(conn, "a duck", "flip", "subject", 1.0)

    assert arm(conn, "a duck", "flip", "subject").n_trials == 1
    assert arm(conn, "a duck", "jigsaw", "subject").n_trials == 0
    assert list_tasks(conn) == ["flip", "jigsaw"]


def test_add_arm_never_overwrites(conn) -> None:
    add_arm(conn, "a duck", T, "low", "author", prior=(3.0, 1.0))
    update_arm(conn, "a duck", T, "low", 1.0)
    assert add_arm(conn, "a duck", T, "low", "mined", prior=(1.0, 1.0)) is False

    a = arm(conn, "a duck")
    assert a.alpha == pytest.approx(4.0)
    assert a.source == "author"
    assert a.n_trials == 1


def test_citation_is_stored(conn) -> None:
    add_arm(conn, "a duck", "flip", "subject", "paper", citation="VA Fig. 1")
    assert arm(conn, "a duck", "flip", "subject").citation == "VA Fig. 1"


def test_list_arms_filters_by_task_and_role(conn) -> None:
    seed_author_vocab(conn)
    everything = list_arms(conn)
    hybrid = list_arms(conn, T)
    hybrid_low = list_arms(conn, T, "low")
    assert len(everything) > len(hybrid) > len(hybrid_low) > 0
    assert all(a.task == T for a in hybrid)
    assert all(a.role == "low" for a in hybrid_low)
    assert list_arms(conn, role="style")


def test_untried_arms_shrink_as_they_are_tried(conn) -> None:
    seed_author_vocab(conn)
    before = untried_arms(conn, T, "low")
    assert before, "a freshly seeded database has untried arms"
    update_arm(conn, before[0].word, T, "low", 0.5)
    after = {a.word for a in untried_arms(conn, T, "low")}
    assert before[0].word not in after
    assert len(after) == len(before) - 1


def test_thompson_sampling_is_reproducible(conn) -> None:
    seed_author_vocab(conn)
    a = thompson_sample(conn, T, "low", np.random.default_rng(SEED))
    b = thompson_sample(conn, T, "low", np.random.default_rng(SEED))
    assert a == b


def test_thompson_sampling_favours_the_better_arm(conn) -> None:
    add_arm(conn, "good", T, "low", "test", prior=(50.0, 1.0))
    add_arm(conn, "bad", T, "low", "test", prior=(1.0, 50.0))
    rng = np.random.default_rng(SEED)
    picks = [thompson_sample(conn, T, "low", rng).word for _ in range(200)]
    assert picks.count("good") > 190


def test_thompson_sampling_respects_exclusions(conn) -> None:
    add_arm(conn, "good", T, "low", "test", prior=(50.0, 1.0))
    add_arm(conn, "bad", T, "low", "test", prior=(1.0, 50.0))
    rng = np.random.default_rng(SEED)
    picked = thompson_sample(conn, T, "low", rng, exclude=frozenset({"good"}))
    assert picked.word == "bad"


def test_thompson_sampling_stays_inside_the_task(conn) -> None:
    add_arm(conn, "flip-only", "flip", "subject", "test", prior=(50.0, 1.0))
    add_arm(conn, "jigsaw-only", "jigsaw", "subject", "test", prior=(1.0, 50.0))
    rng = np.random.default_rng(SEED)
    assert thompson_sample(conn, "jigsaw", "subject", rng).word == "jigsaw-only"


def test_thompson_sampling_with_no_arms_raises(conn) -> None:
    with pytest.raises(LookupError):
        thompson_sample(conn, T, "low", np.random.default_rng(SEED))


def test_credible_interval_brackets_the_mean(conn) -> None:
    add_arm(conn, "a duck", T, "low", "test", prior=(8.0, 2.0))
    lo, hi = arm(conn, "a duck").credible_interval(np.random.default_rng(SEED))
    assert 0.0 <= lo < arm(conn, "a duck").mean < hi <= 1.0


# -- the pooled vocabulary: every task can draw every word ------------------


def test_pooled_arms_offer_other_tasks_words_at_the_prior(tmp_path: Path) -> None:
    from ava.vocab import POOLED_SOURCE, pooled_arms

    conn = connect(tmp_path / "vocab.db")
    add_arm(conn, "a lighthouse", "negate", "subject", "paper", (3.0, 1.0))
    add_arm(conn, "a horse", "flip", "subject", "author")
    add_arm(conn, "a pop art of", "inner_circle", "style", "paper")
    add_arm(conn, "", "flip", "style", "author")

    arms = {a.word: a for a in pooled_arms(conn, "flip", "subject")}
    assert arms["a horse"].source == "author"
    lighthouse = arms["a lighthouse"]
    assert lighthouse.source == POOLED_SOURCE and lighthouse.task == "flip"
    assert (lighthouse.alpha, lighthouse.beta) == (1.0, 1.0), "evidence is per task"
    assert "pooled from negate:subject" in lighthouse.citation
    assert "a pop art of" not in arms, "styles do not pool into subjects"
    styles = {a.word for a in pooled_arms(conn, "flip", "style")}
    assert styles == {"", "a pop art of"}
    conn.close()


def test_draw_arm_registers_what_it_draws(tmp_path: Path) -> None:
    from ava.vocab import draw_arm

    conn = connect(tmp_path / "vocab.db")
    add_arm(conn, "a lighthouse", "negate", "subject", "paper")
    arm = draw_arm(conn, "flip", "subject", np.random.default_rng(0))
    assert arm.word == "a lighthouse" and arm.task == "flip"
    registered = list_arms(conn, "flip", "subject")
    assert [a.word for a in registered] == ["a lighthouse"]
    assert registered[0].n_trials == 0
    update_arm(conn, "a lighthouse", "flip", "subject", 1.0)  # credit can land
    # The negate arm is untouched.
    assert list_arms(conn, "negate", "subject")[0].n_trials == 0
    conn.close()


def test_untried_pool_counts_other_tasks_words_as_untried(tmp_path: Path) -> None:
    from ava.vocab import untried_arms, untried_pool

    conn = connect(tmp_path / "vocab.db")
    add_arm(conn, "a horse", "flip", "subject", "author")
    update_arm(conn, "a horse", "flip", "subject", 0.5)
    add_arm(conn, "a lighthouse", "negate", "subject", "paper")
    update_arm(conn, "a lighthouse", "negate", "subject", 0.5)
    assert untried_arms(conn, "flip", "subject") == []
    assert [a.word for a in untried_pool(conn, "flip", "subject")] == ["a lighthouse"]
    conn.close()
