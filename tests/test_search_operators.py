"""Each operator changes exactly what it says it changes; the bandit explores first."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ava.propose import CandidateBuilder
from ava.search.archive import Archive, Clusterer
from ava.search.embed import CachingEmbedder
from ava.search.operators import (
    OPERATOR_NAMES,
    OPERATORS,
    TRANSFER_SOURCE,
    Child,
    OperatorBandit,
    OperatorContext,
    RunningStats,
    crossover,
    improvement_reward,
    inject,
    restyle,
    swap_slot,
    transfer_task,
    transpose,
)
from ava.spec import CandidateSpec
from ava.vocab import connect, list_arms, seed_author_vocab, update_arm
from tests.search_fakes import CATEGORIES, FakeEmbedder, verdict_for

SEED = 0
ALL_WORDS = [w for words in CATEGORIES.values() for w in words]


@pytest.fixture()
def ctx(tmp_path: Path):
    conn = connect(tmp_path / "vocab.db")
    seed_author_vocab(conn)
    embed = CachingEmbedder(FakeEmbedder())
    archive = Archive(Clusterer.fit(ALL_WORDS, embed, k=4, seed=SEED), embed)
    builder = CandidateBuilder(
        ["flip", "jigsaw", "hybrid", "three_view"], 0, 10.0, 30, None, None
    )
    yield OperatorContext(conn, np.random.default_rng(SEED), builder, archive)
    conn.close()


def parent_of(ctx: OperatorContext, spec: CandidateSpec, *p: float, sep=()):
    return ctx.archive.add(spec, verdict_for(spec, *p, sep=sep), 0, "test")


def flip(a: str, b: str, style: str = "an oil painting of") -> CandidateSpec:
    return CandidateSpec("flip", (a, b), style)


def must(result: Child | None) -> Child:
    assert result is not None, "the operator was expected to apply"
    return result


# -- swap_slot ------------------------------------------------------------


def test_swap_slot_redraws_the_lost_slot_only(ctx) -> None:
    parent = parent_of(ctx, flip("a horse", "a duck"), 0.9, 0.1, sep=[0.1, -0.1])
    child, detail = must(swap_slot(parent, ctx))
    assert child.prompts[0] == "a horse"
    assert child.prompts[1] not in ("a duck", "a horse")
    assert child.style == parent.spec.style
    assert child.task == "flip"
    assert detail.startswith("swap_slot flip")


def test_swap_slot_targets_the_weakest_slot_when_everything_held(ctx) -> None:
    parent = parent_of(ctx, flip("a horse", "a duck"), 0.9, 0.9, sep=[0.02, 0.08])
    child, _ = must(swap_slot(parent, ctx))
    assert child.prompts[1] == "a duck"
    assert child.prompts[0] != "a horse"


def test_swap_slot_uses_the_screening_settings(ctx) -> None:
    parent = parent_of(ctx, flip("a horse", "a duck", "a photo of"), 0.9, 0.1)
    child, _ = must(swap_slot(parent, ctx))
    assert child.seed == 0 and child.num_inference_steps == 30


# -- crossover ------------------------------------------------------------


def test_crossover_needs_a_partner_in_the_same_task(ctx) -> None:
    parent = parent_of(ctx, flip("a horse", "a duck"), 0.9, 0.9)
    assert crossover(parent, ctx) is None
    parent_of(ctx, CandidateSpec("jigsaw", ("a skull", "a landscape"), ""), 0.9, 0.9)
    assert crossover(parent, ctx) is None, "a jigsaw is not a flip partner"


def test_crossover_takes_one_slot_from_each_parent(ctx) -> None:
    parent = parent_of(ctx, flip("a horse", "a duck"), 0.9, 0.9)
    partner = parent_of(ctx, flip("a skull", "a landscape"), 0.9, 0.9)
    child, detail = must(crossover(parent, ctx))
    assert set(child.prompts) & set(parent.spec.prompts)
    assert set(child.prompts) & set(partner.spec.prompts)
    assert len(set(child.prompts)) == 2
    assert child.style == parent.spec.style
    assert "crossover" in detail


# -- restyle --------------------------------------------------------------


def test_restyle_changes_only_the_style(ctx) -> None:
    parent = parent_of(ctx, flip("a horse", "a duck"), 0.9, 0.9)
    child, _ = must(restyle(parent, ctx))
    assert child.prompts == parent.spec.prompts
    assert child.style != parent.spec.style
    assert child.style in {a.word for a in list_arms(ctx.conn, "flip", "style")}


# -- transpose ------------------------------------------------------------


def test_transpose_swaps_two_slots(ctx) -> None:
    parent = parent_of(
        ctx, CandidateSpec("hybrid", ("a panda", "houseplants")), 0.9, 0.9
    )
    child, _ = must(transpose(parent, ctx))
    assert child.prompts == ("houseplants", "a panda")
    assert child.task == "hybrid"


def test_transpose_never_returns_the_identity_for_three_slots(ctx) -> None:
    spec = CandidateSpec("three_view", ("a horse", "a duck", "a skull"), "")
    parent = parent_of(ctx, spec, 0.9, 0.9, 0.9)
    for _ in range(10):
        child, _ = must(transpose(parent, ctx))
        assert sorted(child.prompts) == sorted(spec.prompts)
        assert child.prompts != spec.prompts


def test_transpose_refuses_a_reference_task(ctx) -> None:
    spec = CandidateSpec(
        "inverse_hybrid", ("albert einstein", "waterfalls"), "", ref_image="x.png"
    )
    parent = parent_of(ctx, spec, 0.9, 0.9)
    assert transpose(parent, ctx) is None


# -- transfer_task --------------------------------------------------------


def test_transfer_keeps_the_pair_and_changes_the_task(ctx) -> None:
    parent = parent_of(ctx, flip("a horse", "a duck"), 0.9, 0.9)
    child, detail = must(transfer_task(parent, ctx))
    assert child.prompts == parent.spec.prompts
    assert child.style == parent.spec.style
    assert child.task in ("jigsaw", "hybrid")
    assert child.task != "flip"
    assert "transfer_task flip->" in detail


def test_transfer_registers_missing_arms_so_credit_can_land(ctx) -> None:
    # `a panda` is a hybrid word; no VA task knows it before the transfer.
    assert not [a for a in list_arms(ctx.conn, "flip") if a.word == "a panda"]
    ctx.builder = CandidateBuilder(["hybrid", "flip"], 0, 10.0, 30, None, None)
    parent = parent_of(
        ctx, CandidateSpec("hybrid", ("a panda", "houseplants")), 0.9, 0.9
    )
    child, _ = must(transfer_task(parent, ctx))
    assert child.task == "flip"
    arms = {(a.word, a.role): a for a in list_arms(ctx.conn, "flip")}
    assert ("a panda", "subject") in arms
    assert arms[("a panda", "subject")].source == TRANSFER_SOURCE
    assert "transferred from hybrid" in arms[("a panda", "subject")].citation
    # An existing arm keeps its posterior and provenance.
    plants = arms[("houseplants", "subject")]
    assert plants.source == "author"


def test_transfer_only_targets_tasks_with_the_same_view_count(ctx) -> None:
    spec = CandidateSpec("three_view", ("a horse", "a duck", "a skull"), "")
    parent = parent_of(ctx, spec, 0.9, 0.9, 0.9)
    assert transfer_task(parent, ctx) is None


# -- inject ---------------------------------------------------------------


def test_inject_puts_an_untried_word_into_one_slot(ctx) -> None:
    parent = parent_of(ctx, flip("a horse", "a duck"), 0.9, 0.9)
    child, detail = must(inject(parent, ctx))
    changed = [i for i in range(2) if child.prompts[i] != parent.spec.prompts[i]]
    assert len(changed) == 1
    new = child.prompts[changed[0]]
    arm = next(a for a in list_arms(ctx.conn, "flip", "subject") if a.word == new)
    assert arm.n_trials == 0
    assert "inject" in detail


def test_inject_returns_none_when_every_word_was_tried(ctx) -> None:
    for a in list_arms(ctx.conn, "flip", "subject"):
        update_arm(ctx.conn, a.word, a.task, a.role, 0.5)
    parent = parent_of(ctx, flip("a horse", "a duck"), 0.9, 0.9)
    assert inject(parent, ctx) is None


# -- every operator, generically ------------------------------------------


@pytest.mark.parametrize("name", OPERATOR_NAMES)
def test_every_operator_yields_a_distinct_valid_child_or_none(ctx, name) -> None:
    parent = parent_of(ctx, flip("a horse", "a duck"), 0.9, 0.1, sep=[0.1, -0.1])
    parent_of(ctx, flip("a skull", "a landscape"), 0.9, 0.9)
    result = OPERATORS[name](parent, ctx)
    if result is None:
        return
    child, detail = result
    assert child.uid() != parent.spec.uid()
    assert len(set(child.prompts)) == len(child.prompts)
    assert detail


def test_operators_are_reproducible_from_the_rng(tmp_path: Path) -> None:
    def run() -> list[str]:
        conn = connect(tmp_path / f"v{len(list(tmp_path.iterdir()))}.db")
        seed_author_vocab(conn)
        embed = CachingEmbedder(FakeEmbedder())
        archive = Archive(Clusterer.fit(ALL_WORDS, embed, k=4, seed=SEED), embed)
        builder = CandidateBuilder(["flip", "jigsaw"], 0, 10.0, 30, None, None)
        ctx = OperatorContext(conn, np.random.default_rng(7), builder, archive)
        parent = parent_of(ctx, flip("a horse", "a duck"), 0.9, 0.1)
        parent_of(ctx, flip("a skull", "a landscape"), 0.9, 0.9)
        out = []
        for name in OPERATOR_NAMES:
            r = OPERATORS[name](parent, ctx)
            out.append(r[0].uid() if r else "none")
        conn.close()
        return out

    assert run() == run()


# -- reward and bandit ----------------------------------------------------


def test_improvement_reward_is_zero_for_no_gain_and_bounded() -> None:
    assert improvement_reward(0.05, 0.10, scale=0.05) == 0.0
    assert improvement_reward(0.10, 0.10, scale=0.05) == 0.0
    small = improvement_reward(0.11, 0.10, scale=0.05)
    large = improvement_reward(0.40, 0.10, scale=0.05)
    assert 0.0 < small < large < 1.0


def test_improvement_reward_floor_tames_an_empty_running_std() -> None:
    """Before the fitness spread is known, a tiny gain must not look enormous."""
    assert improvement_reward(0.101, 0.100, scale=0.0) < 0.1


def test_running_stats_match_numpy() -> None:
    xs = [0.01, 0.05, -0.02, 0.10, 0.03]
    stats = RunningStats()
    for x in xs:
        stats.update(x)
    assert stats.mean == pytest.approx(np.mean(xs))
    assert stats.std == pytest.approx(np.std(xs, ddof=1))
    assert RunningStats.from_dict(stats.to_dict()).std == pytest.approx(stats.std)


def test_bandit_tries_every_operator_before_exploiting() -> None:
    bandit = OperatorBandit()
    rng = np.random.default_rng(0)
    drawn = set()
    for _ in range(len(OPERATOR_NAMES)):
        name = bandit.choose(rng)
        drawn.add(name)
        bandit.update(name, 0.0)
    assert drawn == set(OPERATOR_NAMES)


def test_bandit_prefers_the_rewarding_operator_but_keeps_exploring() -> None:
    bandit = OperatorBandit(c=1.0)
    rng = np.random.default_rng(0)
    for name in OPERATOR_NAMES:
        bandit.update(name, 1.0 if name == "crossover" else 0.0)
    picks = []
    for _ in range(60):
        name = bandit.choose(rng)
        picks.append(name)
        bandit.update(name, 1.0 if name == "crossover" else 0.0)
    assert picks.count("crossover") > 30
    assert len(set(picks)) > 1, "UCB1 must keep visiting the others"


def test_bandit_round_trips_and_rejects_unknown_operators() -> None:
    bandit = OperatorBandit()
    bandit.update("restyle", 0.5)
    back = OperatorBandit.from_dict(bandit.to_dict())
    assert back.counts == bandit.counts and back.totals == bandit.totals
    with pytest.raises(KeyError):
        bandit.update("teleport", 1.0)
