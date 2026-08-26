"""Proposal mix, targeted swaps, and per-component credit assignment."""

from pathlib import Path

import numpy as np
import pytest

from ava.propose import (
    EXPLOIT,
    INJECT,
    SWAP,
    BanditProposer,
    ProposalMix,
    assign_credit,
    captions_collide,
    diagnose,
    spec_arms,
)
from ava.spec import CandidateSpec, RunState, Verdict
from ava.vocab import add_arm, connect, list_arms, seed_author_vocab

SEED = 0


@pytest.fixture()
def conn(tmp_path: Path):
    c = connect(tmp_path / "vocab.db")
    seed_author_vocab(c)
    yield c
    c.close()


def make_proposer(conn, **kw) -> BanditProposer:
    return BanditProposer(conn, np.random.default_rng(SEED), **kw)


def verdict_for(spec: CandidateSpec, p_far: float, p_near: float, **kw) -> Verdict:
    return Verdict(
        uid=spec.uid(),
        s_far_low=0.3,
        s_far_high=0.2,
        s_near_low=0.2,
        s_near_high=0.3,
        p_far=p_far,
        p_near=p_near,
        j=min(p_far, p_near),
        **kw,
    )


# -- mix ----------------------------------------------------------------


@pytest.mark.parametrize("k", [4, 8, 12, 20])
def test_counts_sum_to_k(k: int) -> None:
    assert sum(ProposalMix().counts(k)) == k


def test_default_mix_is_half_exploit() -> None:
    assert ProposalMix().counts(20) == (10, 5, 5)


def test_lopsided_mix_does_not_produce_negative_exploit() -> None:
    n_exploit, n_inject, n_swap = ProposalMix(inject=0.8, swap=0.8).counts(10)
    assert n_exploit >= 0
    assert n_exploit + n_inject + n_swap == 10


def test_propose_rejects_non_positive_k(conn) -> None:
    with pytest.raises(ValueError):
        make_proposer(conn).propose(RunState(run_id="r"), 0)


# -- round composition --------------------------------------------------


def test_round_returns_k_unique_candidates(conn) -> None:
    proposals = make_proposer(conn).propose(RunState(run_id="r"), 12)
    assert len(proposals) == 12
    assert len({p.spec.uid() for p in proposals}) == 12


def test_first_round_guarantees_new_word_injection(conn) -> None:
    """The injection quota is what stops the bandit freezing on the seed set."""
    proposals = make_proposer(conn).propose(RunState(run_id="r"), 12)
    assert sum(p.origin == INJECT for p in proposals) == 3


def test_already_evaluated_candidates_are_not_reproposed(conn) -> None:
    state = RunState(run_id="r")
    first = make_proposer(conn).propose(state, 8)
    for p in first:
        state.evaluated_uids.add(p.spec.uid())

    second = make_proposer(conn).propose(state, 8)
    assert not {p.spec.uid() for p in second} & state.evaluated_uids


def test_low_and_high_components_differ(conn) -> None:
    """Pairing a word with itself would make the two views trivially identical."""
    for p in make_proposer(conn).propose(RunState(run_id="r"), 12):
        assert p.spec.prompt_low != p.spec.prompt_high


def test_proposals_are_reproducible_from_the_seed(conn) -> None:
    a = make_proposer(conn).propose(RunState(run_id="r"), 8)
    b = make_proposer(conn).propose(RunState(run_id="r"), 8)
    assert [p.spec.uid() for p in a] == [p.spec.uid() for p in b]


def test_every_proposal_records_its_origin(conn) -> None:
    for p in make_proposer(conn).propose(RunState(run_id="r"), 12):
        assert p.origin in (EXPLOIT, INJECT, SWAP)
        assert p.detail
        assert p.to_dict()["uid"] == p.spec.uid()


def test_exploit_backfills_when_nothing_is_injectable(conn) -> None:
    """With every arm tried, the injection quota is empty and exploit absorbs it."""
    from ava.vocab import update_arm

    for arm in list_arms(conn):
        update_arm(conn, arm.word, arm.role, 0.5)
    proposals = make_proposer(conn).propose(RunState(run_id="r"), 8)
    assert len(proposals) == 8
    assert all(p.origin == EXPLOIT for p in proposals)


# -- targeted swap ------------------------------------------------------


def test_swap_keeps_the_working_component(conn) -> None:
    """A low-side failure must redraw low only, keeping high and style."""
    spec = CandidateSpec("a duck", "houseplants", "an oil painting of")
    state = RunState(run_id="r", last_round=[(spec, verdict_for(spec, 0.05, 0.95))])

    swaps = [p for p in make_proposer(conn).propose(state, 12) if p.origin == SWAP]
    assert swaps, "a repairable failure must generate a swap"
    for p in swaps:
        assert p.spec.prompt_high == spec.prompt_high
        assert p.spec.style == spec.style
        assert p.spec.prompt_low != spec.prompt_low


def test_swap_redraws_the_high_side_when_it_is_absent(conn) -> None:
    spec = CandidateSpec("a duck", "houseplants", "an oil painting of")
    state = RunState(run_id="r", last_round=[(spec, verdict_for(spec, 0.95, 0.05))])

    swaps = [p for p in make_proposer(conn).propose(state, 12) if p.origin == SWAP]
    assert swaps
    for p in swaps:
        assert p.spec.prompt_low == spec.prompt_low
        assert p.spec.prompt_high != spec.prompt_high


def test_pair_mismatch_is_discarded_not_repaired(conn) -> None:
    """When both views fail the triple is abandoned, not partially redrawn."""
    spec = CandidateSpec("a duck", "houseplants", "an oil painting of")
    state = RunState(run_id="r", last_round=[(spec, verdict_for(spec, 0.05, 0.05))])

    swaps = [p for p in make_proposer(conn).propose(state, 12) if p.origin == SWAP]
    assert swaps == []


def test_successful_candidates_are_not_repaired(conn) -> None:
    spec = CandidateSpec("a duck", "houseplants", "an oil painting of")
    state = RunState(run_id="r", last_round=[(spec, verdict_for(spec, 0.95, 0.95))])

    swaps = [p for p in make_proposer(conn).propose(state, 12) if p.origin == SWAP]
    assert swaps == []


def test_collapsed_views_are_detected_and_repaired(conn) -> None:
    """Both views scoring well is not enough if they show the same thing."""
    spec = CandidateSpec("a duck", "houseplants", "an oil painting of")
    v = verdict_for(
        spec,
        0.95,
        0.95,
        caption_far="there is a duck swimming in a pond",
        caption_near="there is a duck swimming in a pond",
    )
    assert v.diagnose() == "ok"
    assert diagnose(v) == "views_collapsed"

    swaps = [
        p
        for p in make_proposer(conn).propose(
            RunState(run_id="r", last_round=[(spec, v)]), 12
        )
        if p.origin == SWAP
    ]
    assert swaps
    assert all(p.spec.prompt_high != spec.prompt_high for p in swaps)


def test_distinct_captions_do_not_trigger_a_collapse(conn) -> None:
    spec = CandidateSpec("a duck", "houseplants", "")
    v = verdict_for(
        spec,
        0.95,
        0.95,
        caption_far="there is a duck swimming in a pond",
        caption_near="a room filled with many potted plants on shelves",
    )
    assert not captions_collide(v)
    assert diagnose(v) == "ok"


def test_collapse_detection_ignores_empty_captions(conn) -> None:
    """use_blip=False leaves captions empty; that must not read as a collapse."""
    spec = CandidateSpec("a duck", "houseplants", "")
    assert not captions_collide(verdict_for(spec, 0.9, 0.9))


def test_text_distance_steers_the_collapsed_repair(tmp_path: Path) -> None:
    """When a distance is supplied, the replacement is the far-away word."""
    conn = connect(tmp_path / "vocab.db")
    for word in ("near-word", "far-word"):
        add_arm(conn, word, "high", "test")
    add_arm(conn, "a duck", "low", "test")
    add_arm(conn, "", "style", "test")

    def distance(a: str, b: str) -> float:
        return 1.0 if b == "far-word" else 0.0

    spec = CandidateSpec("a duck", "houseplants", "")
    v = verdict_for(
        spec,
        0.95,
        0.95,
        caption_far="a duck in a pond",
        caption_near="a duck in a pond",
    )
    # Injection is disabled here: with only four arms the injection quota would
    # build the identical triple first and dedup would then drop the swap. That
    # is correct behaviour, but it hides what this test is about.
    proposer = BanditProposer(
        conn,
        np.random.default_rng(SEED),
        mix=ProposalMix(inject=0.0, swap=0.5),
        text_distance=distance,
    )
    swaps = [
        p
        for p in proposer.propose(RunState(run_id="r", last_round=[(spec, v)]), 8)
        if p.origin == SWAP
    ]
    assert swaps
    assert all(p.spec.prompt_high == "far-word" for p in swaps)


# -- credit assignment --------------------------------------------------


def test_spec_arms_names_all_three_components() -> None:
    spec = CandidateSpec("a panda", "waterfalls", "an oil painting of")
    assert spec_arms(spec) == (
        ("a panda", "low"),
        ("waterfalls", "high"),
        ("an oil painting of", "style"),
    )


def test_credit_routes_each_signal_to_its_own_component(conn) -> None:
    """p_far must not be allowed to reward the high word, or vice versa."""
    spec = CandidateSpec("a duck", "houseplants", "an oil painting of")
    assign_credit(conn, spec, verdict_for(spec, p_far=1.0, p_near=0.0))

    low = next(a for a in list_arms(conn, "low") if a.word == "a duck")
    high = next(a for a in list_arms(conn, "high") if a.word == "houseplants")
    style = next(a for a in list_arms(conn, "style") if a.word == "an oil painting of")

    assert low.alpha == pytest.approx(2.0) and low.beta == pytest.approx(1.0)
    assert high.alpha == pytest.approx(2.0) and high.beta == pytest.approx(2.0)
    # The style arm is judged on J, which is min(1.0, 0.0) = 0.0.
    assert style.beta == pytest.approx(2.0)
    assert all(a.n_trials == 1 for a in (low, high, style))


def test_a_good_low_word_is_not_punished_for_a_bad_partner(conn) -> None:
    """The reason credit is per component rather than per candidate."""
    good_low, bad_high = "a duck", "houseplants"
    for _ in range(5):
        spec = CandidateSpec(good_low, bad_high, "")
        assign_credit(conn, spec, verdict_for(spec, p_far=0.99, p_near=0.01))

    low = next(a for a in list_arms(conn, "low") if a.word == good_low)
    high = next(a for a in list_arms(conn, "high") if a.word == bad_high)
    # `houseplants` is seeded Beta(2, 1), so five bad trials pull it down but
    # cannot drive it to zero. The claim is the separation, not the absolute value.
    assert low.mean > 0.8
    assert high.mean < 0.3
    assert low.mean - high.mean > 0.5


# -- uniform validation sweeps ------------------------------------------


def test_uniform_proposer_balances_style_trials(conn) -> None:
    """The point of the uniform sweep: every style gets comparable evidence.

    Thompson sampling correctly starves a low-prior arm, which leaves it with
    a posterior that is still just its prior. A validation ranking has to be
    built on measurements instead.
    """
    from collections import Counter

    from ava.propose import UNIFORM, UniformProposer
    from ava.vocab import list_arms

    n_styles = len(list_arms(conn, "style"))
    proposer = UniformProposer(conn, np.random.default_rng(SEED))
    proposals = proposer.propose(RunState(run_id="r"), n_styles * 3)

    counts = Counter(p.spec.style for p in proposals)
    assert len(counts) == n_styles, "every style must be exercised"
    assert max(counts.values()) - min(counts.values()) <= 1
    assert all(p.origin == UNIFORM for p in proposals)


def test_uniform_proposer_reaches_the_arms_the_bandit_starves(conn) -> None:
    """`a photo of` is seeded Beta(1, 3); Thompson sampling avoids it by design."""
    from ava.propose import UniformProposer

    bandit = [
        p.spec.style for p in make_proposer(conn).propose(RunState(run_id="r"), 24)
    ]
    uniform = [
        p.spec.style
        for p in UniformProposer(conn, np.random.default_rng(SEED)).propose(
            RunState(run_id="r"), 24
        )
    ]
    assert bandit.count("a photo of") < uniform.count("a photo of")


def test_uniform_proposer_respects_evaluated_uids(conn) -> None:
    from ava.propose import UniformProposer

    state = RunState(run_id="r")
    proposer = UniformProposer(conn, np.random.default_rng(SEED))
    for p in proposer.propose(state, 8):
        state.evaluated_uids.add(p.spec.uid())

    again = UniformProposer(conn, np.random.default_rng(SEED)).propose(state, 8)
    assert not {p.spec.uid() for p in again} & state.evaluated_uids


def test_uniform_proposer_keeps_the_two_views_distinct(conn) -> None:
    from ava.propose import UniformProposer

    proposer = UniformProposer(conn, np.random.default_rng(SEED))
    for p in proposer.propose(RunState(run_id="r"), 16):
        assert p.spec.prompt_low != p.spec.prompt_high
