"""Proposal mix, targeted swaps, and per-component credit assignment."""

from pathlib import Path

import numpy as np
import pytest

from ava.image.tasks import get_task
from ava.propose import (
    EXPLOIT,
    INJECT,
    SWAP,
    UNIFORM,
    BanditProposer,
    ProposalMix,
    UniformProposer,
    assign_credit,
    captions_collide,
    diagnose,
    spec_arms,
)
from ava.spec import CandidateSpec, RunState, Verdict
from ava.vocab import add_arm, connect, list_arms, seed_author_vocab, update_arm

SEED = 0


@pytest.fixture()
def conn(tmp_path: Path):
    c = connect(tmp_path / "vocab.db")
    seed_author_vocab(c)
    yield c
    c.close()


def make_proposer(conn, tasks=("hybrid",), **kw) -> BanditProposer:
    return BanditProposer(conn, np.random.default_rng(SEED), tasks=tasks, **kw)


def hybrid(low: str, high: str, style: str = "an oil painting of") -> CandidateSpec:
    return CandidateSpec("hybrid", (low, high), style)


def verdict_for(spec: CandidateSpec, *p: float, captions=()) -> Verdict:
    """A verdict whose margins agree with the requested per-view probabilities."""
    task = get_task(spec.task)
    n = task.n_views
    scores = [[0.2] * n for _ in range(n)]
    for i, pi in enumerate(p):
        scores[i][i] = 0.3 if pi > 0.5 else 0.1
    return Verdict(
        uid=spec.uid(),
        task=spec.task,
        slots=task.slot_names,
        scores=scores,
        p=list(p),
        j=min(p),
        alignment=min(s[i] for i, s in enumerate(scores)),
        concealment=0.5,
        captions=list(captions),
    )


def arm(conn, word, task, role):
    return next(a for a in list_arms(conn, task, role) if a.word == word)


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


def test_proposer_needs_a_task(conn) -> None:
    with pytest.raises(ValueError):
        make_proposer(conn, tasks=())


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


def test_slots_never_share_a_word(conn) -> None:
    """Pairing a word with itself would make the views trivially identical."""
    for p in make_proposer(conn, tasks=("hybrid", "flip", "three_view")).propose(
        RunState(run_id="r"), 18
    ):
        assert len(set(p.spec.prompts)) == len(p.spec.prompts), p.spec


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
    """With every arm tried, the injection quota is empty and exploit absorbs it.

    Only hybrid arms are left in the database: the vocabulary is pooled, so a
    word another task knows would still count as injectable for hybrid.
    """
    conn.execute("DELETE FROM arm WHERE task != 'hybrid'")  # nothing to pool from
    for a in list_arms(conn):
        update_arm(conn, a.word, a.task, a.role, 0.5)
    proposals = make_proposer(conn).propose(RunState(run_id="r"), 8)
    assert len(proposals) == 8
    assert all(p.origin == EXPLOIT for p in proposals)


# -- tasks --------------------------------------------------------------


def test_tasks_are_visited_round_robin(conn) -> None:
    proposals = make_proposer(conn, tasks=("flip", "hybrid", "jigsaw")).propose(
        RunState(run_id="r"), 12
    )
    from collections import Counter

    counts = Counter(p.spec.task for p in proposals)
    assert set(counts) == {"flip", "hybrid", "jigsaw"}
    assert max(counts.values()) - min(counts.values()) <= 2  # injection is random


def test_va_candidates_fill_every_slot_from_the_subject_pool(conn) -> None:
    for p in make_proposer(conn, tasks=("four_view",)).propose(RunState(run_id="r"), 4):
        assert p.spec.task == "four_view"
        assert p.spec.n_views == 4
        assert p.spec.ref_image is None


def test_proposed_words_are_arms_of_their_task(conn) -> None:
    """Whatever a flip proposes is registered under flip by the time it is proposed.

    The vocabulary is pooled across tasks, so a flip may draw a word only
    another task knew; credit assignment then needs a (word, flip, role) arm.
    """
    for p in make_proposer(conn, tasks=("flip",)).propose(RunState(run_id="r"), 8):
        words = {a.word for a in list_arms(conn, "flip", "subject")}
        assert set(p.spec.prompts) <= words


def test_a_word_only_another_task_knows_reaches_the_flip(conn) -> None:
    """The reason for pooling: paper and generated words were stuck in one task."""
    add_arm(conn, "a lighthouse keeper", "negate", "subject", "paper")
    seen = set()
    for _ in range(6):
        for p in make_proposer(conn, tasks=("flip",)).propose(RunState(run_id="r"), 12):
            seen |= set(p.spec.prompts)
    assert "a lighthouse keeper" in seen
    arm = next(
        a for a in list_arms(conn, "flip", "subject") if a.word == "a lighthouse keeper"
    )
    assert arm.source == "pooled" and arm.n_trials == 0


def test_reference_task_pins_the_reference_slot(conn) -> None:
    proposer = make_proposer(
        conn,
        tasks=("inverse_hybrid",),
        ref_image="assets/einstein.png",
        ref_prompt="albert einstein",
    )
    for p in proposer.propose(RunState(run_id="r"), 4):
        assert p.spec.prompts[0] == "albert einstein"
        assert p.spec.ref_image == "assets/einstein.png"
        assert p.spec.prompts[1] != "albert einstein"


def test_reference_task_without_a_reference_is_refused(conn) -> None:
    with pytest.raises(ValueError):
        make_proposer(conn, tasks=("inverse_hybrid",))


# -- targeted swap ------------------------------------------------------


def swaps_for(conn, spec, verdict, tasks=("hybrid",)):
    state = RunState(run_id="r", last_round=[(spec, verdict)])
    return [
        p
        for p in make_proposer(conn, tasks=tasks).propose(state, 12)
        if p.origin == SWAP
    ]


def test_swap_keeps_the_working_component(conn) -> None:
    """A low-side failure must redraw low only, keeping high and style."""
    spec = hybrid("a duck", "houseplants")
    swaps = swaps_for(conn, spec, verdict_for(spec, 0.05, 0.95))
    assert swaps, "a repairable failure must generate a swap"
    for p in swaps:
        assert p.spec.prompts[1] == "houseplants"
        assert p.spec.style == spec.style
        assert p.spec.prompts[0] != "a duck"


def test_swap_redraws_the_high_side_when_it_is_absent(conn) -> None:
    spec = hybrid("a duck", "houseplants")
    swaps = swaps_for(conn, spec, verdict_for(spec, 0.95, 0.05))
    assert swaps
    for p in swaps:
        assert p.spec.prompts[0] == "a duck"
        assert p.spec.prompts[1] != "houseplants"


def test_swap_redraws_only_the_lost_slots_of_a_three_view(conn) -> None:
    spec = CandidateSpec("three_view", ("a duck", "a rabbit", "a horse"), "")
    swaps = swaps_for(
        conn, spec, verdict_for(spec, 0.9, 0.1, 0.9), tasks=("three_view",)
    )
    assert swaps
    for p in swaps:
        assert p.spec.prompts[0] == "a duck" and p.spec.prompts[2] == "a horse"
        assert p.spec.prompts[1] not in ("a rabbit", "a duck", "a horse")


def test_all_lost_is_discarded_not_repaired(conn) -> None:
    """When every view fails the candidate is abandoned, not partially redrawn."""
    spec = hybrid("a duck", "houseplants")
    assert swaps_for(conn, spec, verdict_for(spec, 0.05, 0.05)) == []


def test_successful_candidates_are_not_repaired(conn) -> None:
    spec = hybrid("a duck", "houseplants")
    assert swaps_for(conn, spec, verdict_for(spec, 0.95, 0.95)) == []


def test_collapsed_views_are_detected_and_repaired(conn) -> None:
    """Both views scoring well is not enough if they show the same thing."""
    spec = hybrid("a duck", "houseplants")
    v = verdict_for(
        spec,
        0.95,
        0.95,
        captions=(
            "there is a duck swimming in a pond",
            "there is a duck swimming in a pond",
        ),
    )
    assert v.diagnose() == "ok"
    assert diagnose(v) == "views_collapsed"

    swaps = swaps_for(conn, spec, v)
    assert swaps
    assert all(p.spec.prompts[0] == "a duck" for p in swaps)
    assert all(p.spec.prompts[1] != "houseplants" for p in swaps)


def test_distinct_captions_do_not_trigger_a_collapse(conn) -> None:
    spec = hybrid("a duck", "houseplants", "")
    v = verdict_for(
        spec,
        0.95,
        0.95,
        captions=(
            "there is a duck swimming in a pond",
            "a room filled with many potted plants on shelves",
        ),
    )
    assert not captions_collide(v)
    assert diagnose(v) == "ok"


def test_collapse_detection_ignores_empty_captions(conn) -> None:
    """use_blip=False leaves captions empty; that must not read as a collapse."""
    spec = hybrid("a duck", "houseplants", "")
    assert not captions_collide(verdict_for(spec, 0.9, 0.9))
    assert not captions_collide(verdict_for(spec, 0.9, 0.9, captions=("", "")))


def test_text_distance_steers_the_collapsed_repair(tmp_path: Path) -> None:
    """When a distance is supplied, the replacement is the far-away word."""
    conn = connect(tmp_path / "vocab.db")
    for word in ("near-word", "far-word"):
        add_arm(conn, word, "hybrid", "high", "test")
    add_arm(conn, "a duck", "hybrid", "low", "test")
    add_arm(conn, "", "hybrid", "style", "test")

    def distance(a: str, b: str) -> float:
        return 1.0 if b == "far-word" else 0.0

    spec = hybrid("a duck", "houseplants", "")
    v = verdict_for(spec, 0.95, 0.95, captions=("a duck in a pond", "a duck in a pond"))
    # Injection is disabled here: with only four arms the injection quota would
    # build the identical candidate first and dedup would then drop the swap.
    proposer = BanditProposer(
        conn,
        np.random.default_rng(SEED),
        tasks=("hybrid",),
        mix=ProposalMix(inject=0.0, swap=0.5),
        text_distance=distance,
    )
    swaps = [
        p
        for p in proposer.propose(RunState(run_id="r", last_round=[(spec, v)]), 8)
        if p.origin == SWAP
    ]
    assert swaps
    assert all(p.spec.prompts[1] == "far-word" for p in swaps)


# -- credit assignment --------------------------------------------------


def test_spec_arms_names_every_searched_slot_and_the_style() -> None:
    assert spec_arms(hybrid("a panda", "waterfalls")) == (
        ("a panda", "hybrid", "low"),
        ("waterfalls", "hybrid", "high"),
        ("an oil painting of", "hybrid", "style"),
    )
    flip = CandidateSpec("flip", ("a duck", "a rabbit"), "a lithograph of")
    assert spec_arms(flip) == (
        ("a duck", "flip", "subject"),
        ("a rabbit", "flip", "subject"),
        ("a lithograph of", "flip", "style"),
    )
    inverse = CandidateSpec(
        "inverse_hybrid", ("albert einstein", "waterfalls"), "", ref_image="x.png"
    )
    assert spec_arms(inverse) == (
        ("waterfalls", "inverse_hybrid", "high"),
        ("", "inverse_hybrid", "style"),
    )


def test_credit_routes_each_signal_to_its_own_component(conn) -> None:
    """p_far must not be allowed to reward the high word, or vice versa."""
    spec = hybrid("a duck", "houseplants")
    assign_credit(conn, spec, verdict_for(spec, 1.0, 0.0))

    low = arm(conn, "a duck", "hybrid", "low")
    high = arm(conn, "houseplants", "hybrid", "high")
    style = arm(conn, "an oil painting of", "hybrid", "style")

    assert low.alpha == pytest.approx(2.0) and low.beta == pytest.approx(1.0)
    assert high.alpha == pytest.approx(2.0) and high.beta == pytest.approx(2.0)
    # The style arm is judged on J, which is min(1.0, 0.0) = 0.0.
    assert style.beta == pytest.approx(2.0)
    assert all(a.n_trials == 1 for a in (low, high, style))


def test_credit_for_a_flip_lands_on_the_shared_subject_role(conn) -> None:
    spec = CandidateSpec("flip", ("a duck", "a rabbit"), "a lithograph of")
    assign_credit(conn, spec, verdict_for(spec, 1.0, 0.0))
    assert arm(conn, "a duck", "flip", "subject").alpha == pytest.approx(2.0)
    assert arm(conn, "a rabbit", "flip", "subject").beta == pytest.approx(2.0)
    # Nothing leaked into another task.
    assert arm(conn, "a duck", "jigsaw", "subject").n_trials == 0


def test_credit_skips_the_reference_slot(conn) -> None:
    spec = CandidateSpec(
        "inverse_hybrid",
        ("albert einstein", "waterfalls"),
        "a lithograph of",
        ref_image="x.png",
    )
    assign_credit(conn, spec, verdict_for(spec, 0.9, 0.9))
    assert arm(conn, "waterfalls", "inverse_hybrid", "high").n_trials == 1
    assert not [
        a for a in list_arms(conn, "inverse_hybrid") if a.word == "albert einstein"
    ]


def test_a_good_low_word_is_not_punished_for_a_bad_partner(conn) -> None:
    """The reason credit is per component rather than per candidate."""
    good_low, bad_high = "a duck", "houseplants"
    for _ in range(5):
        spec = hybrid(good_low, bad_high, "")
        assign_credit(conn, spec, verdict_for(spec, 0.99, 0.01))

    low = arm(conn, good_low, "hybrid", "low")
    high = arm(conn, bad_high, "hybrid", "high")
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

    n_styles = len(list_arms(conn, "hybrid", "style"))
    proposer = UniformProposer(conn, np.random.default_rng(SEED), tasks=("hybrid",))
    proposals = proposer.propose(RunState(run_id="r"), n_styles * 3)

    counts = Counter(p.spec.style for p in proposals)
    assert len(counts) == n_styles, "every style must be exercised"
    assert max(counts.values()) - min(counts.values()) <= 1
    assert all(p.origin == UNIFORM for p in proposals)


def test_uniform_proposer_reaches_the_arms_the_bandit_starves(conn) -> None:
    """`a photo of` is seeded Beta(1, 3); Thompson sampling avoids it by design."""
    bandit = [
        p.spec.style for p in make_proposer(conn).propose(RunState(run_id="r"), 24)
    ]
    uniform = [
        p.spec.style
        for p in UniformProposer(
            conn, np.random.default_rng(SEED), tasks=("hybrid",)
        ).propose(RunState(run_id="r"), 24)
    ]
    assert bandit.count("a photo of") < uniform.count("a photo of")


def test_uniform_proposer_respects_evaluated_uids(conn) -> None:
    state = RunState(run_id="r")
    proposer = UniformProposer(conn, np.random.default_rng(SEED))
    for p in proposer.propose(state, 8):
        state.evaluated_uids.add(p.spec.uid())

    again = UniformProposer(conn, np.random.default_rng(SEED)).propose(state, 8)
    assert not {p.spec.uid() for p in again} & state.evaluated_uids


def test_uniform_proposer_keeps_the_views_distinct(conn) -> None:
    proposer = UniformProposer(
        conn, np.random.default_rng(SEED), tasks=("hybrid", "three_view")
    )
    for p in proposer.propose(RunState(run_id="r"), 16):
        assert len(set(p.spec.prompts)) == p.spec.n_views
