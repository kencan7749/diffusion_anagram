"""Near-duplicate rejection: what it catches, what it must leave alone."""

from __future__ import annotations

import pytest

from ava.search.embed import CachingEmbedder
from ava.search.novelty import Deduplicator, DedupStats
from ava.spec import CandidateSpec
from tests.search_fakes import FakeEmbedder

ETA = 0.95


@pytest.fixture()
def dedup() -> Deduplicator:
    return Deduplicator(CachingEmbedder(FakeEmbedder()), eta=ETA)


def flip(a: str, b: str, style: str = "an oil painting of") -> CandidateSpec:
    return CandidateSpec("flip", (a, b), style)


def test_nothing_known_means_nothing_is_a_duplicate(dedup) -> None:
    accepted, sim = dedup.check(flip("a horse", "marilyn monroe"))
    assert accepted and sim == 0.0


def test_a_synonym_pair_is_rejected(dedup) -> None:
    dedup.add(flip("a horse", "marilyn monroe"))
    accepted, sim = dedup.check(flip("horse", "marilyn monroe"))
    assert not accepted
    assert sim > ETA


def test_a_different_pair_in_the_same_category_is_accepted(dedup) -> None:
    """Semantic neighbours are not duplicates; the cell handles diversity."""
    dedup.add(flip("a horse", "marilyn monroe"))
    accepted, sim = dedup.check(flip("a duck", "albert einstein"))
    assert accepted
    assert sim < ETA


def test_the_same_pair_in_another_style_is_not_a_duplicate(dedup) -> None:
    dedup.add(flip("a horse", "marilyn monroe", "an oil painting of"))
    accepted, _ = dedup.check(flip("a horse", "marilyn monroe", "a photo of"))
    assert accepted


def test_the_same_pair_in_another_task_is_not_a_duplicate(dedup) -> None:
    dedup.add(flip("a horse", "marilyn monroe"))
    accepted, _ = dedup.check(
        CandidateSpec("jigsaw", ("a horse", "marilyn monroe"), "an oil painting of")
    )
    assert accepted


def test_symmetric_tasks_catch_the_swapped_order(dedup) -> None:
    dedup.add(flip("a horse", "marilyn monroe"))
    accepted, sim = dedup.check(flip("marilyn monroe", "horse"))
    assert not accepted and sim > ETA


def test_asymmetric_tasks_keep_the_swapped_order(dedup) -> None:
    dedup.add(CandidateSpec("hybrid", ("a horse", "marilyn monroe"), ""))
    accepted, sim = dedup.check(
        CandidateSpec("hybrid", ("marilyn monroe", "horse"), "")
    )
    assert accepted and sim < 0.5


def test_reference_slot_is_ignored(dedup) -> None:
    a = CandidateSpec(
        "inverse_hybrid", ("albert einstein", "a horse"), "", ref_image="x"
    )
    b = CandidateSpec("inverse_hybrid", ("albert einstein", "horse"), "", ref_image="x")
    dedup.add(a)
    accepted, _ = dedup.check(b)
    assert not accepted


def test_stats_record_every_check_and_summarise_quantiles(dedup) -> None:
    dedup.add(flip("a horse", "marilyn monroe"))
    dedup.check(flip("horse", "marilyn monroe"))
    dedup.check(flip("a skull", "a landscape"))
    s = dedup.summary()
    assert s["n_checked"] == 2 and s["n_rejected"] == 1 and s["eta"] == ETA
    assert set(s["max_similarity_quantiles"]) == {"q50", "q90", "q95", "q99"}
    back = DedupStats.from_dict(dedup.stats.to_dict())
    assert back.n_checked == 2 and len(back.max_similarities) == 2


def test_eta_must_be_a_similarity() -> None:
    with pytest.raises(ValueError):
        Deduplicator(CachingEmbedder(FakeEmbedder()), eta=0.0)
