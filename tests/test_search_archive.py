"""The MAP-Elites archive: cells, elites, parent weights and persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ava.search.archive import Archive, Clusterer, Individual, pair_key
from ava.search.embed import CachingEmbedder, cosine
from ava.spec import CandidateSpec
from tests.search_fakes import CATEGORIES, FakeEmbedder, verdict_for

ALL_WORDS = [w for words in CATEGORIES.values() for w in words]


@pytest.fixture()
def embed() -> CachingEmbedder:
    return CachingEmbedder(FakeEmbedder())


@pytest.fixture()
def clusterer(embed) -> Clusterer:
    return Clusterer.fit(ALL_WORDS, embed, k=4, seed=0)


@pytest.fixture()
def archive(clusterer, embed) -> Archive:
    return Archive(clusterer, embed)


def flip(a: str, b: str, style: str = "an oil painting of", seed: int = 0):
    return CandidateSpec("flip", (a, b), style, seed=seed)


def hybrid(low: str, high: str, style: str = "a photo of", seed: int = 0):
    return CandidateSpec("hybrid", (low, high), style, seed=seed)


# -- embedding plumbing ---------------------------------------------------


def test_fake_embedder_is_deterministic_and_clustered() -> None:
    a = FakeEmbedder()(["a horse", "horse", "marilyn monroe"])
    b = FakeEmbedder()(["a horse", "horse", "marilyn monroe"])
    assert np.array_equal(a, b)
    assert cosine(a[0], a[1]) > 0.95, "synonyms must be near-duplicates"
    assert cosine(a[0], a[2]) < 0.5, "different categories must be far apart"


def test_caching_embedder_calls_the_model_once_per_string(embed) -> None:
    inner = embed._embed
    embed(["a horse", "a duck"])
    embed(["a duck", "a horse", "a duck"])
    assert inner.calls == 1
    embed(["a skull"])
    assert inner.calls == 2
    assert embed(["a horse"]).shape == (1, 16)


# -- clustering -----------------------------------------------------------


def test_kmeans_recovers_the_planted_categories(clusterer, embed) -> None:
    labels = {w: int(clusterer.assign(embed([w]))[0]) for w in ALL_WORDS}
    for words in CATEGORIES.values():
        assert len({labels[w] for w in words}) == 1, words
    assert len(set(labels.values())) == 4


def test_kmeans_is_reproducible_from_its_seed(embed) -> None:
    a = Clusterer.fit(ALL_WORDS, embed, k=4, seed=3)
    b = Clusterer.fit(ALL_WORDS, embed, k=4, seed=3)
    assert np.array_equal(a.centroids, b.centroids)
    assert np.array_equal(a.labels, b.labels)


def test_kmeans_round_trips_through_npz(clusterer, tmp_path: Path) -> None:
    clusterer.save(tmp_path / "clusters.npz")
    back = Clusterer.load(tmp_path / "clusters.npz")
    assert np.array_equal(back.centroids, clusterer.centroids)
    assert back.words == clusterer.words
    assert back.seed == clusterer.seed


def test_kmeans_with_fewer_words_than_clusters_shrinks_k(embed) -> None:
    c = Clusterer.fit(["a horse", "a duck"], embed, k=8, seed=0)
    assert c.k == 2


def test_kmeans_rejects_empty_vocabulary(embed) -> None:
    with pytest.raises(ValueError):
        Clusterer.fit([], embed, k=4, seed=0)


# -- cells ----------------------------------------------------------------


def test_symmetric_task_cells_ignore_prompt_order(archive) -> None:
    assert archive.cell_for(flip("a horse", "marilyn monroe")) == archive.cell_for(
        flip("marilyn monroe", "a horse")
    )


def test_asymmetric_task_cells_keep_prompt_order(archive) -> None:
    a = archive.cell_for(hybrid("a horse", "marilyn monroe"))
    b = archive.cell_for(hybrid("marilyn monroe", "a horse"))
    assert a == tuple(reversed(b)) and a != b


def test_reference_slot_is_not_part_of_the_cell(archive) -> None:
    spec = CandidateSpec(
        "inverse_hybrid", ("albert einstein", "a horse"), "", ref_image="x.png"
    )
    assert len(archive.cell_for(spec)) == 1


# -- elites ---------------------------------------------------------------


def test_the_better_pair_becomes_the_cell_elite(archive) -> None:
    weak = flip("a horse", "marilyn monroe")
    strong = flip("a duck", "albert einstein")  # same cell: animal + face
    assert archive.cell_for(weak) == archive.cell_for(strong)

    archive.add(weak, verdict_for(weak, 0.9, 0.9, sep=[0.02, 0.03]), 0, "x")
    assert archive.elites[(("flip",) + archive.cell_for(weak))] == pair_key(weak)
    archive.add(strong, verdict_for(strong, 0.9, 0.9, sep=[0.08, 0.09]), 0, "x")
    assert archive.elites[(("flip",) + archive.cell_for(weak))] == pair_key(strong)
    assert len(archive) == 2, "the displaced pair stays in the history"


def test_a_pair_that_held_outranks_a_higher_margin_that_did_not(archive) -> None:
    held = flip("a horse", "marilyn monroe")
    lost = flip("a duck", "albert einstein")
    archive.add(lost, verdict_for(lost, 0.9, 0.1, sep=[0.30, -0.01]), 0, "x")
    archive.add(held, verdict_for(held, 0.9, 0.9, sep=[0.02, 0.02]), 0, "x")
    assert archive.elite_individuals()[0].key == pair_key(held)


def test_ties_go_to_the_pair_with_more_seeds(archive) -> None:
    a = flip("a horse", "marilyn monroe")
    b = flip("a duck", "albert einstein")
    archive.add(a, verdict_for(a, 0.9, 0.9, sep=[0.05, 0.05]), 0, "x")
    archive.add(b, verdict_for(b, 0.9, 0.9, sep=[0.05, 0.05]), 0, "x")
    archive.add(
        flip("a horse", "marilyn monroe", seed=1),
        verdict_for(a, 0.9, 0.9, sep=[0.05, 0.05]),
        1,
        "race",
    )
    assert archive.elite_individuals()[0].key == pair_key(a)
    assert archive.get(a) is not None and archive.get(a).n_seeds == 2


def test_more_seeds_update_fitness_and_can_dethrone(archive) -> None:
    a = flip("a horse", "marilyn monroe")
    b = flip("a duck", "albert einstein")
    archive.add(a, verdict_for(a, 0.9, 0.9, sep=[0.10, 0.10]), 0, "x")
    archive.add(b, verdict_for(b, 0.9, 0.9, sep=[0.06, 0.06]), 0, "x")
    # A second seed of `a` fails badly; its mean drops below b's.
    a1 = flip("a horse", "marilyn monroe", seed=1)
    archive.add(a1, verdict_for(a1, 0.1, 0.9, sep=[-0.10, 0.10]), 1, "race")
    assert archive.get(a).fitness == pytest.approx(0.0)
    assert archive.elite_individuals()[0].key == pair_key(b)


def test_readding_a_uid_is_a_no_op(archive) -> None:
    a = flip("a horse", "marilyn monroe")
    archive.add(a, verdict_for(a, 0.9, 0.9), 0, "x")
    archive.add(a, verdict_for(a, 0.9, 0.9), 0, "x")
    assert archive.get(a).n_seeds == 1
    assert archive.has_uid(a.uid())


def test_children_are_counted_on_their_parents(archive) -> None:
    parent = flip("a horse", "marilyn monroe")
    archive.add(parent, verdict_for(parent, 0.9, 0.9), 0, "x")
    child = flip("a duck", "marilyn monroe")
    archive.add(
        child,
        verdict_for(child, 0.9, 0.9),
        1,
        "evolve",
        operator="swap_slot",
        parents=[archive.get(parent).key_str],
    )
    assert archive.get(parent).n_children == 1
    assert archive.get(child).operator == "swap_slot"


def test_coverage_counts_filled_and_held_cells(archive) -> None:
    a = flip("a horse", "marilyn monroe")
    b = flip("a horse", "a landscape")
    archive.add(a, verdict_for(a, 0.9, 0.9), 0, "x")
    archive.add(b, verdict_for(b, 0.1, 0.9), 0, "x")
    assert archive.coverage() == {"flip": {"filled": 2, "held": 1}}


# -- parent selection -----------------------------------------------------


def test_parent_weights_prefer_fitter_and_less_mined_parents(archive) -> None:
    fit = flip("a horse", "marilyn monroe")
    less = flip("a horse", "a landscape")
    archive.add(fit, verdict_for(fit, 0.9, 0.9, sep=[0.10, 0.10]), 0, "x")
    archive.add(less, verdict_for(less, 0.9, 0.9, sep=[0.02, 0.02]), 0, "x")
    elites, w = archive.parent_weights()
    by_key = dict(zip([e.key for e in elites], w, strict=True))
    assert by_key[pair_key(fit)] > by_key[pair_key(less)]

    archive.get(fit).n_children = 5
    elites, w = archive.parent_weights()
    by_key = dict(zip([e.key for e in elites], w, strict=True))
    assert by_key[pair_key(fit)] < by_key[pair_key(less)]
    assert w.sum() == pytest.approx(1.0)


def test_sampling_parents_is_reproducible_and_task_scoped(archive) -> None:
    f = flip("a horse", "marilyn monroe")
    h = hybrid("a horse", "marilyn monroe")
    archive.add(f, verdict_for(f, 0.9, 0.9), 0, "x")
    archive.add(h, verdict_for(h, 0.9, 0.9), 0, "x")
    assert archive.sample_parent(np.random.default_rng(0)) is not None
    picks_a = [archive.sample_parent(np.random.default_rng(s)).key for s in range(5)]
    picks_b = [archive.sample_parent(np.random.default_rng(s)).key for s in range(5)]
    assert picks_a == picks_b
    only = archive.sample_parent(np.random.default_rng(0), task="hybrid")
    assert only is not None and only.spec.task == "hybrid"
    assert archive.sample_parent(np.random.default_rng(0), task="jigsaw") is None


# -- persistence ----------------------------------------------------------


def test_archive_round_trips_through_jsonl(archive, clusterer, embed, tmp_path) -> None:
    a = flip("a horse", "marilyn monroe")
    b = flip("a duck", "albert einstein")
    archive.add(a, verdict_for(a, 0.9, 0.9, sep=[0.02, 0.03]), 0, "bootstrap")
    archive.add(b, verdict_for(b, 0.9, 0.9, sep=[0.08, 0.09]), 1, "evolve", "restyle")
    path = tmp_path / "archive.jsonl"
    archive.write_jsonl(path)

    back = Archive.read_jsonl(path, clusterer, embed)
    assert set(back.individuals) == set(archive.individuals)
    assert back.elites == archive.elites
    b_back, a_back = back.get(b), back.get(a)
    assert b_back is not None and b_back.operator == "restyle"
    assert a_back is not None and a_back.evaluations[0].sep_min == pytest.approx(0.02)

    back.write_jsonl(tmp_path / "again.jsonl")
    assert (tmp_path / "again.jsonl").read_bytes() == path.read_bytes()


def test_individual_needs_an_evaluation_before_it_has_a_fitness() -> None:
    ind = Individual(spec=flip("a", "b"), cell=(0, 1), first_round=0)
    with pytest.raises(ValueError):
        _ = ind.fitness
