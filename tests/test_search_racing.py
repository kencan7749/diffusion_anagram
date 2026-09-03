"""The Beta quantile, the promotion rule, and race proposals."""

from __future__ import annotations

import numpy as np
import pytest

from ava.search.archive import Archive, Clusterer
from ava.search.embed import CachingEmbedder
from ava.search.racing import (
    RACE,
    RacingConfig,
    beta_quantile,
    next_seed,
    promotable,
    race_candidates,
    race_proposals,
    regularized_incomplete_beta,
)
from ava.spec import CandidateSpec
from tests.search_fakes import CATEGORIES, FakeEmbedder, verdict_for

ALL_WORDS = [w for words in CATEGORIES.values() for w in words]


@pytest.fixture()
def archive() -> Archive:
    embed = CachingEmbedder(FakeEmbedder())
    return Archive(Clusterer.fit(ALL_WORDS, embed, k=4, seed=0), embed)


def flip(a: str, b: str, seed: int = 0) -> CandidateSpec:
    return CandidateSpec("flip", (a, b), "an oil painting of", seed=seed)


# -- numerics -------------------------------------------------------------


def test_incomplete_beta_matches_closed_forms() -> None:
    # Beta(1, 1) is uniform; Beta(2, 1) has CDF x^2; Beta(1, 2) has CDF 1-(1-x)^2.
    for x in (0.1, 0.5, 0.9):
        assert regularized_incomplete_beta(x, 1, 1) == pytest.approx(x)
        assert regularized_incomplete_beta(x, 2, 1) == pytest.approx(x**2)
        assert regularized_incomplete_beta(x, 1, 2) == pytest.approx(1 - (1 - x) ** 2)
    # Beta(2, 2): 3x^2 - 2x^3.
    assert regularized_incomplete_beta(0.3, 2, 2) == pytest.approx(3 * 0.09 - 2 * 0.027)
    assert regularized_incomplete_beta(0.0, 3, 4) == 0.0
    assert regularized_incomplete_beta(1.0, 3, 4) == 1.0


def test_incomplete_beta_agrees_with_a_sampled_cdf() -> None:
    rng = np.random.default_rng(0)
    for a, b in ((3.0, 2.0), (5.0, 1.0), (2.5, 7.0)):
        sample = rng.beta(a, b, size=200_000)
        for x in (0.2, 0.5, 0.8):
            assert regularized_incomplete_beta(x, a, b) == pytest.approx(
                float((sample < x).mean()), abs=5e-3
            )


def test_beta_quantile_inverts_the_cdf() -> None:
    assert beta_quantile(2, 1, 0.25) == pytest.approx(0.5, abs=1e-8)
    assert beta_quantile(1, 1, 0.37) == pytest.approx(0.37, abs=1e-8)
    assert beta_quantile(2, 2, 0.5) == pytest.approx(0.5, abs=1e-8)
    q = beta_quantile(3, 2, 0.25)
    assert regularized_incomplete_beta(q, 3, 2) == pytest.approx(0.25, abs=1e-8)
    with pytest.raises(ValueError):
        beta_quantile(1, 1, 1.5)


# -- promotion rule -------------------------------------------------------


def add(archive: Archive, spec: CandidateSpec, ok: bool, round_index: int = 0):
    p = (0.9, 0.9) if ok else (0.9, 0.1)
    sep = [0.05, 0.05] if ok else [0.05, -0.05]
    return archive.add(spec, verdict_for(spec, *p, sep=sep), round_index, "test")


def test_one_success_promotes_and_one_failure_does_not(archive) -> None:
    cfg = RacingConfig()
    held = add(archive, flip("a horse", "marilyn monroe"), ok=True)
    lost = add(archive, flip("a duck", "a landscape"), ok=False)
    assert promotable(held, cfg)
    assert not promotable(lost, cfg)


def test_a_failure_after_a_success_stops_the_race(archive) -> None:
    cfg = RacingConfig()
    ind = add(archive, flip("a horse", "marilyn monroe"), ok=True)
    add(archive, flip("a horse", "marilyn monroe", seed=1), ok=False)
    assert ind.n_seeds == 2 and ind.n_ok == 1
    assert not promotable(ind, cfg)


def test_repeated_success_promotes_up_to_the_seed_cap(archive) -> None:
    cfg = RacingConfig(max_seeds=4)
    ind = add(archive, flip("a horse", "marilyn monroe"), ok=True)
    for seed in (1, 2):
        assert promotable(ind, cfg)
        add(archive, flip("a horse", "marilyn monroe", seed=seed), ok=True)
    assert promotable(ind, cfg)  # 3/3, one seed left
    add(archive, flip("a horse", "marilyn monroe", seed=3), ok=True)
    assert not promotable(ind, cfg), "the cap is where v1's harvest stopped too"


def test_next_seed_skips_the_seeds_already_generated(archive) -> None:
    ind = add(archive, flip("a horse", "marilyn monroe", seed=0), ok=True)
    assert next_seed(ind) == 1
    add(archive, flip("a horse", "marilyn monroe", seed=1), ok=True)
    assert next_seed(ind) == 2
    other = add(archive, flip("a duck", "a skull", seed=5), ok=True)
    assert next_seed(other) == 0


# -- proposals ------------------------------------------------------------


def test_race_proposals_spend_the_budget_on_the_fittest_first(archive) -> None:
    cfg = RacingConfig()
    strong = flip("a horse", "marilyn monroe")
    weak = flip("a duck", "a skull")
    archive.add(strong, verdict_for(strong, 0.9, 0.9, sep=[0.10, 0.10]), 0, "t")
    archive.add(weak, verdict_for(weak, 0.9, 0.9, sep=[0.02, 0.02]), 0, "t")
    add(archive, flip("a lemur", "a landscape"), ok=False)

    assert [i.spec.prompts for i in race_candidates(archive, cfg)] == [
        strong.prompts,
        weak.prompts,
    ]
    proposals = race_proposals(archive, cfg, budget=1, exclude=set())
    assert len(proposals) == 1
    p = proposals[0]
    assert p.origin == RACE
    assert p.spec.prompts == strong.prompts and p.spec.seed == 1
    assert p.extra["n_seeds"] == 1 and p.extra["n_ok"] == 1
    assert p.extra["rate_q"] == pytest.approx(0.5, abs=1e-3)


def test_race_proposals_skip_uids_already_evaluated(archive) -> None:
    cfg = RacingConfig()
    ind = add(archive, flip("a horse", "marilyn monroe"), ok=True)
    taken = flip("a horse", "marilyn monroe", seed=1).uid()
    assert race_proposals(archive, cfg, 4, exclude={taken}) == []
    assert ind.n_seeds == 1


def test_racing_config_validates_itself() -> None:
    with pytest.raises(ValueError):
        RacingConfig(max_seeds=0)
    with pytest.raises(ValueError):
        RacingConfig(fraction=1.0)
