"""The surrogate: features, the GP, the skill check, and when it steps aside."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ava.search.embed import CachingEmbedder, cosine
from ava.search.surrogate import (
    UCB,
    UNIFORM,
    FeatureMap,
    GaussianProcess,
    Surrogate,
    SurrogateConfig,
    spearman,
)
from ava.spec import CandidateSpec
from ava.vocab import connect, seed_author_vocab, update_arm
from tests.search_fakes import CATEGORIES, FakeEmbedder

ALL_WORDS = [w for words in CATEGORIES.values() for w in words]
STYLES = ("an oil painting of", "a photo of", "a lithograph of", "")


@pytest.fixture()
def conn(tmp_path: Path):
    c = connect(tmp_path / "vocab.db")
    seed_author_vocab(c)
    yield c
    c.close()


@pytest.fixture()
def features(conn) -> FeatureMap:
    return FeatureMap(CachingEmbedder(FakeEmbedder()), conn, ["flip", "hybrid"])


def flip(a: str, b: str, style: str = "an oil painting of") -> CandidateSpec:
    return CandidateSpec("flip", (a, b), style)


def all_pairs(task: str = "flip") -> list[CandidateSpec]:
    """Every ordered pair of distinct words, cycling through styles."""
    out = []
    k = 0
    for a in ALL_WORDS:
        for b in ALL_WORDS:
            if a != b:
                out.append(CandidateSpec(task, (a, b), STYLES[k % len(STYLES)]))
                k += 1
    return out


# -- spearman -------------------------------------------------------------


def test_spearman_hand_examples() -> None:
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    # Ranks (1, 2, 3, 4) vs (2, 1, 4, 3): rho = 1 - 6 * 4 / (4 * 15) = 0.6
    assert spearman([1, 2, 3, 4], [2, 1, 4, 3]) == pytest.approx(0.6)
    assert spearman([1, 2, 3], [5, 5, 5]) == 0.0
    with pytest.raises(ValueError):
        spearman([1], [2])


def test_spearman_averages_ties() -> None:
    assert spearman([1, 1, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9486833, abs=1e-6)


# -- features -------------------------------------------------------------


def test_features_have_one_length_whatever_the_view_count(features) -> None:
    two = features(flip("a horse", "a duck"))
    three = features(CandidateSpec("three_view", ("a horse", "a duck", "a skull"), ""))
    assert two.shape == three.shape
    assert np.isfinite(two).all()


def test_features_end_with_the_task_one_hot(features) -> None:
    assert list(features(flip("a horse", "a duck"))[-2:]) == [1.0, 0.0]
    assert list(features(CandidateSpec("hybrid", ("a horse", "a duck")))[-2:]) == [0, 1]
    assert list(features(CandidateSpec("jigsaw", ("a horse", "a duck")))[-2:]) == [0, 0]


def test_features_carry_the_slot_cosine(features) -> None:
    embed = features.embed
    f = features(flip("a horse", "horse"))
    d = 16
    expected = cosine(embed(["a horse"])[0], embed(["horse"])[0])
    assert f[4 * d + d] == pytest.approx(expected)  # first pairwise cosine slot


def test_features_read_the_component_posteriors(features, conn) -> None:
    before = features(flip("a horse", "a duck"))
    update_arm(conn, "a horse", "flip", "subject", 1.0)
    update_arm(conn, "a horse", "flip", "subject", 1.0)
    after = features(flip("a horse", "a duck"))
    changed = np.nonzero(before != after)[0]
    assert len(changed) == 1
    assert after[changed[0]] > before[changed[0]]


# -- gaussian process -----------------------------------------------------


def test_gp_reproduces_a_smooth_function_and_ranks_it_out_of_sample() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 3))
    y = np.sin(x[:, 0]) + 0.5 * x[:, 1]
    gp = GaussianProcess(noise=0.05)
    gp.fit(x, y, ["t"] * 40)
    mean, std = gp.predict(x, ["t"] * 40)
    assert np.corrcoef(mean, y)[0, 1] > 0.99
    assert (std >= 0).all()
    assert spearman(gp.loo_predictions(), y) > 0.9


def test_gp_loo_matches_a_brute_force_refit() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(25, 2))
    y = x[:, 0] ** 2 - x[:, 1]
    gp = GaussianProcess(noise=0.05)
    gp.fit(x, y, ["t"] * 25)
    loo = gp.loo_predictions()
    brute = []
    for i in range(25):
        mask = np.arange(25) != i
        other = GaussianProcess(noise=0.05)
        # Same lengthscale, so the closed form and the refit agree up to the
        # standardisation being taken on 24 rather than 25 points.
        other.fit(x[mask], y[mask], ["t"] * 24, lengthscale=gp.lengthscale)
        brute.append(other.predict(x[i : i + 1], ["t"])[0][0])
    assert np.abs(loo - np.array(brute)).max() < 0.05 * np.abs(y).max()
    assert spearman(loo, brute) > 0.99


def test_gp_task_kernel_discounts_evidence_from_another_task() -> None:
    x = np.array([[0.0], [0.0]])
    gp = GaussianProcess(task_similarity=0.5, noise=0.01)
    gp.fit(np.array([[0.0], [1.0]]), np.array([1.0, -1.0]), ["a", "a"])
    same, _ = gp.predict(x[:1], ["a"])
    other, _ = gp.predict(x[:1], ["b"])
    assert abs(other[0] - gp._y_mean) < abs(same[0] - gp._y_mean)


def test_gp_refuses_a_single_point() -> None:
    with pytest.raises(ValueError):
        GaussianProcess().fit(np.zeros((1, 2)), np.zeros(1), ["t"])
    with pytest.raises(RuntimeError):
        GaussianProcess().predict(np.zeros((1, 2)), ["t"])


# -- the surrogate ----------------------------------------------------------


def learnable_target(spec: CandidateSpec, embed) -> float:
    """A `sep_min` that depends on the features: far-apart subjects score higher."""
    a, b = embed(list(spec.prompts))
    return 0.1 * (1.0 - cosine(a, b)) + (0.03 if spec.style == "a photo of" else 0.0)


def test_surrogate_becomes_skilled_on_a_learnable_target(features) -> None:
    cfg = SurrogateConfig(min_train=8, window=2)
    sur = Surrogate(cfg, features)
    pairs = all_pairs()
    for spec in pairs[:60]:
        sur.add(spec, learnable_target(spec, features.embed))
    first, second = sur.refit(0).rho, sur.refit(1).rho
    assert first is not None and second is not None and second > 0.5
    assert sur.skilled and sur.mode == UCB

    rest = pairs[60:120]
    chosen, preds, mode = sur.select(rest, 8, np.random.default_rng(0))
    assert mode == UCB and len(chosen) == 8
    truth = [learnable_target(s, features.embed) for s in rest]
    picked = np.mean([truth[i] for i in chosen])
    assert picked > np.mean(truth), "UCB selection must beat the average child"
    assert all({"mean", "std", "ucb"} <= set(p) for p in preds)


def test_surrogate_steps_aside_on_an_unlearnable_target(features) -> None:
    cfg = SurrogateConfig(min_train=8, window=2, rho_threshold=0.2)
    sur = Surrogate(cfg, features)
    rng = np.random.default_rng(0)
    pairs = all_pairs()
    # Enough points that a chance rho of 0.2 is a many-sigma event, and the
    # test says something about the check rather than about one draw.
    train, rest = pairs[:300], pairs[300:340]
    for spec in train:
        sur.add(spec, float(rng.normal()))
    sur.refit(0)
    entry = sur.refit(1)
    assert entry.rho is not None
    assert abs(entry.rho) < 0.15
    assert not sur.skilled and entry.mode == UNIFORM

    chosen, preds, mode = sur.select(rest, 5, np.random.default_rng(3))
    assert mode == UNIFORM and len(chosen) == 5 and chosen == sorted(chosen)
    assert all(p for p in preds), "predictions are still recorded when untrusted"
    again, _, _ = sur.select(rest, 5, np.random.default_rng(3))
    assert again == chosen


def test_surrogate_is_uniform_until_it_has_enough_data(features) -> None:
    sur = Surrogate(SurrogateConfig(min_train=8, window=1), features)
    for spec in all_pairs()[:5]:
        sur.add(spec, 0.1)
    entry = sur.refit(0)
    assert entry.rho is None and entry.mode == UNIFORM and entry.n_train == 5
    chosen, preds, mode = sur.select(all_pairs()[5:15], 3, np.random.default_rng(0))
    assert mode == UNIFORM and len(chosen) == 3
    assert preds == [{} for _ in range(10)]


def test_surrogate_select_handles_empty_and_short_inputs(features) -> None:
    sur = Surrogate(SurrogateConfig(), features)
    assert sur.select([], 3, np.random.default_rng(0)) == ([], [], UNIFORM)
    chosen, _, _ = sur.select(all_pairs()[:2], 5, np.random.default_rng(0))
    assert chosen == [0, 1]


def test_skill_log_round_trips(features) -> None:
    sur = Surrogate(SurrogateConfig(min_train=8, window=1), features)
    for spec in all_pairs()[:10]:
        sur.add(spec, learnable_target(spec, features.embed))
    sur.refit(0)
    rows = sur.log_dicts()
    back = Surrogate.log_from_dicts(rows)
    assert [e.to_dict() for e in back] == rows
    assert rows[0]["n_train"] == 10 and rows[0]["rho"] is not None


def test_surrogate_config_validates_itself() -> None:
    with pytest.raises(ValueError):
        SurrogateConfig(task_similarity=1.5)
    with pytest.raises(ValueError):
        SurrogateConfig(noise=0.0)
    with pytest.raises(ValueError):
        SurrogateConfig(window=0)
