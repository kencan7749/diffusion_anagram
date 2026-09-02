"""Racing: spend extra seeds only on the pairs that keep holding.

A single seed's J is one Bernoulli trial of "does this pair form an illusion",
and seed luck is real (the same pair has scored 0.997 and 0.012). v1 dealt with
it by regenerating a fixed top-8 across four seeds after screening. Racing
does the same thing incrementally, per pair, on the evidence so far:

    rung 0   every new pair gets one seed
    rung 1   pairs that held get a second
    rung 2   pairs that held twice get a third and a fourth

The promotion rule is a lower quantile of the pair's Beta(ok + 1, fail + 1)
posterior on its success rate: promote while the 25% quantile is at least
0.5. With one seed that is exactly the boundary (Beta(2, 1) has its quartile
at 0.5), so one success promotes, one failure out of two stops.

The quantile is computed from the regularised incomplete beta function by
bisection. That is a few lines of numerics rather than a scipy dependency, and
it is exact where a sampled estimate would jitter around the boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from ava.propose import Proposal
from ava.search.archive import Archive, Individual

RACE = "race"


# ---------------------------------------------------------------------------
# Beta quantile without scipy
# ---------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float, max_iter: int = 300, eps: float = 3e-14):
    """Lentz's continued fraction for the incomplete beta (Numerical Recipes 6.4)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / (c if abs(c) > 1e-300 else 1e-300)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / (c if abs(c) > 1e-300 else 1e-300)
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """I_x(a, b): the CDF of Beta(a, b) at x."""
    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"a and b must be positive, got a={a}, b={b}")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_quantile(a: float, b: float, q: float, tol: float = 1e-10) -> float:
    """The q-quantile of Beta(a, b), by bisection on the CDF."""
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if regularized_incomplete_beta(mid, a, b) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RacingConfig:
    max_seeds: int = 4  # rung 2 tops out where v1's harvest did
    quantile: float = 0.25
    threshold: float = 0.5
    fraction: float = 0.3  # share of a round's budget spent on promotions

    def __post_init__(self) -> None:
        if self.max_seeds < 1:
            raise ValueError("max_seeds must be at least 1")
        if not 0.0 <= self.fraction < 1.0:
            raise ValueError("fraction must be in [0, 1)")


def success_rate_quantile(ind: Individual, quantile: float) -> float:
    """Lower quantile of the pair's posterior success rate."""
    return beta_quantile(ind.n_ok + 1.0, ind.n_seeds - ind.n_ok + 1.0, quantile)


def promotable(ind: Individual, cfg: RacingConfig) -> bool:
    if ind.n_seeds == 0 or ind.n_seeds >= cfg.max_seeds:
        return False
    # The one-seed case sits exactly on the boundary; a tolerance keeps the
    # bisection's last bit from deciding it.
    return success_rate_quantile(ind, cfg.quantile) >= cfg.threshold - 1e-9


def next_seed(ind: Individual) -> int:
    """The smallest seed the pair has not been generated with."""
    used = set(ind.seeds)
    seed = 0
    while seed in used:
        seed += 1
    return seed


def race_candidates(archive: Archive, cfg: RacingConfig) -> list[Individual]:
    """Every promotable pair, most promising first, in a reproducible order."""
    eligible = [ind for ind in archive.individuals.values() if promotable(ind, cfg)]
    eligible.sort(key=lambda i: (-i.fitness, -i.n_seeds, i.key_str))
    return eligible


def race_proposals(
    archive: Archive, cfg: RacingConfig, budget: int, exclude: set[str]
) -> list[Proposal]:
    """Spend up to `budget` evaluations on the next seed of promotable pairs."""
    out: list[Proposal] = []
    for ind in race_candidates(archive, cfg):
        if len(out) >= budget:
            break
        spec = replace(ind.spec, seed=next_seed(ind))
        if spec.uid() in exclude:
            continue
        extra: dict[str, Any] = {
            "pair": ind.key_str,
            "n_seeds": ind.n_seeds,
            "n_ok": ind.n_ok,
            "rate_q": round(success_rate_quantile(ind, cfg.quantile), 4),
        }
        out.append(
            Proposal(
                spec,
                RACE,
                f"race seed {spec.seed} of {ind.key_str} "
                f"({ind.n_ok}/{ind.n_seeds} held)",
                extra,
            )
        )
    return out
