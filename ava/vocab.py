"""Vocabulary database: the bandit arms and their Beta posteriors.

An arm is a (word, role) pair, not a word. That way "einstein works as the
low-frequency subject but not as the high-frequency one" is expressible as
data, and no one has to predict a new word's role in advance. Predicting it
from CLIP text similarity would not work: CLIP's text embedding clusters by
semantic topic, not by spatial-frequency behaviour. Both arms exist
independently and Thompson sampling tries both.

The database lives at runs/vocab.db and is shared across runs on purpose. The
accumulated estimate of which prompts work is the asset this system builds; a
per-run reset would throw it away. Run-specific progress belongs in state.json.

Seeding policy: the initial database contains ONLY strings that literally occur
in dev/visual_anagrams (its tests/*.sh, readme.md, readme_factorized_diffusion.md).
LLM-generated and BLIP-mined words enter later as untried arms, never as seeds.
tests/test_vocab_seed.py enforces this mechanically, because seeds that merely
sounded plausible have been introduced by mistake before.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

Role = Literal["low", "high", "style"]
ROLES: tuple[Role, ...] = ("low", "high", "style")

SUBJECT_ROLES: tuple[Role, ...] = ("low", "high")

SCHEMA = """
CREATE TABLE IF NOT EXISTS arm (
    word              TEXT    NOT NULL,
    role              TEXT    NOT NULL CHECK (role IN ('low', 'high', 'style')),
    source            TEXT    NOT NULL,
    alpha             REAL    NOT NULL CHECK (alpha > 0),
    beta              REAL    NOT NULL CHECK (beta  > 0),
    n_trials          INTEGER NOT NULL DEFAULT 0,
    first_seen_round  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (word, role)
);
"""

# ---------------------------------------------------------------------------
# Seed vocabulary. Every string below must occur verbatim somewhere in
# dev/visual_anagrams; see the module docstring.
# ---------------------------------------------------------------------------

# Tier 1: role is directly observable in a Factorized Diffusion example.
TIER1: tuple[tuple[str, Role], ...] = (
    ("a panda", "low"),  # hybrid example, low_pass side
    ("a flower arrangement", "high"),  # hybrid example, high_pass side
    ("a yin yang", "low"),  # triple_low_pass
    ("waterfalls", "high"),  # triple_high_pass, and the inverse example
    ("albert einstein", "low"),  # inverse example, low_pass reference image
)
TIER1_PRIOR = (3.0, 1.0)

# Tier 2: the readme's tips section states the role explicitly.
#   "faces are very good subjects to hide"  -> low, in a hybrid: a face
#   survives blurring (the classic Einstein/Marilyn construction).
#   "subjects with freedom in how they are depicted are good" -> high, these
#   are distinguished by fine detail.
TIER2: tuple[tuple[str, Role], ...] = (
    ("marilyn monroe", "low"),
    ("an old man", "low"),
    ("houseplants", "high"),
    ("wine and cheese", "high"),
    ("a kitchen", "high"),
)
TIER2_PRIOR = (2.0, 1.0)

# Tier 3: real prompts from the other view types (flip / rotate / jigsaw /
# skew / negate / motion / scale / color). Those views have no low/high
# distinction, so the role cannot be read off. Registered under BOTH roles with
# a uniform prior and left for the data to decide.
#
# `a skull` appears in triple_medium_pass, which does not map onto either half
# of a two-way hybrid, so it belongs here too. The intuition that a crisp
# silhouette "must" be the low side is deliberately not used: the readme states
# that intuition and reasoning are less reliable than search, and that applies
# to the seeding as much as to the loop.
TIER3: tuple[str, ...] = (
    "a horse",
    "a snowy mountain village",
    "a landscape",
    "a teddy bear",
    "a rabbit",
    "a duck",
    "a lemur",
    "a kangaroo",
    "a tudor portrait",
    "a skull",
    "people around a campfire",
    "a canyon",
    "a church",
    "a purple sky",
    "a corgi",
    "tiger",
    "birds",
    "a person",
)
UNIFORM_PRIOR = (1.0, 1.0)

# Style arms. The tips section is explicit about the two extremes; the rest are
# used upstream without any stated preference.
STYLES: tuple[tuple[str, tuple[float, float]], ...] = (
    ("an oil painting of", (3.0, 1.0)),  # "works well"
    ("a photo of", (1.0, 3.0)),  # "is hard"
    ("a lithograph of", UNIFORM_PRIOR),
    ("a water color of", UNIFORM_PRIOR),  # repo spelling, not "a watercolor of"
    ("a pencil sketch of", UNIFORM_PRIOR),
    ("a mosaic of", UNIFORM_PRIOR),
    ("a painting of", UNIFORM_PRIOR),
    ("", UNIFORM_PRIOR),  # no style prefix
)

AUTHOR_SOURCE = "author"


@dataclass(frozen=True)
class Arm:
    """One bandit arm and its Beta posterior."""

    word: str
    role: Role
    source: str
    alpha: float
    beta: float
    n_trials: int
    first_seen_round: int

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def stderr(self) -> float:
        """Standard deviation of the Beta posterior."""
        a, b = self.alpha, self.beta
        return float(np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))))

    def credible_interval(
        self,
        rng: np.random.Generator,
        mass: float = 0.9,
        draws: int = 20000,
    ) -> tuple[float, float]:
        """Equal-tailed credible interval, for the components report.

        Estimated by sampling rather than by inverting the Beta CDF, because
        the incomplete beta function would mean adding scipy for one number.
        `rng` is passed in so the reported interval is reproducible.
        """
        if not 0.0 < mass < 1.0:
            raise ValueError(f"mass must be in (0, 1), got {mass}")
        tail = (1.0 - mass) / 2.0
        sample = rng.beta(self.alpha, self.beta, size=draws)
        lo, hi = np.quantile(sample, [tail, 1.0 - tail])
        return float(lo), float(hi)


def connect(path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) the vocabulary database."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def add_arm(
    conn: sqlite3.Connection,
    word: str,
    role: Role,
    source: str,
    prior: tuple[float, float] = UNIFORM_PRIOR,
    round_index: int = 0,
) -> bool:
    """Register an arm. Returns False if it was already present.

    Existing arms are never overwritten: their posterior is the accumulated
    result of real trials and must survive re-seeding and re-mining.
    """
    cur = conn.execute(
        "INSERT OR IGNORE INTO arm "
        "(word, role, source, alpha, beta, n_trials, first_seen_round) "
        "VALUES (?, ?, ?, ?, ?, 0, ?)",
        (word, role, source, prior[0], prior[1], round_index),
    )
    conn.commit()
    return cur.rowcount > 0


def seed_author_vocab(conn: sqlite3.Connection) -> int:
    """Insert the visual_anagrams-derived seed vocabulary. Idempotent."""
    added = 0
    for word, role in TIER1:
        added += add_arm(conn, word, role, AUTHOR_SOURCE, TIER1_PRIOR)
    for word, role in TIER2:
        added += add_arm(conn, word, role, AUTHOR_SOURCE, TIER2_PRIOR)
    for word in TIER3:
        for role in SUBJECT_ROLES:
            added += add_arm(conn, word, role, AUTHOR_SOURCE, UNIFORM_PRIOR)
    for style, prior in STYLES:
        added += add_arm(conn, style, "style", AUTHOR_SOURCE, prior)
    return added


def _row_to_arm(row: sqlite3.Row) -> Arm:
    return Arm(
        word=row["word"],
        role=row["role"],
        source=row["source"],
        alpha=row["alpha"],
        beta=row["beta"],
        n_trials=row["n_trials"],
        first_seen_round=row["first_seen_round"],
    )


def list_arms(conn: sqlite3.Connection, role: Role | None = None) -> list[Arm]:
    if role is None:
        rows = conn.execute("SELECT * FROM arm").fetchall()
    else:
        rows = conn.execute("SELECT * FROM arm WHERE role = ?", (role,)).fetchall()
    return [_row_to_arm(r) for r in rows]


def untried_arms(conn: sqlite3.Connection, role: Role) -> list[Arm]:
    """Arms that have never been evaluated; the new-word injection quota draws here."""
    rows = conn.execute(
        "SELECT * FROM arm WHERE role = ? AND n_trials = 0", (role,)
    ).fetchall()
    return [_row_to_arm(r) for r in rows]


def update_arm(conn: sqlite3.Connection, word: str, role: Role, p: float) -> None:
    """Fold one observation into the arm's posterior.

    `p` is already a probability in [0, 1] (p_far for a low arm, p_near for a
    high arm, J for a style arm), so alpha += p / beta += 1 - p is the natural
    conjugate update and needs no scaling.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be a probability in [0, 1], got {p}")
    cur = conn.execute(
        "UPDATE arm SET alpha = alpha + ?, beta = beta + ?, n_trials = n_trials + 1 "
        "WHERE word = ? AND role = ?",
        (p, 1.0 - p, word, role),
    )
    if cur.rowcount == 0:
        raise KeyError(f"no such arm: ({word!r}, {role!r})")
    conn.commit()


def thompson_sample(
    conn: sqlite3.Connection,
    role: Role,
    rng: np.random.Generator,
    exclude: frozenset[str] = frozenset(),
) -> Arm:
    """Draw one arm by Thompson sampling over the role's Beta posteriors."""
    arms = [a for a in list_arms(conn, role) if a.word not in exclude]
    if not arms:
        raise LookupError(f"no available arms for role {role!r}")
    draws = rng.beta([a.alpha for a in arms], [a.beta for a in arms])
    return arms[int(np.argmax(draws))]
