"""Vocabulary database: the bandit arms and their Beta posteriors.

An arm is a (word, task, role) triple, not a word. "einstein works as the
low-frequency subject of a hybrid but not as the high-frequency one, and works
either way in a flip" is expressible as data, and no one has to predict a new
word's behaviour in advance. Predicting it from CLIP text similarity would not
work: CLIP's text embedding clusters by semantic topic, not by how a subject
survives blurring or rotation. Every arm exists independently and Thompson
sampling tries each.

`role` is the slot's role from `ava.image.tasks`: `low` / `high` / `mid` /
`gray` / `color` / `moving` / `still` for the asymmetric Factorized Diffusion
slots, `subject` for every Visual Anagrams slot (swapping the two prompts of a
flip gives the same illusion flipped), and `style` for the style arm.

The database lives at runs/vocab.db and is shared across runs on purpose. The
accumulated estimate of which prompts work is the asset this system builds; a
per-run reset would throw it away. Run-specific progress belongs in state.json.

The vocabulary is shared across tasks as well (`pooled_arms`): a word that
only `negate` knows is offered to `flip` at the uniform prior and registered
under `flip` when first proposed there. The posteriors are not shared -- the
whole point of the (word, task, role) key is that evidence from one view says
nothing about another.

Seeding policy: `source='author'` arms are strings that literally occur in
dev/visual_anagrams (its tests/*.sh and readmes); `source='paper'` arms are
prompts quoted verbatim from the two papers, each with a figure citation
(`ava.image.paper_examples`). LLM-generated and BLIP-mined words enter later as
untried arms, never as seeds. tests/test_vocab_seed.py enforces the provenance
mechanically, because seeds that merely sounded plausible have been introduced
by mistake before. Priors are Beta(1, 1) unless the upstream readme states a
preference; the paper's stated preferences are recorded as citations, not as
priors, so the data decides.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS arm (
    word              TEXT    NOT NULL,
    task              TEXT    NOT NULL,
    role              TEXT    NOT NULL,
    source            TEXT    NOT NULL,
    alpha             REAL    NOT NULL CHECK (alpha > 0),
    beta              REAL    NOT NULL CHECK (beta  > 0),
    n_trials          INTEGER NOT NULL DEFAULT 0,
    first_seen_round  INTEGER NOT NULL DEFAULT 0,
    citation          TEXT    NOT NULL DEFAULT '',
    PRIMARY KEY (word, task, role)
);
"""

STYLE = "style"
SUBJECT = "subject"

UNIFORM_PRIOR = (1.0, 1.0)
AUTHOR_SOURCE = "author"
PAPER_SOURCE = "paper"

# ---------------------------------------------------------------------------
# Seed vocabulary from the upstream checkout. Every string below must occur
# verbatim somewhere in dev/visual_anagrams; see the module docstring.
# ---------------------------------------------------------------------------

# Hybrid images, tier 1: role is directly observable in a Factorized Diffusion
# example. These priors predate the task split and carry real trials in
# existing databases, so they are left as they were.
HYBRID_TIER1: tuple[tuple[str, str], ...] = (
    ("a panda", "low"),  # hybrid example, low_pass side
    ("a flower arrangement", "high"),  # hybrid example, high_pass side
    ("a yin yang", "low"),  # triple_low_pass
    ("waterfalls", "high"),  # triple_high_pass, and the inverse example
    ("albert einstein", "low"),  # inverse example, low_pass reference image
)
HYBRID_TIER1_PRIOR = (3.0, 1.0)

# Tier 2: the readme's tips section states the role explicitly.
HYBRID_TIER2: tuple[tuple[str, str], ...] = (
    ("marilyn monroe", "low"),
    ("an old man", "low"),
    ("houseplants", "high"),
    ("wine and cheese", "high"),
    ("a kitchen", "high"),
)
HYBRID_TIER2_PRIOR = (2.0, 1.0)

# Tier 3: subjects from the other view types, registered under both hybrid
# roles with a uniform prior and left for the data to decide.
HYBRID_TIER3: tuple[str, ...] = (
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

# Hybrid style arms. The readme tips are explicit about the two extremes.
HYBRID_STYLES: tuple[tuple[str, tuple[float, float]], ...] = (
    ("an oil painting of", (3.0, 1.0)),  # "works well"
    ("a photo of", (1.0, 3.0)),  # "is hard"
    ("a lithograph of", UNIFORM_PRIOR),
    ("a water color of", UNIFORM_PRIOR),  # repo spelling, not "a watercolor of"
    ("a pencil sketch of", UNIFORM_PRIOR),
    ("a mosaic of", UNIFORM_PRIOR),
    ("a painting of", UNIFORM_PRIOR),
    ("", UNIFORM_PRIOR),  # no style prefix
)

# Visual Anagrams subjects from the upstream tests and readme. Every VA task
# gets the whole pool: the readme's own advice is that which subject suits which
# view is unintuitive and has to be searched, so no word is withheld from a view.
VA_SUBJECTS: tuple[str, ...] = (
    "people around a campfire",  # flip
    "an old man",
    "a snowy mountain village",  # rotate_cw / rotate_ccw
    "a horse",
    "houseplants",  # jigsaw
    "marilyn monroe",
    "albert einstein",  # inner_circle
    "a landscape",  # negate
    "a lemur",  # patch_permute
    "a kangaroo",
    "a duck",  # pixel_permute, square_hinge
    "a rabbit",
    "a tudor portrait",  # skew
    "a skull",
    "a waterfall",  # three_view
    "a teddy bear",
)
VA_TASKS: tuple[str, ...] = (
    "flip",
    "rotate_cw",
    "rotate_ccw",
    "rotate_180",
    "skew",
    "jigsaw",
    "inner_circle",
    "negate",
    "patch_permute",
    "pixel_permute",
    "square_hinge",
    "three_view",
    "four_view",
)

# Factorized Diffusion tasks other than the plain hybrid, from the FD readme.
# One example each, registered in the slot it was used in.
FD_SUBJECTS: tuple[tuple[str, str, str], ...] = (
    # (task, word, role)
    ("triple_hybrid", "a yin yang", "low"),
    ("triple_hybrid", "a skull", "mid"),
    ("triple_hybrid", "waterfalls", "high"),
    ("color_hybrid", "landscape", "gray"),
    ("color_hybrid", "tiger", "color"),
    ("motion_hybrid", "a panda", "moving"),
    ("motion_hybrid", "a canyon", "still"),
    ("inverse_hybrid", "waterfalls", "high"),
)

# Style arms for every non-hybrid task: the upstream styles plus the FD readme's
# comma template, all uniform. Which style suits which view is left to the data.
COMMON_STYLES: tuple[str, ...] = tuple(s for s, _ in HYBRID_STYLES) + (
    "{}, oil painting style",  # FD readme: "landscape, oil painting style"
)
NON_HYBRID_TASKS: tuple[str, ...] = VA_TASKS + (
    "triple_hybrid",
    "color_hybrid",
    "motion_hybrid",
    "inverse_hybrid",
)


@dataclass(frozen=True)
class Arm:
    """One bandit arm and its Beta posterior."""

    word: str
    task: str
    role: str
    source: str
    alpha: float
    beta: float
    n_trials: int
    first_seen_round: int
    citation: str = ""

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


# ---------------------------------------------------------------------------
# Connection and schema migration
# ---------------------------------------------------------------------------


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def migrate(conn: sqlite3.Connection) -> int:
    """Bring a database to SCHEMA_VERSION. Returns the version it started at.

    Version 0 keyed arms on (word, role) and knew only the hybrid task. Its
    rows are carried over as task='hybrid' with their posteriors intact: those
    alpha/beta values are accumulated trials and must survive the schema change.
    """
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if version >= SCHEMA_VERSION:
        return version

    has_arm = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='arm'"
    ).fetchone()
    if has_arm and "task" not in _columns(conn, "arm"):
        conn.executescript(
            "ALTER TABLE arm RENAME TO arm_v0;"
            + SCHEMA
            + """
            INSERT INTO arm (word, task, role, source, alpha, beta, n_trials,
                             first_seen_round, citation)
            SELECT word, 'hybrid', role, source, alpha, beta, n_trials,
                   first_seen_round, ''
            FROM arm_v0;
            DROP TABLE arm_v0;
            """
        )
    else:
        conn.executescript(SCHEMA)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()
    return version


def connect(path: str | Path) -> sqlite3.Connection:
    """Open (creating or migrating if needed) the vocabulary database."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    migrate(conn)
    return conn


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def add_arm(
    conn: sqlite3.Connection,
    word: str,
    task: str,
    role: str,
    source: str,
    prior: tuple[float, float] = UNIFORM_PRIOR,
    round_index: int = 0,
    citation: str = "",
) -> bool:
    """Register an arm. Returns False if it was already present.

    Existing arms are never overwritten: their posterior is the accumulated
    result of real trials and must survive re-seeding and re-mining.
    """
    cur = conn.execute(
        "INSERT OR IGNORE INTO arm "
        "(word, task, role, source, alpha, beta, n_trials, first_seen_round, citation) "
        "VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)",
        (word, task, role, source, prior[0], prior[1], round_index, citation),
    )
    conn.commit()
    return cur.rowcount > 0


def seed_author_vocab(conn: sqlite3.Connection) -> int:
    """Insert the dev/visual_anagrams-derived seed vocabulary. Idempotent."""
    added = 0
    for word, role in HYBRID_TIER1:
        added += add_arm(conn, word, "hybrid", role, AUTHOR_SOURCE, HYBRID_TIER1_PRIOR)
    for word, role in HYBRID_TIER2:
        added += add_arm(conn, word, "hybrid", role, AUTHOR_SOURCE, HYBRID_TIER2_PRIOR)
    for word in HYBRID_TIER3:
        for role in ("low", "high"):
            added += add_arm(conn, word, "hybrid", role, AUTHOR_SOURCE, UNIFORM_PRIOR)
    for style, prior in HYBRID_STYLES:
        added += add_arm(conn, style, "hybrid", STYLE, AUTHOR_SOURCE, prior)

    for task in VA_TASKS:
        for word in VA_SUBJECTS:
            added += add_arm(conn, word, task, SUBJECT, AUTHOR_SOURCE, UNIFORM_PRIOR)
    for task, word, role in FD_SUBJECTS:
        added += add_arm(conn, word, task, role, AUTHOR_SOURCE, UNIFORM_PRIOR)
    for task in NON_HYBRID_TASKS:
        for style in COMMON_STYLES:
            added += add_arm(conn, style, task, STYLE, AUTHOR_SOURCE, UNIFORM_PRIOR)
    return added


def _row_to_arm(row: sqlite3.Row) -> Arm:
    return Arm(
        word=row["word"],
        task=row["task"],
        role=row["role"],
        source=row["source"],
        alpha=row["alpha"],
        beta=row["beta"],
        n_trials=row["n_trials"],
        first_seen_round=row["first_seen_round"],
        citation=row["citation"],
    )


def list_arms(
    conn: sqlite3.Connection, task: str | None = None, role: str | None = None
) -> list[Arm]:
    clauses, params = [], []
    if task is not None:
        clauses.append("task = ?")
        params.append(task)
    if role is not None:
        clauses.append("role = ?")
        params.append(role)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(f"SELECT * FROM arm{where}", params).fetchall()
    return [_row_to_arm(r) for r in rows]


def list_tasks(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute("SELECT DISTINCT task FROM arm ORDER BY task")]


def untried_arms(conn: sqlite3.Connection, task: str, role: str) -> list[Arm]:
    """Arms that have never been evaluated; the new-word injection quota draws here."""
    rows = conn.execute(
        "SELECT * FROM arm WHERE task = ? AND role = ? AND n_trials = 0", (task, role)
    ).fetchall()
    return [_row_to_arm(r) for r in rows]


def update_arm(
    conn: sqlite3.Connection, word: str, task: str, role: str, p: float
) -> None:
    """Fold one observation into the arm's posterior.

    `p` is already a probability in [0, 1] (the view's own-prompt probability
    for a subject arm, J for a style arm), so alpha += p / beta += 1 - p is the
    natural conjugate update and needs no scaling.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be a probability in [0, 1], got {p}")
    cur = conn.execute(
        "UPDATE arm SET alpha = alpha + ?, beta = beta + ?, n_trials = n_trials + 1 "
        "WHERE word = ? AND task = ? AND role = ?",
        (p, 1.0 - p, word, task, role),
    )
    if cur.rowcount == 0:
        raise KeyError(f"no such arm: ({word!r}, {task!r}, {role!r})")
    conn.commit()


def thompson_sample(
    conn: sqlite3.Connection,
    task: str,
    role: str,
    rng: np.random.Generator,
    exclude: frozenset[str] = frozenset(),
) -> Arm:
    """Draw one arm by Thompson sampling over the (task, role) Beta posteriors.

    Registered arms only. The proposers draw through `draw_arm`, which also
    sees the words other tasks know; this is the validation-sweep primitive.
    """
    arms = [a for a in list_arms(conn, task, role) if a.word not in exclude]
    if not arms:
        raise LookupError(f"no available arms for ({task!r}, {role!r})")
    draws = rng.beta([a.alpha for a in arms], [a.beta for a in arms])
    return arms[int(np.argmax(draws))]


# ---------------------------------------------------------------------------
# The shared pool: every task can draw every word
# ---------------------------------------------------------------------------

POOLED_SOURCE = "pooled"


def pooled_arms(conn: sqlite3.Connection, task: str, role: str) -> list[Arm]:
    """The (task, role) arms plus every word another task knows, as untried arms.

    Posteriors are per (word, task, role) and stay that way: what is shared is
    the *vocabulary*, not the evidence. A word registered only under `negate`
    (a paper example, a generated word) is offered to `flip` at the uniform
    prior, and becomes a real arm of `flip` the moment it is proposed there
    (`ensure_arm`). Styles pool with styles and subjects with subjects; a
    hybrid's `low` word is a perfectly good `subject` for a flip, and the
    data decides whether it was.
    """
    own = list_arms(conn, task, role)
    known = {a.word for a in own}
    clause = "role = ?" if role == STYLE else "role != ?"
    rows = conn.execute(
        f"SELECT word, MIN(task || ':' || role) AS origin FROM arm "
        f"WHERE {clause} AND task != ? GROUP BY word ORDER BY word",
        (STYLE, task),
    ).fetchall()
    pooled = [
        Arm(
            word=r["word"],
            task=task,
            role=role,
            source=POOLED_SOURCE,
            alpha=UNIFORM_PRIOR[0],
            beta=UNIFORM_PRIOR[1],
            n_trials=0,
            first_seen_round=0,
            citation=f"pooled from {r['origin']}",
        )
        for r in rows
        if r["word"] not in known
    ]
    # One fixed order whatever has been registered so far, so a seeded draw
    # gives the same word before and after a pooled arm becomes a real one.
    return sorted(own + pooled, key=lambda a: a.word)


def ensure_arm(conn: sqlite3.Connection, arm: Arm) -> bool:
    """Register a pooled arm so credit has somewhere to land. Idempotent."""
    return add_arm(
        conn,
        arm.word,
        arm.task,
        arm.role,
        arm.source,
        (arm.alpha, arm.beta),
        arm.first_seen_round,
        arm.citation,
    )


def draw_arm(
    conn: sqlite3.Connection,
    task: str,
    role: str,
    rng: np.random.Generator,
    exclude: frozenset[str] = frozenset(),
) -> Arm:
    """Thompson-sample over the pooled vocabulary and register what was drawn.

    The one deliberate write in a draw: a pooled word that wins the draw is
    added to (task, role) at its prior, so the candidate built from it can be
    credited. A word drawn and then discarded by the caller leaves an untried
    arm behind, which is harmless and is exactly what an untried arm is.
    """
    arms = [a for a in pooled_arms(conn, task, role) if a.word not in exclude]
    if not arms:
        raise LookupError(f"no available arms for ({task!r}, {role!r})")
    draws = rng.beta([a.alpha for a in arms], [a.beta for a in arms])
    chosen = arms[int(np.argmax(draws))]
    if chosen.source == POOLED_SOURCE:
        ensure_arm(conn, chosen)
    return chosen


def untried_pool(conn: sqlite3.Connection, task: str, role: str) -> list[Arm]:
    """Untried arms of (task, role), including the words only other tasks know.

    What the injection quota and the `inject` operator draw from. Pooled arms
    are not registered here; the caller registers the one it uses.
    """
    return [a for a in pooled_arms(conn, task, role) if a.n_trials == 0]
