"""The seed vocabulary must be traceable, not invented.

Words that merely sounded plausible have been introduced into the seed set by
mistake before. This is the mechanical stop: every `author` string has to occur
verbatim in the upstream checkout, and every `paper` string has to carry a
figure citation and fit the slots of the task it is seeded for.
"""

from pathlib import Path

import pytest

from ava.image.paper_examples import (
    INVERSE_HIGH,
    PAPER_EXAMPLES,
    paper_arms,
    seed_paper_vocab,
)
from ava.image.tasks import TASKS, get_task
from ava.spec import apply_style
from ava.vocab import (
    AUTHOR_SOURCE,
    COMMON_STYLES,
    FD_SUBJECTS,
    HYBRID_STYLES,
    HYBRID_TIER1,
    HYBRID_TIER2,
    HYBRID_TIER3,
    NON_HYBRID_TASKS,
    PAPER_SOURCE,
    SCHEMA_VERSION,
    UNIFORM_PRIOR,
    VA_SUBJECTS,
    VA_TASKS,
    connect,
    list_arms,
    seed_author_vocab,
    update_arm,
)

UPSTREAM = Path("dev/visual_anagrams")
# The files the plan admits as evidence of a word's provenance.
SOURCE_GLOBS = ("tests/*.sh", "readme.md", "readme_factorized_diffusion.md")


def _upstream_text() -> str:
    if not UPSTREAM.exists():
        pytest.skip(f"{UPSTREAM} not checked out")
    chunks = []
    for pattern in SOURCE_GLOBS:
        for path in sorted(UPSTREAM.glob(pattern)):
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def _probe(style_or_word: str) -> str:
    """What to look for upstream: a template minus its placeholder."""
    return style_or_word.replace("{}", "").strip()


def _author_words() -> list[str]:
    words = [w for w, _ in HYBRID_TIER1] + [w for w, _ in HYBRID_TIER2]
    words += list(HYBRID_TIER3) + list(VA_SUBJECTS) + [w for _, w, _ in FD_SUBJECTS]
    # The empty style means "no prefix" and is not a quotable string.
    words += [_probe(s) for s, _ in HYBRID_STYLES if s]
    words += [_probe(s) for s in COMMON_STYLES if s]
    return sorted(set(words))


@pytest.mark.parametrize("word", _author_words())
def test_author_seed_word_occurs_upstream(word: str) -> None:
    assert word in _upstream_text(), (
        f"{word!r} is seeded as source={AUTHOR_SOURCE!r} but does not occur in "
        f"{UPSTREAM}. Either quote the real string or drop the seed."
    )


def test_author_tasks_are_registered_tasks() -> None:
    for task in VA_TASKS + NON_HYBRID_TASKS:
        assert task in TASKS, task
    for task, _, role in FD_SUBJECTS:
        assert role in get_task(task).roles, (task, role)
    assert all(get_task(t).paper == "VA" for t in VA_TASKS)


def test_hybrid_style_priors_reflect_the_readme() -> None:
    priors = dict(HYBRID_STYLES)
    # The tips section calls one out as working well and the other as hard.
    assert priors["an oil painting of"] == (3.0, 1.0)
    assert priors["a photo of"] == (1.0, 3.0)
    for style, prior in HYBRID_STYLES:
        if style not in ("an oil painting of", "a photo of"):
            assert prior == UNIFORM_PRIOR, f"{style!r} has an unjustified prior"


def test_new_task_seeds_start_uniform(tmp_path: Path) -> None:
    """Only the pre-existing hybrid tiers carry informative priors."""
    conn = connect(tmp_path / "vocab.db")
    seed_author_vocab(conn)
    for a in list_arms(conn):
        if a.task != "hybrid":
            assert (a.alpha, a.beta) == UNIFORM_PRIOR, a


def test_every_task_gets_subjects_and_styles(tmp_path: Path) -> None:
    conn = connect(tmp_path / "vocab.db")
    seed_author_vocab(conn)
    seed_paper_vocab(conn)
    for task in TASKS.values():
        for i in task.searched_slots():
            role = task.slots[i].role
            assert list_arms(conn, task.name, role), (task.name, role)
        assert list_arms(conn, task.name, "style"), task.name


def test_seeding_is_idempotent(tmp_path: Path) -> None:
    conn = connect(tmp_path / "vocab.db")
    assert seed_author_vocab(conn) > 0
    assert seed_paper_vocab(conn) > 0
    assert seed_author_vocab(conn) == 0, "re-seeding must not add or reset arms"
    assert seed_paper_vocab(conn) == 0


# -- paper seeds ----------------------------------------------------------------


@pytest.mark.parametrize(
    "example", PAPER_EXAMPLES, ids=lambda e: f"{e.task}:{e.prompts[0]}"
)
def test_paper_example_fits_its_task(example) -> None:
    task = get_task(example.task)
    assert len(example.prompts) == task.n_views
    assert example.citation.startswith(("VA Fig.", "FD Fig."))
    for prompt in example.prompts:
        assert prompt == prompt.strip() and prompt
        full = apply_style(example.style, prompt)
        assert "{}" not in full and "  " not in full


def test_paper_examples_recombine_to_the_published_prompt() -> None:
    """A spot check that the style/subject split reproduces the figure text."""
    ex = next(e for e in PAPER_EXAMPLES if e.style == "oil painting style, {}")
    assert apply_style(ex.style, ex.prompts[0]).startswith("oil painting style, ")
    ex = next(e for e in PAPER_EXAMPLES if e.task == "inner_circle")
    assert apply_style(ex.style, ex.prompts[0]) == "a pop art of albert einstein"


def test_paper_arms_carry_citations_and_valid_roles() -> None:
    arms = paper_arms()
    assert len(arms) == len({(w, t, r) for w, t, r, _ in arms})
    for _word, task, role, citation in arms:
        assert citation
        assert role in get_task(task).roles or role == "style", (task, role)
    assert all(citation for _, _, citation in INVERSE_HIGH)


def test_paper_seeds_are_uniform_and_cited(tmp_path: Path) -> None:
    conn = connect(tmp_path / "vocab.db")
    seed_author_vocab(conn)
    seed_paper_vocab(conn)
    papers = [a for a in list_arms(conn) if a.source == PAPER_SOURCE]
    assert papers
    for a in papers:
        assert (a.alpha, a.beta) == UNIFORM_PRIOR
        assert a.citation


def test_paper_seeding_does_not_touch_author_arms(tmp_path: Path) -> None:
    """`houseplants` is both an author and a paper word; author wins, once."""
    conn = connect(tmp_path / "vocab.db")
    seed_author_vocab(conn)
    update_arm(conn, "houseplants", "hybrid", "high", 1.0)
    seed_paper_vocab(conn)
    a = next(x for x in list_arms(conn, "hybrid", "high") if x.word == "houseplants")
    assert a.source == AUTHOR_SOURCE and a.n_trials == 1


# -- schema migration -------------------------------------------------------------


def test_version_zero_databases_migrate_with_posteriors_intact(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "vocab.db"
    old = sqlite3.connect(path)
    old.executescript(
        """
        CREATE TABLE arm (
            word TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('low', 'high', 'style')),
            source TEXT NOT NULL, alpha REAL NOT NULL, beta REAL NOT NULL,
            n_trials INTEGER NOT NULL DEFAULT 0,
            first_seen_round INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (word, role));
        INSERT INTO arm VALUES ('a panda', 'low', 'author', 7.5, 2.5, 6, 0);
        INSERT INTO arm VALUES ('a photo of', 'style', 'author', 1.0, 3.0, 0, 0);
        """
    )
    old.commit()
    old.close()

    conn = connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    panda = next(a for a in list_arms(conn, "hybrid", "low") if a.word == "a panda")
    assert (panda.alpha, panda.beta, panda.n_trials) == (7.5, 2.5, 6)
    assert panda.citation == ""
    assert list_arms(conn, "hybrid", "style")[0].word == "a photo of"
    # Re-seeding after migration adds the new tasks but leaves the carried rows alone.
    seed_author_vocab(conn)
    again = next(a for a in list_arms(conn, "hybrid", "low") if a.word == "a panda")
    assert again.alpha == 7.5


def test_connect_is_idempotent_on_a_current_database(tmp_path: Path) -> None:
    path = tmp_path / "vocab.db"
    conn = connect(path)
    seed_author_vocab(conn)
    n = len(list_arms(conn))
    conn.close()
    conn = connect(path)
    assert len(list_arms(conn)) == n
