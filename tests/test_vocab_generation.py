"""ava_vocab writes into a table owned by ava.vocab.

The generator is kept out of the package so its model can be swapped without
touching `ava/`, which means its INSERT column list is a second, independent
statement of the schema. These tests keep the two from drifting, and cover the
parsing that stands between GPT-2's unfiltered prose and the database.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from ava.vocab import (
    SCHEMA,
    connect,
    list_arms,
    seed_author_vocab,
    untried_arms,
    update_arm,
)
from ava_vocab.generate_vocab import (
    EXEMPLARS,
    INSERT_COLUMNS,
    LLM_SOURCE,
    SUBJECT_ROLES,
    SamplingConfig,
    Suggestion,
    as_suggestions,
    build_prompt,
    clean,
    dedup_key,
    extract_phrase,
    insert_suggestions,
    main,
    require_schema,
    split_items,
    write_provenance,
)


def test_insert_columns_match_the_owning_schema() -> None:
    """If ava.vocab's table changes, this fails instead of a live run."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    actual = {row[1] for row in conn.execute("PRAGMA table_info(arm)")}
    assert set(INSERT_COLUMNS) == actual


def test_require_schema_refuses_an_empty_database(tmp_path: Path) -> None:
    """The generator must never create the table itself; a copy would drift."""
    conn = sqlite3.connect(tmp_path / "empty.db")
    with pytest.raises(SystemExit):
        require_schema(conn)


# -- prompting -----------------------------------------------------------


def test_prompt_has_no_trailing_space() -> None:
    """GPT-2's BPE attaches the space to the next token.

    A prompt ending in " " tokenises to an orphan 'Ġ' and the continuations
    collapse into fragments. This was observed, not theorised.
    """
    prompt = build_prompt(0)
    assert prompt == prompt.rstrip()


def test_prompt_rotates_its_exemplars() -> None:
    """Successive draws must not all be anchored on the same subjects."""
    assert build_prompt(0) != build_prompt(1)


def test_exemplars_are_seed_vocabulary() -> None:
    """Conditioning must not smuggle in vocabulary the project never committed to."""
    from ava.vocab import TIER1, TIER2, TIER3

    seeded = {w for w, _ in TIER1} | {w for w, _ in TIER2} | set(TIER3)
    assert set(EXEMPLARS) <= seeded


# -- parsing -------------------------------------------------------------


def test_splits_a_comma_list() -> None:
    assert [i.strip() for i in split_items(" a fox, a barn, a gecko")] == [
        "a fox",
        "a barn",
        "a gecko",
    ]


def test_stops_at_the_newline_where_the_list_ends() -> None:
    """Past the newline GPT-2 has gone back to writing prose."""
    items = split_items("a fox, a barn\n\nThe painting was sold in 1929, later")
    assert [i.strip() for i in items] == ["a fox", "a barn"]


@pytest.mark.parametrize(
    "item", ["a fox", "an elephant", "waterfalls", "a small cottage"]
)
def test_accepts_plain_subjects(item: str) -> None:
    assert extract_phrase(item) == item


def test_keeps_proper_nouns() -> None:
    """The seed vocabulary has `albert einstein`; faces are the best subjects.

    Rejecting names would discard exactly the category the upstream readme
    reports works best for the hidden side.
    """
    assert extract_phrase("Albert Einstein") == "albert einstein"


@pytest.mark.parametrize(
    "item",
    [
        "and an ancient",  # sentence fragment
        "which will be",  # relative clause
        "the same name",  # definite: refers back
        "a duck's feathers",  # riffing on parts of the previous item
        "a black",  # bare adjective
        "the background",  # positional
        "an oil painting",  # the medium, supplied by the style prefix
        "a woman who was raped in a war",  # blocked
        "a very long run on phrase that is really a sentence",
        "",
    ],
)
def test_rejects_what_is_not_a_subject(item: str) -> None:
    assert extract_phrase(item) is None


def test_blocklist_covers_what_the_model_actually_produced() -> None:
    """GPT-2 is unfiltered; a violent continuation appeared in the first run.

    Anything reaching the database is eventually rendered as an image, so this
    is refused at the source rather than left for a human to catch on a sheet.
    """
    assert extract_phrase("a raped woman") is None
    # Plain age words are deliberately not blocked: an ordinary subject.
    assert extract_phrase("a boy") == "a boy"


def test_dedup_is_article_insensitive() -> None:
    """Otherwise the same subject claims two injection slots."""
    assert dedup_key("an underwater volcano") == dedup_key("underwater volcano")
    assert clean(["an underwater volcano, underwater volcano"]) == [
        "an underwater volcano"
    ]


def test_clean_preserves_order_and_drops_duplicates() -> None:
    assert clean(["a fox, a barn, a fox, a gecko"]) == ["a fox", "a barn", "a gecko"]


# -- roles ---------------------------------------------------------------


def test_both_roles_are_registered_for_every_word() -> None:
    """No role is inferred; list continuation carries no frequency signal."""
    suggestions = as_suggestions(["a gecko"])
    assert {s.role for s in suggestions} == set(SUBJECT_ROLES)
    assert {s.word for s in suggestions} == {"a gecko"}


# -- sampling ------------------------------------------------------------


def test_sampling_is_on_by_default() -> None:
    """Greedy decoding caps the vocabulary at its first draw.

    This is the only supply route reaching outside what the system already
    knows -- caption mining only fills in neighbours -- so a deterministic
    default would freeze the search space.
    """
    assert SamplingConfig().greedy is False
    assert 0.0 < SamplingConfig().temperature <= 1.5
    assert 0.0 < SamplingConfig().top_p <= 1.0


def test_seeds_differ_across_draws() -> None:
    cfg = SamplingConfig(seed=0, draws=5)
    seeds = [cfg.seed_for(build_prompt(d), d) for d in range(5)]
    assert len(set(seeds)) == len(seeds)


def test_seed_derivation_is_reproducible() -> None:
    a, b = SamplingConfig(seed=7), SamplingConfig(seed=7)
    assert a.seed_for("p", 2) == b.seed_for("p", 2)


def test_changing_draw_count_does_not_shift_other_seeds() -> None:
    """Adding draws must not invalidate the provenance of earlier words."""
    few, many = SamplingConfig(seed=0, draws=1), SamplingConfig(seed=0, draws=9)
    assert few.seed_for("p", 0) == many.seed_for("p", 0)


def test_different_run_seeds_give_different_draws() -> None:
    """Varying --seed between invocations is how the vocabulary accumulates."""
    assert SamplingConfig(seed=0).seed_for("p", 0) != SamplingConfig(seed=1).seed_for(
        "p", 0
    )


def test_greedy_with_multiple_draws_is_refused() -> None:
    """It would repeat one identical continuation and look like real sampling."""
    with pytest.raises(SystemExit):
        main(["--greedy", "--draws", "3", "--dry-run"])


# -- insertion and provenance -------------------------------------------


def test_inserted_words_are_untried_with_a_uniform_prior(tmp_path: Path) -> None:
    conn = connect(tmp_path / "vocab.db")
    assert insert_suggestions(conn, [Suggestion("a lighthouse", "low")]) == 1

    arm = next(a for a in list_arms(conn, "low") if a.word == "a lighthouse")
    assert (arm.alpha, arm.beta) == (1.0, 1.0)
    assert arm.n_trials == 0
    assert arm.source == LLM_SOURCE


def test_insertion_never_resets_an_arm_that_has_evidence(tmp_path: Path) -> None:
    """A re-sampled word must keep the trials it already earned."""
    conn = connect(tmp_path / "vocab.db")
    seed_author_vocab(conn)
    update_arm(conn, "a panda", "low", 1.0)
    before = next(a for a in list_arms(conn, "low") if a.word == "a panda")

    assert insert_suggestions(conn, [Suggestion("a panda", "low")]) == 0
    after = next(a for a in list_arms(conn, "low") if a.word == "a panda")
    assert (after.alpha, after.n_trials, after.source) == (
        before.alpha,
        before.n_trials,
        "author",
    )


def test_generated_words_become_injectable_arms(tmp_path: Path) -> None:
    """The whole point: new words must reach the loop's injection quota."""
    conn = connect(tmp_path / "vocab.db")
    seed_author_vocab(conn)
    for arm in list_arms(conn):
        update_arm(conn, arm.word, arm.role, 0.5)
    assert untried_arms(conn, "low") == []

    insert_suggestions(conn, as_suggestions(["a lighthouse"]))
    assert [a.word for a in untried_arms(conn, "low")] == ["a lighthouse"]
    assert [a.word for a in untried_arms(conn, "high")] == ["a lighthouse"]


def test_provenance_records_seed_and_hyperparameters(tmp_path: Path) -> None:
    """The arm table stores only source='llm'; the draw is traceable from here."""
    cfg = SamplingConfig(seed=3, temperature=0.8, top_p=0.9, draws=2)
    record = {
        "model": "gpt2",
        "prompt": build_prompt(0),
        "draw": 0,
        "seed": cfg.seed_for(build_prompt(0), 0),
        "n_usable": 12,
        **cfg.as_dict(),
    }
    log = tmp_path / "vocab_generation_log.jsonl"
    write_provenance(log, [record], added=9)

    entry = json.loads(log.read_text().strip())
    for key in (
        "model",
        "prompt",
        "seed",
        "temperature",
        "top_p",
        "draws",
        "greedy",
        "utc",
        "arms_added",
        "draw",
    ):
        assert key in entry, f"provenance is missing {key!r}"
    assert entry["arms_added"] == 9
    assert entry["greedy"] is False


def test_provenance_appends_rather_than_overwrites(tmp_path: Path) -> None:
    """Each invocation adds words; the log must keep the whole history."""
    log = tmp_path / "log.jsonl"
    for seed in (0, 1):
        record = {"draw": 0, **SamplingConfig(seed=seed).as_dict()}
        write_provenance(log, [record], added=seed)
    assert len(log.read_text().strip().splitlines()) == 2
