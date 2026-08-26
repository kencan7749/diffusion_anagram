"""The seed vocabulary must be traceable to dev/visual_anagrams, not invented.

Words that merely sounded plausible have been introduced into the seed set by
mistake before. This is the mechanical stop: every string seeded with
source='author' has to occur verbatim in the upstream checkout.
"""

from pathlib import Path

import pytest

from ava.vocab import (
    AUTHOR_SOURCE,
    STYLES,
    TIER1,
    TIER2,
    TIER3,
    UNIFORM_PRIOR,
    connect,
    list_arms,
    seed_author_vocab,
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


def _seed_words() -> list[str]:
    words = [w for w, _ in TIER1] + [w for w, _ in TIER2] + list(TIER3)
    # The empty style means "no prefix" and is not a quotable string.
    words += [s for s, _ in STYLES if s]
    return words


@pytest.mark.parametrize("word", _seed_words())
def test_seed_word_occurs_upstream(word: str) -> None:
    assert word in _upstream_text(), (
        f"{word!r} is seeded as source={AUTHOR_SOURCE!r} but does not occur in "
        f"{UPSTREAM}. Either quote the real string or drop the seed."
    )


def test_no_duplicate_arms_across_tiers() -> None:
    """A word must not be seeded twice for the same role with different priors."""
    seen: set[tuple[str, str]] = set()
    for word, role in TIER1 + TIER2:
        assert (word, role) not in seen, f"duplicate seed: {(word, role)}"
        seen.add((word, role))
    for word in TIER3:
        for role in ("low", "high"):
            assert (word, role) not in seen, (
                f"{word!r} is in TIER3 but already seeded with an informative "
                f"prior for role {role!r}"
            )


def test_style_priors_reflect_the_readme() -> None:
    priors = dict(STYLES)
    # The tips section calls one out as working well and the other as hard.
    assert priors["an oil painting of"] == (3.0, 1.0)
    assert priors["a photo of"] == (1.0, 3.0)
    # Everything else is used upstream without a stated preference.
    for style, prior in STYLES:
        if style not in ("an oil painting of", "a photo of"):
            assert prior == UNIFORM_PRIOR, f"{style!r} has an unjustified prior"


def test_tier3_registers_both_roles(tmp_path: Path) -> None:
    """Tier 3 words have no observable role, so both arms must exist."""
    conn = connect(tmp_path / "vocab.db")
    seed_author_vocab(conn)
    low = {a.word for a in list_arms(conn, "low")}
    high = {a.word for a in list_arms(conn, "high")}
    for word in TIER3:
        assert word in low and word in high, f"{word!r} missing a role arm"


def test_seeding_is_idempotent(tmp_path: Path) -> None:
    conn = connect(tmp_path / "vocab.db")
    first = seed_author_vocab(conn)
    second = seed_author_vocab(conn)
    assert first > 0
    assert second == 0, "re-seeding must not add or reset arms"
