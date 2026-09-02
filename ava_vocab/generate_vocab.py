"""Propose new vocabulary by sampling list continuations, and insert it as arms.

This is the only supply route that reaches outside the vocabulary the system
already has. Caption mining draws on captions of images the system generated
from prompts it already knew, so it fills in neighbours of existing words; it
cannot introduce a subject nobody has tried.

Continuation, not instruction. An instruct model can be asked for "subjects
recognisable from their silhouette", but that role judgement is not trusted
anyway: vocab.py registers both `(word, 'low')` and `(word, 'high')` and lets
Thompson sampling decide, because the upstream readme is explicit that
intuition about which prompts work is less reliable than search. What is
actually needed is noun phrases people really depict, and a plain language
model continuing a list of them produces exactly that. A small early model
suffices.

GPT-2 runs under the transformers 4.35.2 that diffusers 0.24.0 pins, so this
needs no separate environment; it stays in its own directory only so the model
can be swapped without touching `ava/`. It deliberately does not import `ava`:
the loop must not depend on this script having ever run.

Sampling, not greedy decoding, is the point. Greedy returns the argmax, so the
same prompt yields the same continuation every time and a second invocation
adds nothing. Reproducibility comes from seeding explicitly instead: every draw
records its seed and sampling parameters to a provenance log.

Run from the repository root:
    .venv/bin/python -m ava_vocab.generate_vocab --db runs/vocab.db --draws 40
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

MODEL_ID = "gpt2"

# Few-shot list continuation, not free continuation. Three things were found by
# running it:
#
#   * No trailing space. GPT-2's BPE attaches a leading space to the following
#     token, so a prompt ending in " " tokenises to an orphan 'Ġ' and the
#     continuations degenerate into fragments ("ichthyosis", "ileo-moe").
#   * Free continuation never says where the subject ends. It runs on into
#     prose, leaving fragments like "an extinct species called" or bare
#     adjectives like "an unbroken".
#   * A comma-separated list fixes both: the comma is the terminator, every
#     item is already a noun phrase, and one forward pass yields several words.
LIST_PROMPT = "Subjects to paint: {examples},"

# Exemplars come from the same visual_anagrams vocabulary that seeds the
# database, so the few-shot conditioning introduces nothing the project has not
# already committed to. A different subset conditions each draw, which varies
# the output without making this source self-referential the way reading the
# database back would.
EXEMPLARS: tuple[str, ...] = (
    "a panda",
    "a flower arrangement",
    "a yin yang",
    "waterfalls",
    "houseplants",
    "a kitchen",
    "a horse",
    "a teddy bear",
    "a canyon",
    "a church",
    "a skull",
    "a duck",
    "a snowy mountain village",
    "wine and cheese",
)
EXAMPLES_PER_PROMPT = 4

# The arm table is owned by ava/vocab.py. This script only inserts into it, and
# must never create it: a schema copied here would drift.
# tests/test_vocab_generation.py asserts these columns still match.
INSERT_COLUMNS = (
    "word",
    "task",
    "role",
    "source",
    "alpha",
    "beta",
    "n_trials",
    "first_seen_round",
    "citation",
)
LLM_SOURCE = "llm"
UNIFORM_PRIOR = (1.0, 1.0)
# Which (task, role) arms a generated word is registered under, as "task:role".
# The default is the hybrid image's two slots, which is what this generator was
# built for; pass --arm to register words for other tasks.
DEFAULT_ARMS = ("hybrid:low", "hybrid:high")

_VALID = re.compile(r"^[a-z][a-z '\-]{2,48}$")
# Positional words describe where something is, never what it is.
_POSITION = frozenset(
    "background foreground ground middle side top bottom center centre corner "
    "distance front back left right it them frame".split()
)
# The medium is supplied by the style prefix; a subject must not repeat it.
_MEDIUM = frozenset(
    "photo picture image painting drawing print poster art artwork canvas "
    "illustration sketch portrait".split()
)
# A list item starting with one of these is a fragment of the sentence GPT-2
# drifted back into, not an entry ("and an ancient", "which will be").
_FUNCTION_FIRST = frozenset(
    "and or but which that who with in on of for to from while when as is are "
    "was were it its their his her this these those there then than".split()
)
_ADJ_ONLY = frozenset(
    "black white blue red green yellow brown grey gray colorful colourful "
    "bright dark large small old young big little new good great".split()
)
# GPT-2 is unfiltered web text and does produce violent and sexual
# continuations; one appeared in the first trial run. These are refused rather
# than left for a human to catch on a contact sheet, because anything that
# reaches the database will eventually be rendered as an image. Plain age words
# are deliberately absent: "a boy" is an ordinary painting subject, and every
# harmful combination carries one of the terms listed here.
_BLOCKED = frozenset(
    "raped rape raping naked nude nudity porn pornographic sex sexual sexy "
    "erotic fetish corpse corpses dying killed killing murder murdered slain "
    "bloody gore mutilated massacre execution suicide terrorist nazi hitler "
    "isis shooting stabbed torture tortured abuse abused molested pedophile "
    "lynching genocide".split()
)


# ---------------------------------------------------------------------------
# List formats: what the model is asked to continue, and what counts as an item
# ---------------------------------------------------------------------------

# Weapons and violence are refused at the source for sounds as for subjects;
# the first sentence-style run produced "people being shot" unprompted.
_SOUND_BLOCKED = frozenset(
    "gun guns gunshot gunshots shot shots rifle pistol bomb bombs bombing "
    "grenade stabbing scream screams screaming".split()
)

# Exemplars are the short sound sources in ava/audio/vocab.py, so the
# conditioning introduces nothing the audio track has not already committed to.
SOUND_EXEMPLARS: tuple[str, ...] = (
    "a church bell",
    "a door slamming",
    "a kettle whistle",
    "a passing train",
    "a hammer on an anvil",
    "a balloon popping",
    "a gust of wind",
    "a cymbal crash",
    "a car engine",
    "a glass breaking",
    "a crowd cheering",
    "a dripping tap",
)


def extract_sound(item: str) -> str | None:
    """Validate one list item as a sound source: `extract_phrase` plus a blocklist.

    The same short-noun-phrase rule as the painting subjects. Envelope
    sentences ("a bell struck once, ringing and slowly fading") were tried
    first and GPT-2 mostly riffed on the examples; a plain source ("a church
    bell") is what a list continuation produces reliably, and whether it
    works in either direction is for the judge to measure.
    """
    phrase = extract_phrase(item)
    if phrase is None or any(w in _SOUND_BLOCKED for w in phrase.split()):
        return None
    return phrase


@dataclass(frozen=True)
class ListFormat:
    """How a list is presented to the model and read back from it.

    `prompt` is a template with `{examples}`; `joiner` joins the exemplars in
    it; `separator` splits the continuation into items (a comma for short
    noun phrases, a newline for sentences that contain commas of their own);
    `validate` turns one raw item into a clean entry or rejects it.
    """

    name: str
    prompt: str
    exemplars: tuple[str, ...]
    validate: Callable[[str], str | None]
    examples_per_prompt: int = 4
    joiner: str = ", "
    separator: str = ","
    max_new_tokens: int = 48

    def build_prompt(self, draw: int) -> str:
        """Condition draw `draw` on its own rotating slice of the exemplars."""
        n = self.examples_per_prompt
        picked = [
            self.exemplars[(draw * n + i) % len(self.exemplars)] for i in range(n)
        ]
        return self.prompt.format(examples=self.joiner.join(picked))

    def split(self, continuation: str) -> list[str]:
        """Cut a continuation into raw items, stopping where the list ends."""
        if self.separator == ",":
            return split_items(continuation)
        items: list[str] = []
        for k, line in enumerate(continuation.split("\n")):
            stripped = line.strip()
            if k > 0 and not stripped.startswith("-"):
                break  # the model has left the bulleted list
            if stripped.lstrip("- ").strip():
                items.append(stripped)
        return items

    def dedup_key(self, entry: str) -> str:
        return dedup_key(entry) if self.separator == "," else entry

    def clean(self, continuations: Sequence[str]) -> list[str]:
        """Validated, de-duplicated entries in first-seen order."""
        seen: set[str] = set()
        out: list[str] = []
        for continuation in continuations:
            for item in self.split(str(continuation)):
                entry = self.validate(item)
                if entry is None:
                    continue
                key = self.dedup_key(entry)
                if key in seen:
                    continue
                seen.add(key)
                out.append(entry)
        return out


@dataclass(frozen=True)
class SamplingConfig:
    """How the model is sampled. Recorded with every draw.

    `greedy` exists to reproduce an old deterministic run, not for normal use:
    with it, repeated invocations return the same continuation and the
    vocabulary stops growing.
    """

    seed: int = 0
    temperature: float = 0.9
    top_p: float = 0.95
    draws: int = 40
    max_new_tokens: int = 48
    greedy: bool = False

    def seed_for(self, prompt: str, draw: int) -> int:
        """A distinct, reproducible seed per (prompt, draw).

        Derived from the prompt text rather than incremented, so that changing
        the exemplar set or the draw count does not silently shift the seeds of
        other draws, which would break the provenance of words already stored.
        """
        offset = zlib.crc32(prompt.encode("utf-8")) % 10_000
        return self.seed + offset + draw * 1_000_003

    def as_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "draws": self.draws,
            "max_new_tokens": self.max_new_tokens,
            "greedy": self.greedy,
        }


@dataclass(frozen=True)
class Suggestion:
    word: str
    task: str
    role: str


def split_items(continuation: str) -> list[str]:
    """Cut a continuation into list items.

    Stops at the first newline: past it GPT-2 has left the list and gone back
    to writing prose.
    """
    head = continuation.split("\n", 1)[0]
    return [chunk for chunk in head.split(",") if chunk.strip()]


def extract_phrase(item: str) -> str | None:
    """Validate one list item as a subject, or return None.

    Proper nouns are kept on purpose. The seed vocabulary contains
    `albert einstein` and `marilyn monroe`, and the upstream readme calls faces
    the best subjects to hide, so rejecting names would throw away exactly the
    category the author reports works best.
    """
    phrase = " ".join(item.lower().strip().strip("\"'“”-—.:;!?").split())
    if not _VALID.match(phrase):
        return None

    # A possessive means the model is riffing on parts of the previous item
    # ("a duck's feathers", "a duck's tongue", ...). Parts are not subjects.
    if "'s " in phrase or phrase.endswith("'s"):
        return None

    words = phrase.split()
    if len(words) > 4:  # the list ran on into prose
        return None
    if words[0] == "the":  # definite: refers back, not a new subject
        return None
    if words[0] in _FUNCTION_FIRST:
        return None
    body = words[1:] if words[0] in ("a", "an") else words
    if not body:
        return None
    if body[-1] in _POSITION or body[-1] in _ADJ_ONLY:
        return None
    if any(w in _MEDIUM for w in body):
        return None
    if any(w in _BLOCKED for w in words):
        return None
    return phrase


def dedup_key(phrase: str) -> str:
    """Article-insensitive identity.

    "an underwater volcano" and "underwater volcano" are the same subject and
    must not each claim an injection slot.
    """
    words = phrase.split()
    return " ".join(words[1:] if words[0] in ("a", "an", "the") else words)


SUBJECT_FORMAT = ListFormat(
    name="subject",
    prompt=LIST_PROMPT,
    exemplars=EXEMPLARS,
    validate=extract_phrase,
    examples_per_prompt=EXAMPLES_PER_PROMPT,
)

SOUND_FORMAT = ListFormat(
    name="sound",
    prompt="Sounds to record: {examples},",
    exemplars=SOUND_EXEMPLARS,
    validate=extract_sound,
    examples_per_prompt=EXAMPLES_PER_PROMPT,
)

# Which format a task's vocabulary is written in. Anything not listed is a
# painting subject.
FORMATS_BY_TASK: dict[str, ListFormat] = {"time_reverse": SOUND_FORMAT}


def clean(continuations) -> list[str]:
    """Turn raw continuations into validated, de-duplicated subject phrases."""
    seen: set[str] = set()
    out: list[str] = []
    for continuation in continuations:
        for item in split_items(str(continuation)):
            phrase = extract_phrase(item)
            if phrase is None:
                continue
            key = dedup_key(phrase)
            if key in seen:
                continue
            seen.add(key)
            out.append(phrase)
    return out


def parse_arm(spec: str) -> tuple[str, str]:
    """`"hybrid:low"` -> `("hybrid", "low")`."""
    task, sep, role = spec.partition(":")
    if not sep or not task or not role:
        raise SystemExit(f"--arm must look like task:role, got {spec!r}")
    return task, role


def as_suggestions(
    words: list[str], arms: tuple[str, ...] = DEFAULT_ARMS
) -> list[Suggestion]:
    """Register every word under every requested (task, role) arm.

    No role is inferred. List continuation gives no signal about spatial
    frequency, and a guess would not be trusted even if it did: every arm
    exists independently and the bandit tries each.
    """
    parsed = [parse_arm(a) for a in arms]
    return [Suggestion(w, task, role) for w in words for task, role in parsed]


def require_schema(conn: sqlite3.Connection) -> None:
    """Fail loudly rather than creating a second copy of the schema."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='arm'"
    ).fetchone()
    if row is None:
        raise SystemExit(
            "vocab.db has no 'arm' table. Create it with the main package "
            "first (running ava.loop once seeds the schema and the author "
            "vocabulary); this script only inserts."
        )


def insert_suggestions(
    conn: sqlite3.Connection, suggestions: list[Suggestion], round_index: int = 0
) -> int:
    """Insert as untried arms. Existing arms are left untouched.

    INSERT OR IGNORE matters: a word already carrying real trial evidence must
    not be reset to the prior just because it was sampled again.
    """
    placeholders = ", ".join("?" for _ in INSERT_COLUMNS)
    statement = (
        f"INSERT OR IGNORE INTO arm ({', '.join(INSERT_COLUMNS)}) "
        f"VALUES ({placeholders})"
    )
    added = 0
    for s in suggestions:
        cur = conn.execute(
            statement,
            (s.word, s.task, s.role, LLM_SOURCE, *UNIFORM_PRIOR, 0, round_index, ""),
        )
        added += cur.rowcount
    conn.commit()
    return added


def build_prompt(draw: int) -> str:
    """The subject format's prompt for draw `draw`."""
    return SUBJECT_FORMAT.build_prompt(draw)


class ListSampler:
    """GPT-2 held resident, sampling list continuations in a given format.

    Loading the model once matters when the loop asks for words every few
    rounds; the CLI's one-shot `generate` is a thin wrapper over this.
    """

    def __init__(self, model_id: str = MODEL_ID, device: str = "cpu") -> None:
        # Imported lazily so --dry-run tests and the test suite stay cheap.
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        self.model_id = model_id
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_id)
        self.model = GPT2LMHeadModel.from_pretrained(model_id).to(device).eval()

    def sample(
        self, sampling: SamplingConfig, fmt: ListFormat = SUBJECT_FORMAT
    ) -> tuple[list[str], list[dict[str, object]]]:
        """Sample `sampling.draws` continuations. Returns (entries, provenance)."""
        import torch

        words: list[str] = []
        records: list[dict[str, object]] = []
        for draw in range(sampling.draws):
            prompt = fmt.build_prompt(draw)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            prompt_len = int(inputs["input_ids"].shape[-1])

            seed = sampling.seed_for(prompt, draw)
            # Seeding the global RNG is what makes a sampled draw reproducible;
            # transformers' generate() reads from it.
            torch.manual_seed(seed)
            kwargs: dict[str, object] = {
                "max_new_tokens": max(sampling.max_new_tokens, fmt.max_new_tokens),
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            if sampling.greedy:
                kwargs["do_sample"] = False
            else:
                kwargs.update(
                    do_sample=True,
                    temperature=sampling.temperature,
                    top_p=sampling.top_p,
                )
            with torch.no_grad():
                generated = self.model.generate(**inputs, **kwargs)
            continuation = self.tokenizer.decode(
                generated[0][prompt_len:], skip_special_tokens=True
            )
            found = fmt.clean([continuation])
            words += found
            records.append(
                {
                    "model": self.model_id,
                    "format": fmt.name,
                    "prompt": prompt,
                    "draw": draw,
                    "seed": seed,
                    "continuation": continuation.strip()[:240],
                    "n_usable": len(found),
                    **sampling.as_dict(),
                }
            )

        # De-duplicate across draws.
        seen: set[str] = set()
        unique: list[str] = []
        for word in words:
            key = fmt.dedup_key(word)
            if key in seen:
                continue
            seen.add(key)
            unique.append(word)
        return unique, records


def generate(
    model_id: str,
    sampling: SamplingConfig,
    device: str = "cpu",
    fmt: ListFormat = SUBJECT_FORMAT,
) -> tuple[list[str], list[dict[str, object]]]:
    """One-shot sampling: load the model, draw, return (words, provenance)."""
    return ListSampler(model_id, device).sample(sampling, fmt)


class LlmSupply:
    """Vocabulary supply the loops call between rounds.

    `supply(conn, task, role, round_index)` samples a few continuations in
    the task's format, inserts what survives validation as untried arms of
    (task, role) with `source='llm'`, and appends the provenance. Each call
    uses a seed derived from the round and the call count, so successive calls
    keep drawing new words rather than repeating the first draw. The model
    stays resident on `device` (CPU by default: the GPU is the sampler's).
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cpu",
        draws: int = 10,
        seed: int = 0,
        log_path: Path | None = None,
        formats: dict[str, ListFormat] | None = None,
    ) -> None:
        self.draws = draws
        self.seed = seed
        self.log_path = log_path
        self.formats = dict(FORMATS_BY_TASK if formats is None else formats)
        self._sampler = ListSampler(model_id, device)
        self._calls: dict[int, int] = {}

    def format_for(self, task: str) -> ListFormat:
        return self.formats.get(task, SUBJECT_FORMAT)

    def supply(
        self, conn: sqlite3.Connection, task: str, role: str, round_index: int
    ) -> int:
        n = self._calls.get(round_index, 0)
        self._calls[round_index] = n + 1
        sampling = SamplingConfig(
            seed=self.seed + 104_729 * round_index + 7_919 * n, draws=self.draws
        )
        words, records = self._sampler.sample(sampling, self.format_for(task))
        added = insert_suggestions(
            conn, [Suggestion(w, task, role) for w in words], round_index
        )
        if self.log_path is not None:
            write_provenance(
                self.log_path,
                [{**r, "task": task, "role": role} for r in records],
                added,
            )
        return added


def write_provenance(
    log_path: Path, records: list[dict[str, object]], added: int
) -> None:
    """Append what produced these words, so any arm can be traced back.

    The arm table stores only `source='llm'`; the seed, prompt and sampling
    parameters that actually determine a draw live here.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).isoformat()
    with log_path.open("a", encoding="utf-8") as f:
        for record in records:
            entry = {**record, "utc": stamp, "arms_added": added}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=Path("runs/vocab.db"))
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--device", default="cpu", help="CPU keeps the GPU for the loop")
    p.add_argument(
        "--format",
        choices=("subject", "sound"),
        default="subject",
        help="subject: painting subjects; sound: sound sources for the "
        "time-reversal task. Both are comma lists of short noun phrases",
    )
    p.add_argument("--round", type=int, default=0)
    p.add_argument(
        "--arm",
        action="append",
        default=None,
        metavar="TASK:ROLE",
        help="register words under this arm; repeatable "
        f"(default: {' '.join(DEFAULT_ARMS)})",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="vary between invocations to accumulate new words; the same value "
        "reproduces the same draw",
    )
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--draws", type=int, default=40, help="list continuations sampled")
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument(
        "--greedy",
        action="store_true",
        help="decode greedily. Reproduces an old run; useless for growing the "
        "vocabulary, since every invocation returns the same continuation",
    )
    p.add_argument(
        "--log",
        type=Path,
        default=None,
        help="provenance log (default: <db parent>/vocab_generation_log.jsonl)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the words without touching the database",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sampling = SamplingConfig(
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        draws=args.draws,
        max_new_tokens=args.max_new_tokens,
        greedy=args.greedy,
    )
    if sampling.greedy and sampling.draws > 1:
        raise SystemExit(
            "--greedy with --draws > 1 repeats one identical continuation; "
            "use sampling."
        )

    fmt = SOUND_FORMAT if args.format == "sound" else SUBJECT_FORMAT
    words, records = generate(args.model, sampling, args.device, fmt)
    print(
        f"{len(words)} distinct words from {len(records)} continuations",
        file=sys.stderr,
    )
    if args.dry_run:
        for w in words:
            print(w)
        return

    if not args.db.exists():
        raise SystemExit(f"{args.db} does not exist; run the loop once first.")
    conn = sqlite3.connect(args.db)
    try:
        require_schema(conn)
        arms = tuple(args.arm) if args.arm else DEFAULT_ARMS
        added = insert_suggestions(conn, as_suggestions(words, arms), args.round)
    finally:
        conn.close()

    log_path = args.log or args.db.parent / "vocab_generation_log.jsonl"
    write_provenance(log_path, records, added)
    print(
        f"added {added} new arms from {len(words)} words; "
        f"provenance appended to {log_path}"
    )


if __name__ == "__main__":
    main()
