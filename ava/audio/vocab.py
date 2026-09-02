"""Seed vocabulary for the time-reversal anagram.

Unlike the image track there is no upstream checkout or paper to quote
prompts from, so every prompt here was written for this repository and is
registered with `source='authored'`; nothing below claims a citation it does
not have. The three pairs Step 1 generated are included and cite the script
they came from.

What the prompts have in common is deliberate. Step 0a measured that CLAP
separates a sound from its reverse along the *temporal envelope* (a
percussive burst drops to 0.525 self-similarity, against 0.998 for a control)
and barely at all along pitch direction. So the vocabulary is built from two
envelope families that are each other's mirror image:

    decay   a sharp attack followed by a long decay (a bell, a slam, a splash)
    swell   a sound building up and stopping at once (a balloon inflating,
            a crescendo cut off, an engine revving then killed)

A workable pair is one of each, with a timbre that is plausible both ways.
Both families sit in one pool under the symmetric role `subject`: which
combinations work is what the search measures, and the archive's cells keep
the families from crowding each other out.
"""

from __future__ import annotations

import sqlite3

from ava.audio.tasks import TIME_REVERSE
from ava.image.tasks import STYLE, SUBJECT
from ava.vocab import UNIFORM_PRIOR, add_arm

AUTHORED_SOURCE = "authored"
STEP1_CITATION = "scripts/step1_first_anagram.py PROMPT_PAIRS"

# The Step 1 pairs, in their original (forward, reverse) order.
STEP1_PROMPTS: tuple[str, ...] = (
    "a match being struck, sharp attack then a slow decay",
    "a fire dying down into silence, then stopping",
    "a hammer hitting an anvil once, ringing out",
    "a metallic sound swelling out of silence to a sudden stop",
    "air rushing out of a balloon, fading away",
    "a balloon being inflated, building up to a stop",
)

DECAY_PROMPTS: tuple[str, ...] = (
    "a large bell struck once, ringing and slowly fading",
    "a door slamming shut, the echo dying away",
    "a cymbal crash decaying into silence",
    "a gong struck once, a long shimmering decay",
    "a champagne cork popping, then a fizz fading out",
    "a single drum hit with a long reverb tail",
    "a glass shattering, the fragments settling and going quiet",
    "a firework exploding, the boom rolling away",
    "a stone dropped into water, the splash settling",
    "a car horn blast fading into the distance",
    "a thunderclap fading into a soft rumble",
)

SWELL_PROMPTS: tuple[str, ...] = (
    "a reversed cymbal swelling up and cutting off sharply",
    "wind building from a whisper to a gust, then sudden silence",
    "a crowd murmur rising louder and louder, then cut off",
    "an engine revving up higher and higher, then cut dead",
    "a kettle whistle rising to full pitch, then stopping at once",
    "an orchestra crescendo that stops abruptly",
    "water filling a glass, the pitch rising until it stops",
    "a swarm of bees approaching, growing loud, then silence",
    "a train approaching, getting louder, then a sudden cut",
    "air hissing into a tank, the pressure building, then a click",
    "a rising drone that swells and ends in a sharp stop",
)

# Short sound sources, the form the in-run GPT-2 supply produces (the same
# noun-phrase rule as the painting subjects). They carry no envelope in words;
# a bell implies a decay and an approaching train a swell, and whether that is
# enough for the judge is measured, not assumed. Also the supply's exemplars.
SOURCE_PROMPTS: tuple[str, ...] = (
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

# Stable Audio prompts read as descriptions, not captions with a medium, so the
# only style is none. The arm exists so the loop's credit assignment and the
# component table have somewhere to put J.
STYLES: tuple[str, ...] = ("",)


def seed_audio_vocab(conn: sqlite3.Connection) -> int:
    """Register the authored prompts as arms of `time_reverse`. Idempotent."""
    task = TIME_REVERSE.name
    added = 0
    for word in STEP1_PROMPTS:
        added += add_arm(
            conn,
            word,
            task,
            SUBJECT,
            AUTHORED_SOURCE,
            UNIFORM_PRIOR,
            citation=STEP1_CITATION,
        )
    for word in DECAY_PROMPTS + SWELL_PROMPTS + SOURCE_PROMPTS:
        added += add_arm(conn, word, task, SUBJECT, AUTHORED_SOURCE, UNIFORM_PRIOR)
    for style in STYLES:
        added += add_arm(conn, style, task, STYLE, AUTHORED_SOURCE, UNIFORM_PRIOR)
    return added
