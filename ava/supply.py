"""Growing the vocabulary while a run is in progress.

The injection quota (v1) and the `inject` operator (v2) draw from the arms
that have never been tried. Once those run out -- and with a small
vocabulary they run out within a few rounds -- nothing new enters the search
and every round is a recombination of what is already there. A supply is a
source of *new words*, called between rounds whenever a (task, role)'s
untried pool has fallen below a floor.

What the supply is stays outside this module: the loops take it as a
dependency (`VocabSupply`), and `ava_vocab.generate_vocab.LlmSupply` is the
implementation that samples list continuations from GPT-2. Styles are never
supplied; they are few and deliberate.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ava.image.tasks import STYLE, get_task
from ava.vocab import untried_pool


class VocabSupply(Protocol):
    def supply(
        self, conn: sqlite3.Connection, task: str, role: str, round_index: int
    ) -> int:
        """Insert new untried arms for (task, role); return how many were added."""
        ...


@dataclass(frozen=True)
class SupplyPolicy:
    """When to ask for more words: below this many untried arms for a role."""

    min_untried: int = 8

    def __post_init__(self) -> None:
        if self.min_untried < 0:
            raise ValueError("min_untried must be non-negative")


def searched_roles(tasks: Sequence[str]) -> list[tuple[str, str]]:
    """Every (task, role) a run draws subjects for, in a fixed order."""
    out: list[tuple[str, str]] = []
    for name in tasks:
        task = get_task(name)
        for role in sorted({task.slots[i].role for i in task.searched_slots()}):
            if role != STYLE:
                out.append((task.name, role))
    return out


def replenish(
    conn: sqlite3.Connection,
    tasks: Sequence[str],
    supply: VocabSupply,
    policy: SupplyPolicy,
    round_index: int,
) -> dict[str, int]:
    """Top up every (task, role) whose untried pool is below the floor.

    Returns what was added per `task:role`, so the loop can log it. A supply
    that finds nothing new (every sampled word already known) adds zero and
    is simply called again after the next round.
    """
    added: dict[str, int] = {}
    for task, role in searched_roles(tasks):
        if len(untried_pool(conn, task, role)) < policy.min_untried:
            added[f"{task}:{role}"] = supply.supply(conn, task, role, round_index)
    return added
