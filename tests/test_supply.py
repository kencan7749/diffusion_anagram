"""In-run vocabulary supply: when it is asked, what it adds, and who calls it."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from ava.audio.loop import AudioLoopConfig
from ava.audio.loop import run_loop as run_audio_loop
from ava.audio.vocab import seed_audio_vocab
from ava.loop import LoopConfig, RunPaths, run_loop
from ava.propose import BanditProposer
from ava.supply import SupplyPolicy, replenish, searched_roles
from ava.vocab import (
    add_arm,
    connect,
    list_arms,
    seed_author_vocab,
    untried_pool,
    update_arm,
)
from tests.test_audio_loop import FakeAudioGenerator, FakeClapJudge
from tests.test_loop import FakeGenerator, FakeJudge


class FakeSupply:
    """Adds `per_call` fresh words for the (task, role) it is asked about."""

    def __init__(self, per_call: int = 3) -> None:
        self.per_call = per_call
        self.calls: list[tuple[str, str, int]] = []

    def supply(self, conn: sqlite3.Connection, task: str, role: str, round_index: int):
        self.calls.append((task, role, round_index))
        added = 0
        for i in range(self.per_call):
            word = f"supplied {task} {role} {len(self.calls)}-{i}"
            added += add_arm(conn, word, task, role, "fake", round_index=round_index)
        return added


def exhaust(conn: sqlite3.Connection, tasks: list[str]) -> None:
    """Try every word under every searched (task, role), so nothing is untried.

    Marking only the registered arms is not enough: a word tried under hybrid
    is still untried for flip, and the pooled vocabulary offers it there.
    """
    words = {a.word for a in list_arms(conn) if a.role != "style"}
    for task, role in searched_roles(tasks):
        for word in sorted(words):
            add_arm(conn, word, task, role, "test")
            update_arm(conn, word, task, role, 0.5)


# -- the policy -----------------------------------------------------------


def test_searched_roles_skip_style_and_the_reference_slot() -> None:
    assert searched_roles(["flip", "hybrid", "inverse_hybrid"]) == [
        ("flip", "subject"),
        ("hybrid", "high"),
        ("hybrid", "low"),
        ("inverse_hybrid", "high"),
    ]


def test_replenish_only_asks_where_the_pool_ran_low(tmp_path: Path) -> None:
    conn = connect(tmp_path / "vocab.db")
    seed_author_vocab(conn)
    supply = FakeSupply()
    # Plenty untried everywhere: nothing is asked for.
    assert replenish(conn, ["flip", "hybrid"], supply, SupplyPolicy(8), 0) == {}
    assert supply.calls == []

    exhaust(conn, ["flip", "hybrid"])
    added = replenish(conn, ["flip", "hybrid"], supply, SupplyPolicy(8), 3)
    assert added == {"flip:subject": 3, "hybrid:high": 3, "hybrid:low": 3}
    assert {c[2] for c in supply.calls} == {3}
    assert len(untried_pool(conn, "flip", "subject")) >= 3
    conn.close()


def test_a_supplied_word_pools_into_every_task(tmp_path: Path) -> None:
    """Words are shared: supplying flip also refills jigsaw's untried pool."""
    conn = connect(tmp_path / "vocab.db")
    seed_author_vocab(conn)
    exhaust(conn, ["flip", "jigsaw"])
    replenish(conn, ["flip"], FakeSupply(2), SupplyPolicy(8), 0)
    assert len(untried_pool(conn, "jigsaw", "subject")) == 2
    conn.close()


def test_policy_rejects_a_negative_floor() -> None:
    with pytest.raises(ValueError):
        SupplyPolicy(-1)


# -- the loops ------------------------------------------------------------


def test_image_loop_tops_up_after_every_round_when_the_floor_is_high(tmp_path) -> None:
    conn = connect(tmp_path / "runs" / "vocab.db")
    seed_author_vocab(conn)
    config = LoopConfig(
        run_id="t",
        tasks=["flip"],
        rounds=3,
        k=4,
        harvest_seeds=0,
        harvest_top=0,
        supply="llm",
        supply_min_untried=10_000,  # always below the floor: asked every round
    )
    paths = RunPaths(tmp_path / "runs" / "t")
    proposer = BanditProposer(conn, np.random.default_rng(0), tasks=["flip"])
    supply = FakeSupply()
    run_loop(config, paths, conn, proposer, FakeGenerator(), FakeJudge(), supply)
    assert [c[2] for c in supply.calls] == [0, 1, 2]
    assert [a for a in list_arms(conn, "flip", "subject") if a.source == "fake"]
    assert "supply: llm" in paths.config.read_text()
    conn.close()


def test_image_loop_leaves_the_vocabulary_alone_without_a_supply(tmp_path) -> None:
    conn = connect(tmp_path / "runs" / "vocab.db")
    seed_author_vocab(conn)
    before = len(list_arms(conn))
    config = LoopConfig(
        run_id="t", tasks=["flip"], rounds=1, k=2, harvest_seeds=0, harvest_top=0
    )
    paths = RunPaths(tmp_path / "runs" / "t")
    proposer = BanditProposer(conn, np.random.default_rng(0), tasks=["flip"])
    run_loop(config, paths, conn, proposer, FakeGenerator(), FakeJudge())
    # Pooling may register existing words under flip, but no new word appears.
    words_before = {a.word for a in list_arms(conn)}
    assert len({a.word for a in list_arms(conn)}) == len(words_before)
    assert len(list_arms(conn)) >= before
    conn.close()


def test_loop_config_rejects_an_unknown_supply() -> None:
    with pytest.raises(ValueError):
        LoopConfig(run_id="x", supply="oracle")


def test_audio_loop_asks_for_sounds_when_its_pool_runs_out(tmp_path) -> None:
    conn = connect(tmp_path / "runs" / "vocab.db")
    seed_audio_vocab(conn)
    exhaust(conn, ["time_reverse"])
    config = AudioLoopConfig(
        run_id="a", rounds=2, k=3, proposer="bandit", supply="llm", supply_min_untried=8
    )
    paths = RunPaths(tmp_path / "runs" / "a")
    proposer = BanditProposer(
        conn,
        np.random.default_rng(0),
        tasks=["time_reverse"],
        guidance_scale=7.0,
        num_inference_steps=100,
    )
    supply = FakeSupply(4)
    run_audio_loop(
        config, paths, conn, proposer, FakeAudioGenerator(), FakeClapJudge(), supply
    )
    assert supply.calls and all(
        c[:2] == ("time_reverse", "subject") for c in supply.calls
    )
    assert config.duration_s == 5.0
    conn.close()
