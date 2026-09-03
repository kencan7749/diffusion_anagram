"""The evolutionary proposer driving the real loop, with stand-ins for the GPU.

What is checked here is the wiring and the guarantees the design makes:
every round is full, every file the design names is written, a run is
reproducible from its seed byte for byte, a resumed run continues rather than
restarts, racing spends seeds on pairs that held, and nothing in `ava.search`
needs torch.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from ava.loop import LoopConfig, RunPaths, run_loop
from ava.search.evolve import (
    ARCHIVE_FILENAME,
    BOOTSTRAP,
    CLUSTERS_FILENAME,
    EVOLVE,
    STATE_FILENAME,
    EvolutionaryProposer,
    SearchConfig,
)
from ava.search.racing import RACE, RacingConfig
from ava.search.surrogate import SurrogateConfig
from ava.spec import RunState, Verdict
from ava.vocab import connect, list_arms, seed_author_vocab
from tests.search_fakes import FakeEmbedder
from tests.test_loop import FakeGenerator

SEED = 0
TASKS = ["flip", "jigsaw", "hybrid"]


class StructuredJudge:
    """Margins that depend on (word, task), so there is something to search for.

    Each (word, task) has a fixed margin in [-0.08, 0.12] drawn from a hash of
    the pair, so some words are reliably good in some views. The seed adds a
    small deterministic wobble so racing has seed variation to react to.
    """

    def views(self, img, viewset):
        return viewset.perceive(img)

    @staticmethod
    def _margin(word: str, task: str, seed: int) -> float:
        h = hashlib.sha1(f"{task}|{word}".encode()).digest()
        base = -0.08 + 0.20 * (int.from_bytes(h[:4], "little") / 2**32)
        wobble = 0.03 * ((seed * 7919 + len(word)) % 5 - 2) / 2
        return base + wobble

    def evaluate(self, img, spec, viewset) -> Verdict:
        n = spec.n_views
        sep = [self._margin(w, spec.task, spec.seed) for w in spec.prompts]
        scores = [[0.2] * n for _ in range(n)]
        for i in range(n):
            scores[i][i] = 0.2 + sep[i]
        p = [1.0 / (1.0 + float(np.exp(-100.0 * s))) for s in sep]
        return Verdict(
            uid=spec.uid(),
            task=spec.task,
            slots=viewset.task.slot_names,
            scores=scores,
            p=p,
            j=min(p),
            alignment=min(scores[i][i] for i in range(n)),
            concealment=0.5,
            captions=[f"a view of {w}" for w in spec.prompts],
        )


def make_proposer(
    conn, root: Path | None, seed: int = SEED, **cfg
) -> EvolutionaryProposer:
    return EvolutionaryProposer(
        conn,
        np.random.default_rng(seed),
        FakeEmbedder(),
        tasks=TASKS,
        cfg=SearchConfig(
            clusters=4,
            children_per_slot=8,
            surrogate=SurrogateConfig(min_train=8, window=2),
            **cfg,
        ),
        state_dir=root,
    )


def run(tmp_path: Path, name: str, rounds: int = 3, k: int = 8, seed: int = SEED):
    conn = connect(tmp_path / name / "vocab.db")
    seed_author_vocab(conn)
    config = LoopConfig(
        run_id=name,
        tasks=list(TASKS),
        rounds=rounds,
        k=k,
        seed=seed,
        harvest_seeds=0,
        harvest_top=0,
        proposer="evolve",
    )
    paths = RunPaths(tmp_path / name / name)
    proposer = make_proposer(conn, paths.root, seed=seed)
    run_loop(config, paths, conn, proposer, FakeGenerator(), StructuredJudge())
    return config, paths, conn, proposer


def rows_of(paths: RunPaths) -> list[dict]:
    out = []
    for p in paths.all_scores():
        out += [json.loads(line) for line in p.read_text().splitlines() if line]
    return out


# -- layout and budget ---------------------------------------------------


def test_run_writes_the_search_files_and_full_rounds(tmp_path: Path) -> None:
    config, paths, conn, proposer = run(tmp_path, "a")
    for name in (ARCHIVE_FILENAME, CLUSTERS_FILENAME, STATE_FILENAME):
        assert (paths.root / name).exists(), name
    rows = rows_of(paths)
    assert len(rows) == config.rounds * config.k
    assert {r["origin"] for r in rows} <= {BOOTSTRAP, EVOLVE, RACE}
    assert all(r["origin"] == BOOTSTRAP for r in rows if r["round"] == 0)
    later = [r for r in rows if r["round"] > 0]
    assert any(r["origin"] == EVOLVE for r in later)
    assert {r["task"] for r in rows} == set(TASKS)
    conn.close()


def test_config_records_the_search_settings(tmp_path: Path) -> None:
    config, paths, conn, _ = run(tmp_path, "cfg", rounds=1)
    written = yaml.safe_load(paths.config.read_text())
    assert written["proposer"] == "evolve"
    assert written["search"]["eta"] == 0.95
    assert written["search"]["racing"]["max_seeds"] == 4
    assert LoopConfig(run_id="x", search=written["search"]).search_config().eta == 0.95
    conn.close()


def test_loop_config_rejects_an_unknown_proposer() -> None:
    with pytest.raises(ValueError):
        LoopConfig(run_id="x", proposer="oracle")


def test_scores_carry_the_search_provenance(tmp_path: Path) -> None:
    _, paths, conn, _ = run(tmp_path, "prov")
    for r in rows_of(paths):
        assert "extra" in r
        if r["origin"] == EVOLVE:
            assert r["extra"]["operator"] in (
                "swap_slot",
                "crossover",
                "restyle",
                "transpose",
                "transfer_task",
                "inject",
            )
            assert r["extra"]["parents"]
            assert r["extra"]["selection"] in ("ucb", "uniform")
            assert "surrogate" in r["extra"]
        if r["origin"] == RACE:
            assert r["extra"]["n_seeds"] >= 1
    conn.close()


# -- racing ----------------------------------------------------------------


def test_pairs_that_held_get_a_second_seed(tmp_path: Path) -> None:
    _, paths, conn, proposer = run(tmp_path, "race", rounds=3)
    races = [r for r in rows_of(paths) if r["origin"] == RACE]
    assert races, "with 30% of the budget on racing, something must be raced"
    for r in races:
        assert r["seed"] > 0
    raced = [ind for ind in proposer.archive if ind.n_seeds > 1]
    assert raced
    assert all(ind.evaluations[0].ok for ind in raced), "only pairs that held"
    conn.close()


def test_racing_leaves_at_least_one_new_candidate_per_round(tmp_path: Path) -> None:
    """k=1 with a 90% race share must still propose something new each round."""
    conn = connect(tmp_path / "vocab.db")
    seed_author_vocab(conn)
    proposer = make_proposer(conn, None, racing=RacingConfig(fraction=0.9))
    config = LoopConfig(
        run_id="k1", tasks=TASKS, rounds=4, k=1, harvest_seeds=0, proposer="evolve"
    )
    paths = RunPaths(tmp_path / "k1")
    run_loop(config, paths, conn, proposer, FakeGenerator(), StructuredJudge())
    rows = rows_of(paths)
    assert len(rows) == 4
    assert all(r["origin"] != RACE for r in rows)
    conn.close()


def test_racing_can_be_switched_off(tmp_path: Path) -> None:
    conn = connect(tmp_path / "vocab.db")
    seed_author_vocab(conn)
    proposer = make_proposer(conn, None, racing=RacingConfig(fraction=0.0))
    config = LoopConfig(
        run_id="norace", tasks=TASKS, rounds=3, k=6, harvest_seeds=0, proposer="evolve"
    )
    paths = RunPaths(tmp_path / "norace")
    run_loop(config, paths, conn, proposer, FakeGenerator(), StructuredJudge())
    assert not [r for r in rows_of(paths) if r["origin"] == RACE]
    assert all(ind.n_seeds == 1 for ind in proposer.archive)
    conn.close()


# -- archive and state -----------------------------------------------------


def test_archive_holds_every_evaluation_and_names_its_elites(tmp_path: Path) -> None:
    config, paths, conn, proposer = run(tmp_path, "arch")
    lines = [
        json.loads(x) for x in (paths.root / ARCHIVE_FILENAME).read_text().splitlines()
    ]
    uids = {e["uid"] for ind in lines for e in ind["evaluations"]}
    assert uids == {r["uid"] for r in rows_of(paths)}
    assert any(ind["elite"] for ind in lines)
    assert all(len(ind["cell"]) == 2 for ind in lines)
    state = json.loads((paths.root / STATE_FILENAME).read_text())
    assert len(state["rounds"]) == config.rounds
    assert state["rounds"][-1]["coverage"]
    assert state["dedup"]["n_checked"] > 0
    assert len(state["surrogate_log"]) == config.rounds
    assert not state["pending"], "everything proposed was observed"
    conn.close()


def test_observe_is_idempotent(tmp_path: Path) -> None:
    conn = connect(tmp_path / "vocab.db")
    seed_author_vocab(conn)
    proposer = make_proposer(conn, None)
    state = RunState(run_id="r")
    proposals = proposer.propose(state, 4)
    judge = StructuredJudge()
    from ava.image.tasks import get_task
    from ava.image.views import ViewSet

    completed = []
    for p in proposals:
        vs = ViewSet.build(get_task(p.spec.task))
        completed.append((p.spec, judge.evaluate(None, p.spec, vs)))
        state.record(p.spec, completed[-1][1])
    state.last_round = completed
    state.round_index = 1
    proposer.observe(state)
    n_pairs, n_train = len(proposer.archive), proposer.surrogate.n_train
    proposer.observe(state)
    assert len(proposer.archive) == n_pairs
    assert proposer.surrogate.n_train == n_train
    assert proposer.fitness_stats.n == n_train
    conn.close()


def test_propose_rejects_non_positive_k(tmp_path: Path) -> None:
    conn = connect(tmp_path / "vocab.db")
    seed_author_vocab(conn)
    with pytest.raises(ValueError):
        make_proposer(conn, None).propose(RunState(run_id="r"), 0)
    conn.close()


def test_transferred_words_are_registered_in_the_target_task(tmp_path: Path) -> None:
    _, paths, conn, _ = run(tmp_path, "xfer", rounds=4)
    transfers = [
        r for r in rows_of(paths) if r["extra"].get("operator") == "transfer_task"
    ]
    if not transfers:
        pytest.skip("the operator bandit did not select a transfer in this run")
    for r in transfers:
        words = {a.word for a in list_arms(conn, r["task"])}
        assert set(r["prompts"]) <= words
    conn.close()


# -- reproducibility and resume -------------------------------------------


def test_same_seed_gives_the_same_archive_byte_for_byte(tmp_path: Path) -> None:
    _, a, ca, _ = run(tmp_path, "left")
    _, b, cb, _ = run(tmp_path, "right")
    assert (a.root / ARCHIVE_FILENAME).read_bytes() == (
        b.root / ARCHIVE_FILENAME
    ).read_bytes()
    assert [r["uid"] for r in rows_of(a)] == [r["uid"] for r in rows_of(b)]
    ca.close()
    cb.close()


def test_a_different_seed_gives_a_different_run(tmp_path: Path) -> None:
    _, a, ca, _ = run(tmp_path, "s0", seed=0)
    _, b, cb, _ = run(tmp_path, "s1", seed=1)
    assert [r["uid"] for r in rows_of(a)] != [r["uid"] for r in rows_of(b)]
    ca.close()
    cb.close()


def test_resume_continues_from_the_persisted_search_state(tmp_path: Path) -> None:
    config, paths, conn, first = run(tmp_path, "res", rounds=2)
    before = {r["uid"] for r in rows_of(paths)}
    ops_before = dict(first.operators.counts)

    # A fresh proposer object, as a restarted process would build it.
    second = make_proposer(conn, paths.root)
    assert len(second.archive) == len(first.archive)
    assert second.operators.counts == ops_before
    assert second.surrogate.n_train == first.surrogate.n_train
    assert len(second.rounds) == config.rounds

    run_loop(config, paths, conn, second, FakeGenerator(), StructuredJudge())
    rows = rows_of(paths)
    uids = [r["uid"] for r in rows]
    assert len(uids) == len(set(uids)), "a resumed run re-evaluated a candidate"
    assert before < set(uids)
    assert len(rows) == 2 * config.rounds * config.k
    state = json.loads((paths.root / STATE_FILENAME).read_text())
    assert len(state["rounds"]) == 2 * config.rounds
    conn.close()


# -- dependency hygiene ------------------------------------------------------


def test_search_package_does_not_need_torch() -> None:
    code = (
        "import sys; sys.modules['torch'] = None\n"
        "import ava.search.evolve, ava.search.archive, ava.search.operators\n"
        "import ava.search.novelty, ava.search.racing, ava.search.surrogate\n"
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


# -- the summary script reads only what the run wrote --------------------------


def test_search_summary_reads_the_persisted_run(tmp_path: Path) -> None:
    from scripts.search_summary import summarise, to_markdown

    config, paths, conn, proposer = run(tmp_path, "sum")
    s = summarise(paths.root, top=10)
    assert s["pairs"] == len(proposer.archive)
    assert s["evaluations"] == config.rounds * config.k
    assert set(s["coverage"]) <= set(TASKS)
    assert s["elites"] and all(i["elite"] for i in s["elites"])
    assert len(s["surrogate_log"]) == config.rounds
    text = to_markdown(s)
    for heading in (
        "## Coverage",
        "## Elites",
        "## Racing",
        "## Surrogate",
        "## Operators",
    ):
        assert heading in text
    conn.close()
