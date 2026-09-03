"""End-to-end orchestration, run without a GPU.

The Generator and Judge protocols exist precisely so this test can drive the
whole loop -- proposal, generation, scoring, credit assignment, persistence,
resume and reporting -- with stand-ins, and still exercise the real control
flow, the real directory layout and the real report code, across several
illusion tasks at once.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ava.image.judge import to_pil  # noqa: F401  (asserts the import graph is sane)
from ava.image.views import ViewSet
from ava.loop import LoopConfig, RunPaths, candidate_key, run_loop
from ava.propose import BanditProposer
from ava.spec import CandidateSpec, Verdict
from ava.vocab import connect, list_arms, seed_author_vocab

SEED = 0
# 64 px: the smallest size the permutation views accept.
SIZE = 64
TASKS = ["hybrid", "flip", "patch_permute"]


class FakeGenerator:
    """Returns a deterministic image whose content depends only on the spec."""

    def __init__(self) -> None:
        self.calls: list[CandidateSpec] = []

    def generate(self, spec, viewset):
        self.calls.append(spec)
        rng = np.random.default_rng(abs(hash(spec.uid())) % (2**32))
        img = torch.from_numpy(rng.random((3, SIZE, SIZE), dtype=np.float32))
        return img, img

    def upscale_1024(self, spec, image_256, viewset):
        return image_256


class FakeJudge:
    """Scores by a fixed rule so credit assignment is checkable.

    `a panda` always reads in its view; the words in FAILING never do, so every
    task produces some failures. Everything else lands mid-range. The views are
    the real perceptual views, so the loop persists exactly what a real judge
    would have looked at.
    """

    FAILING = frozenset({"houseplants", "a horse", "a rabbit", "a duck"})

    def views(self, img, viewset: ViewSet):
        return viewset.perceive(img)

    def evaluate(self, img, spec, viewset: ViewSet) -> Verdict:
        p = []
        for word in spec.prompts:
            p.append(
                0.95 if word == "a panda" else 0.02 if word in self.FAILING else 0.6
            )
        n = len(p)
        scores = [[0.2] * n for _ in range(n)]
        for i, pi in enumerate(p):
            scores[i][i] = 0.3 if pi > 0.5 else 0.1
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


@pytest.fixture()
def wired(tmp_path: Path):
    conn = connect(tmp_path / "runs" / "vocab.db")
    seed_author_vocab(conn)
    config = LoopConfig(run_id="t", tasks=list(TASKS), rounds=2, k=6, harvest_seeds=2)
    paths = RunPaths(tmp_path / "runs" / "t")
    proposer = BanditProposer(
        conn, np.random.default_rng(SEED), tasks=config.tasks, mix=config.mix()
    )
    yield config, paths, conn, proposer, FakeGenerator(), FakeJudge()
    conn.close()


def load_json_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().strip().splitlines()]


def test_run_produces_the_documented_layout(wired) -> None:
    config, paths, conn, proposer, engine, judge = wired
    run_loop(config, paths, conn, proposer, engine, judge)

    assert paths.config.exists()
    assert paths.state.exists()
    assert paths.components.exists()
    assert paths.contact_sheet.exists()
    for i in range(config.rounds):
        d = paths.round_dir(i)
        assert (d / "candidates.jsonl").exists()
        assert (d / "scores.jsonl").exists()
        assert (d / "report.md").exists()
    assert (paths.harvest / "scores.jsonl").exists()
    # The randomised task's permutation is on disk; the deterministic ones are not.
    assert paths.view_state("patch_permute").exists()
    assert not paths.view_state("flip").exists()


def test_every_candidate_is_scored_and_nothing_is_discarded(wired) -> None:
    """Thresholds must be re-drawable later, so no candidate may be dropped."""
    config, paths, conn, proposer, engine, judge = wired
    run_loop(config, paths, conn, proposer, engine, judge)

    for i in range(config.rounds):
        d = paths.round_dir(i)
        proposed = load_json_lines(d / "candidates.jsonl")
        scored = load_json_lines(d / "scores.jsonl")
        assert len(scored) == len(proposed) == config.k


def test_every_configured_task_is_exercised(wired) -> None:
    config, paths, conn, proposer, engine, judge = wired
    run_loop(config, paths, conn, proposer, engine, judge)
    tasks = {r["task"] for p in paths.all_scores() for r in load_json_lines(p)}
    assert tasks == set(TASKS)


def test_scores_carry_everything_a_figure_needs(wired) -> None:
    """A report must be buildable from scores.jsonl alone."""
    config, paths, conn, proposer, engine, judge = wired
    run_loop(config, paths, conn, proposer, engine, judge)

    for row in load_json_lines(paths.round_dir(0) / "scores.jsonl"):
        for key in (
            "j",
            "p",
            "sep",
            "sep_min",
            "scores",
            "alignment",
            "concealment",
            "diagnosis",
            "task",
            "slots",
            "prompts",
            "style",
            "seed",
            "origin",
            "detail",
            "image_path",
            "view_paths",
            "round",
        ):
            assert key in row, f"scores.jsonl is missing {key!r}"
        assert Path(row["image_path"]).exists()
        assert len(row["view_paths"]) == len(row["slots"]) == len(row["prompts"])
        for slot, path in zip(row["slots"], row["view_paths"], strict=True):
            assert Path(path).name == f"view_{slot}.png"
            assert Path(path).exists()


def test_persisted_views_are_what_the_judge_saw(wired) -> None:
    """The flip's second view on disk must be the flipped image, not a copy."""
    config, paths, conn, proposer, engine, judge = wired
    run_loop(config, paths, conn, proposer, engine, judge)
    flips = [
        r for p in paths.all_scores() for r in load_json_lines(p) if r["task"] == "flip"
    ]
    assert flips
    row = flips[0]
    identity = np.asarray(Image.open(row["view_paths"][0]))
    flipped = np.asarray(Image.open(row["view_paths"][1]))
    assert np.array_equal(flipped, identity[::-1])


def test_credit_reaches_the_database_and_separates_components(wired) -> None:
    config, paths, conn, proposer, engine, judge = wired
    run_loop(config, paths, conn, proposer, engine, judge)

    tried = [a for a in list_arms(conn) if a.n_trials > 0]
    assert tried, "the run must have taught the vocabulary something"
    assert {a.task for a in tried} == set(TASKS)

    panda = next(
        (a for a in list_arms(conn, "hybrid", "low") if a.word == "a panda"), None
    )
    plants = next(
        (a for a in list_arms(conn, "hybrid", "high") if a.word == "houseplants"), None
    )
    if panda is not None and panda.n_trials:
        assert panda.mean > 0.6
    if plants is not None and plants.n_trials:
        assert plants.mean < 0.5


def test_screening_holds_the_diffusion_seed_fixed(wired) -> None:
    """Screening controls for noise; seed variation is bought back at harvest."""
    config, paths, conn, proposer, engine, judge = wired
    run_loop(config, paths, conn, proposer, engine, judge)

    screening = [s for s in engine.calls if s.seed == config.screening_seed]
    harvest_seeds = {s.seed for s in engine.calls} - {config.screening_seed}
    assert len(screening) >= config.rounds * config.k
    assert harvest_seeds, "harvest must explore seeds the screening pass did not"


def test_resume_continues_instead_of_restarting(wired) -> None:
    config, paths, conn, proposer, engine, judge = wired
    run_loop(config, paths, conn, proposer, engine, judge)
    first_uids = set(json.loads(paths.state.read_text())["evaluated_uids"])
    assert len(first_uids) == config.rounds * config.k

    # A second invocation resumes from state.json and adds fresh rounds.
    run_loop(config, paths, conn, proposer, engine, judge)
    state = json.loads(paths.state.read_text())
    assert state["round_index"] == 2 * config.rounds
    assert paths.round_dir(2 * config.rounds - 1).exists()
    assert first_uids <= set(state["evaluated_uids"])


def test_resumed_rounds_do_not_repeat_earlier_candidates(wired) -> None:
    config, paths, conn, proposer, engine, judge = wired
    run_loop(config, paths, conn, proposer, engine, judge)
    run_loop(config, paths, conn, proposer, engine, judge)

    uids = [r["uid"] for p in paths.all_scores() for r in load_json_lines(p)]
    assert len(uids) == len(set(uids)), "a resumed run re-evaluated a candidate"


def test_resume_reuses_the_persisted_permutation(wired, tmp_path: Path) -> None:
    """A randomised view must not be redrawn on resume, or old images stop matching."""
    config, paths, conn, proposer, engine, judge = wired
    run_loop(config, paths, conn, proposer, engine, judge)
    saved = torch.load(paths.view_state("patch_permute"))["patch_permute"]

    # Even asking for a different view seed must not change a run already started.
    config.view_seed = 99
    run_loop(config, paths, conn, proposer, engine, judge)
    again = torch.load(paths.view_state("patch_permute"))["patch_permute"]
    assert torch.equal(saved, again)
    fresh = ViewSet.build(proposer.tasks[2], seed=99).state()["patch_permute"]
    assert not torch.equal(fresh, saved)


def test_config_records_the_fixed_view_parameters(wired) -> None:
    """View parameters are not a search axis; the run must still record them."""
    import yaml

    config, paths, conn, proposer, engine, judge = wired
    run_loop(config, paths, conn, proposer, engine, judge)
    written = yaml.safe_load(paths.config.read_text())
    assert written["view_params"]["sigma"] == 2.0
    assert written["view_params"]["patch_grid"] == 8
    assert written["tasks"] == TASKS
    assert written["view_seed"] == 0
    assert written["k"] == config.k


def test_components_table_is_sectioned_by_task_and_role(wired) -> None:
    config, paths, conn, proposer, engine, judge = wired
    run_loop(config, paths, conn, proposer, engine, judge)
    text = paths.components.read_text()
    assert "never tried" in text
    for task, role in (
        ("hybrid", "low"),
        ("hybrid", "high"),
        ("hybrid", "style"),
        ("flip", "subject"),
        ("patch_permute", "style"),
    ):
        assert f"task = `{task}`, role = `{role}`" in text


# -- J screens downstream, but never destroys data ----------------------


def test_raw_scores_keep_every_candidate_regardless_of_the_cutoff(wired) -> None:
    """The cutoff spends GPU time; it must never remove evidence.

    Keeping the full record is what lets the threshold be moved later and the
    sheet redrawn without regenerating anything.
    """
    from ava.report import load_scores

    config, paths, conn, proposer, engine, judge = wired
    config.min_j = 0.5
    run_loop(config, paths, conn, proposer, engine, judge)

    rows = [r for p in paths.all_scores() for r in load_scores(p)]
    assert len(rows) == config.rounds * config.k
    assert any(float(r["j"]) < config.min_j for r in rows), "fixture must fail some"


def test_contact_sheet_hides_the_low_j_tail(wired) -> None:
    from ava.report import build_contact_sheet, load_scores

    config, paths, conn, proposer, engine, judge = wired
    run_loop(config, paths, conn, proposer, engine, judge)
    rows = [r for p in paths.all_scores() for r in load_scores(p)]

    # Redrawing at a different threshold must not need a single regeneration.
    # One column, so the sheet's height counts the candidates it shows.
    wide = build_contact_sheet(rows, paths.root / "all.png", min_j=None, columns=1)
    narrow = build_contact_sheet(rows, paths.root / "held.png", min_j=0.5, columns=1)
    assert Image.open(wide).size[1] > Image.open(narrow).size[1]


def test_contact_sheet_refuses_to_hide_everything(wired) -> None:
    """Silently emitting an empty sheet would read as 'nothing was found'."""
    from ava.report import build_contact_sheet, load_scores

    config, paths, conn, proposer, engine, judge = wired
    run_loop(config, paths, conn, proposer, engine, judge)
    rows = [r for p in paths.all_scores() for r in load_scores(p)]

    with pytest.raises(ValueError):
        build_contact_sheet(rows, paths.root / "empty.png", min_j=1.01)


def test_contact_sheet_stacks_every_view_of_a_cell(wired) -> None:
    """A three-view candidate needs a taller cell than a two-view one."""
    from ava.report import build_contact_sheet, load_scores

    config, paths, conn, proposer, engine, judge = wired
    run_loop(config, paths, conn, proposer, engine, judge)
    rows = [r for p in paths.all_scores() for r in load_scores(p)]
    two = build_contact_sheet(rows[:1], paths.root / "two.png", min_j=None, columns=1)

    three = dict(rows[0])
    three.update(
        task="three_view",
        slots=["identity", "rotate_cw", "rotate_ccw"],
        prompts=["a", "b", "c"],
        view_paths=[rows[0]["view_paths"][0]] * 3,
    )
    tall = build_contact_sheet([three], paths.root / "three.png", min_j=None, columns=1)
    assert Image.open(tall).size[1] > Image.open(two).size[1]


def test_harvest_cutoff_is_reported_not_silent(wired, capsys) -> None:
    """A cutoff decides where GPU time goes, so it must say what it skipped."""
    config, paths, conn, proposer, engine, judge = wired
    config.harvest_top = 1
    run_loop(config, paths, conn, proposer, engine, judge)

    out = capsys.readouterr().out
    assert "cutoff at top 1" in out
    assert "highest skipped sep" in out

    harvested = {
        candidate_key(r) for r in load_json_lines(paths.harvest / "scores.jsonl")
    }
    assert len(harvested) == 1


def test_harvest_can_be_told_to_cover_every_candidate(wired) -> None:
    config, paths, conn, proposer, engine, judge = wired
    config.harvest_top = None
    run_loop(config, paths, conn, proposer, engine, judge)

    screened = {
        candidate_key(r) for p in paths.all_scores() for r in load_json_lines(p)
    }
    harvested = {
        candidate_key(r) for r in load_json_lines(paths.harvest / "scores.jsonl")
    }
    assert harvested == screened


def test_harvest_top_zero_skips_harvesting_without_crashing(wired, capsys) -> None:
    """`--harvest-top 0` is the natural way to say "screen only".

    Regression: the cutoff branch reported the lowest kept J by indexing the
    kept list, which is empty when nothing is kept. It crashed only after every
    generation had already been paid for.
    """
    config, paths, conn, proposer, engine, judge = wired
    config.harvest_top = 0
    run_loop(config, paths, conn, proposer, engine, judge)

    out = capsys.readouterr().out
    assert "nothing to harvest" in out
    assert "cutoff at top 0" in out  # it still says what it skipped
    assert not (paths.harvest / "scores.jsonl").exists()
    # Screening results and reports must still be complete.
    assert paths.contact_sheet.exists()
    assert paths.components.exists()


def test_each_candidate_directory_explains_itself(wired) -> None:
    """A directory named by a content hash says nothing on its own.

    Cross-referencing scores.jsonl to find out what an image was is a step
    nobody takes, so the prompts sit beside the images.
    """
    config, paths, conn, proposer, engine, judge = wired
    run_loop(config, paths, conn, proposer, engine, judge)

    for row in load_json_lines(paths.round_dir(0) / "scores.jsonl"):
        card = Path(row["image_path"]).parent / "prompt.txt"
        assert card.exists(), f"{card} missing"
        text = card.read_text()
        assert f"task   : {row['task']}" in text
        for prompt in row["prompts"]:
            assert prompt in text
        assert row["uid"] in text
        assert row["diagnosis"] in text


def test_harvested_candidates_get_a_card_too(wired) -> None:
    config, paths, conn, proposer, engine, judge = wired
    run_loop(config, paths, conn, proposer, engine, judge)

    for row in load_json_lines(paths.harvest / "scores.jsonl"):
        card = Path(row["image_path"]).parent / "prompt.txt"
        assert card.exists()
        assert f"seed   : {row['seed']}" in card.read_text()


def test_card_is_written_before_generation(wired, tmp_path: Path) -> None:
    """An interrupted candidate must still say what it was attempting."""
    from ava.loop import write_prompt_card

    spec = CandidateSpec(
        "three_view", ("a panda", "a barn", "a duck"), "an oil painting of", seed=3
    )
    card = write_prompt_card(tmp_path / "uid", spec, origin="inject")
    text = card.read_text()
    assert "a panda" in text and "a barn" in text and "a duck" in text
    assert "rotate_ccw" in text
    assert "an oil painting of a duck" in text
    assert "seed   : 3" in text
    # No verdict yet, and that must not break the format.
    assert "J      :" not in text


def test_backfill_is_idempotent_and_reads_only_scores(wired) -> None:
    from scripts.backfill_prompts import backfill

    config, paths, conn, proposer, engine, judge = wired
    run_loop(config, paths, conn, proposer, engine, judge)

    for card in paths.root.rglob("prompt.txt"):
        card.unlink()
    written, _ = backfill(paths.root)
    assert written == config.rounds * config.k + len(
        load_json_lines(paths.harvest / "scores.jsonl")
    )

    again, skipped = backfill(paths.root)
    assert again == 0 and skipped > 0
