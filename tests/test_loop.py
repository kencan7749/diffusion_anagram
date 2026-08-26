"""End-to-end orchestration, run without a GPU.

The Generator and Judge protocols exist precisely so this test can drive the
whole loop -- proposal, generation, scoring, credit assignment, persistence,
resume and reporting -- with stand-ins, and still exercise the real control
flow, the real directory layout and the real report code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ava.judge import to_pil  # noqa: F401  (kept: asserts the import graph is sane)
from ava.loop import LoopConfig, RunPaths, run_loop
from ava.propose import BanditProposer
from ava.spec import CandidateSpec, Verdict
from ava.vocab import connect, list_arms, seed_author_vocab

SEED = 0
SIZE = 32


class FakeGenerator:
    """Returns a deterministic image whose content depends only on the spec."""

    def __init__(self) -> None:
        self.calls: list[CandidateSpec] = []

    def generate(self, spec, sigma=2.0, kernel_size=33):
        self.calls.append(spec)
        rng = np.random.default_rng(abs(hash(spec.uid())) % (2**32))
        img = torch.from_numpy(rng.random((3, SIZE, SIZE), dtype=np.float32))
        return img, img

    def upscale_1024(self, spec, image_256):
        return image_256


class FakeJudge:
    """Scores by a fixed rule so credit assignment is checkable.

    `a panda` as the low word always survives blurring; `houseplants` as the
    high word never appears. Everything else lands mid-range.
    """

    def views(self, img):
        return img * 0.5, img

    def evaluate(self, img, spec) -> Verdict:
        p_far = 0.95 if spec.prompt_low == "a panda" else 0.4
        p_near = 0.02 if spec.prompt_high == "houseplants" else 0.6
        return Verdict(
            uid=spec.uid(),
            s_far_low=0.3,
            s_far_high=0.2,
            s_near_low=0.2,
            s_near_high=0.3,
            p_far=p_far,
            p_near=p_near,
            j=min(p_far, p_near),
            caption_far="a far view",
            caption_near="a near view",
        )


@pytest.fixture()
def wired(tmp_path: Path):
    conn = connect(tmp_path / "runs" / "vocab.db")
    seed_author_vocab(conn)
    config = LoopConfig(run_id="t", rounds=2, k=4, harvest_seeds=2)
    paths = RunPaths(tmp_path / "runs" / "t")
    proposer = BanditProposer(
        conn, np.random.default_rng(SEED), mix=config.mix()
    )
    yield config, paths, conn, proposer, FakeGenerator(), FakeJudge()
    conn.close()


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


def test_every_candidate_is_scored_and_nothing_is_discarded(wired) -> None:
    """Thresholds must be re-drawable later, so no candidate may be dropped."""
    config, paths, conn, proposer, engine, judge = wired
    run_loop(config, paths, conn, proposer, engine, judge)

    for i in range(config.rounds):
        d = paths.round_dir(i)
        proposed = (d / "candidates.jsonl").read_text().strip().splitlines()
        scored = (d / "scores.jsonl").read_text().strip().splitlines()
        assert len(scored) == len(proposed) == config.k


def test_scores_carry_everything_a_figure_needs(wired) -> None:
    """A report must be buildable from scores.jsonl alone."""
    config, paths, conn, proposer, engine, judge = wired
    run_loop(config, paths, conn, proposer, engine, judge)

    row = json.loads((paths.round_dir(0) / "scores.jsonl").read_text().splitlines()[0])
    for key in (
        "j", "p_far", "p_near", "sep_far", "sep_near", "diagnosis",
        "prompt_low", "prompt_high", "style", "seed", "origin", "detail",
        "image_path", "far_image_path", "round",
    ):
        assert key in row, f"scores.jsonl is missing {key!r}"
    assert Path(row["image_path"]).exists()
    assert Path(row["far_image_path"]).exists()


def test_credit_reaches_the_database_and_separates_components(wired) -> None:
    config, paths, conn, proposer, engine, judge = wired
    run_loop(config, paths, conn, proposer, engine, judge)

    tried = [a for a in list_arms(conn) if a.n_trials > 0]
    assert tried, "the run must have taught the vocabulary something"

    panda = next((a for a in list_arms(conn, "low") if a.word == "a panda"), None)
    plants = next((a for a in list_arms(conn, "high") if a.word == "houseplants"), None)
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

    uids = [
        json.loads(line)["uid"]
        for path in paths.all_scores()
        for line in path.read_text().strip().splitlines()
    ]
    assert len(uids) == len(set(uids)), "a resumed run re-evaluated a candidate"


def test_config_records_the_fixed_sigma(wired) -> None:
    """Sigma is not a search axis; the run must still record what it used."""
    import yaml

    config, paths, conn, proposer, engine, judge = wired
    run_loop(config, paths, conn, proposer, engine, judge)
    written = yaml.safe_load(paths.config.read_text())
    assert written["sigma"] == 2.0
    assert written["k"] == config.k


def test_components_table_reports_untried_arms(wired) -> None:
    config, paths, conn, proposer, engine, judge = wired
    run_loop(config, paths, conn, proposer, engine, judge)
    text = paths.components.read_text()
    assert "never tried" in text
    for role in ("low", "high", "style"):
        assert f"role = `{role}`" in text


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
    wide = build_contact_sheet(rows, paths.root / "all.png", min_j=None)
    narrow = build_contact_sheet(rows, paths.root / "held.png", min_j=0.5)
    assert Image.open(wide).size[1] > Image.open(narrow).size[1]


def test_contact_sheet_refuses_to_hide_everything(wired) -> None:
    """Silently emitting an empty sheet would read as 'nothing was found'."""
    from ava.report import build_contact_sheet, load_scores

    config, paths, conn, proposer, engine, judge = wired
    run_loop(config, paths, conn, proposer, engine, judge)
    rows = [r for p in paths.all_scores() for r in load_scores(p)]

    with pytest.raises(ValueError):
        build_contact_sheet(rows, paths.root / "empty.png", min_j=1.01)


def test_harvest_cutoff_is_reported_not_silent(wired, capsys) -> None:
    """A cutoff decides where GPU time goes, so it must say what it skipped."""
    config, paths, conn, proposer, engine, judge = wired
    config.harvest_top = 1
    run_loop(config, paths, conn, proposer, engine, judge)

    out = capsys.readouterr().out
    assert "cutoff at top 1" in out
    assert "highest skipped sep" in out

    harvested = {
        (r["prompt_low"], r["prompt_high"], r["style"])
        for r in load_json_lines(paths.harvest / "scores.jsonl")
    }
    assert len(harvested) == 1


def test_harvest_can_be_told_to_cover_every_pair(wired) -> None:
    config, paths, conn, proposer, engine, judge = wired
    config.harvest_top = None
    run_loop(config, paths, conn, proposer, engine, judge)

    screened = {
        (r["prompt_low"], r["prompt_high"], r["style"])
        for p in paths.all_scores()
        for r in load_json_lines(p)
    }
    harvested = {
        (r["prompt_low"], r["prompt_high"], r["style"])
        for r in load_json_lines(paths.harvest / "scores.jsonl")
    }
    assert harvested == screened


def load_json_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().strip().splitlines()]


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
