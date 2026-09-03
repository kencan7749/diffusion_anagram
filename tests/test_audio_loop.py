"""The audio loop end to end, without Stable Audio or CLAP.

Stand-ins for the generator and the judge drive the real control flow: the
task registration, the audio vocabulary, both proposers (bandit and evolve),
persistence of both directions as WAV, credit assignment into the shared
database, and the audition list.
"""

from __future__ import annotations

import hashlib
import json
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from ava.audio.loop import (
    AudioLoopConfig,
    run_loop,
    to_audio_spec,
    verdict_for_wave,
    write_audition,
)
from ava.audio.spec import AudioCandidateSpec
from ava.audio.tasks import TIME_REVERSE
from ava.audio.vocab import (
    DECAY_PROMPTS,
    SOURCE_PROMPTS,
    STEP1_PROMPTS,
    SWELL_PROMPTS,
    seed_audio_vocab,
)
from ava.image.tasks import TASKS, all_tasks, get_task, register_task
from ava.loop import RunPaths
from ava.propose import BanditProposer
from ava.search.evolve import ARCHIVE_FILENAME, EvolutionaryProposer, SearchConfig
from ava.search.surrogate import SurrogateConfig
from ava.spec import CandidateSpec
from ava.vocab import connect, list_arms
from tests.search_fakes import FakeEmbedder

SEED = 0
RATE = 8_000


class FakeAudioGenerator:
    sample_rate = RATE

    def __init__(self) -> None:
        self.calls: list[AudioCandidateSpec] = []

    def generate(self, spec: AudioCandidateSpec) -> np.ndarray:
        self.calls.append(spec)
        digest = hashlib.sha1(spec.uid().encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        n = int(spec.duration_s * RATE)
        # Stereo, decaying, so forward and reverse differ audibly.
        envelope = np.exp(-np.linspace(0, 4, n))
        return (rng.normal(size=(2, n)) * envelope * 0.5).astype(np.float32)


class FakeClapJudge:
    """Margins from a hash of (prompt, direction), so some pairs reliably hold."""

    logit_scale = 33.0

    def __init__(self) -> None:
        self._embed = FakeEmbedder()

    @staticmethod
    def _margin(prompt: str, direction: str) -> float:
        h = hashlib.sha1(f"{direction}|{prompt}".encode()).digest()
        return -0.08 + 0.20 * (int.from_bytes(h[:4], "little") / 2**32)

    def score_matrix(self, wave, prompt_forward: str, prompt_reverse: str):
        a = self._margin(prompt_forward, "forward")
        b = self._margin(prompt_reverse, "reverse")
        return torch.tensor([[0.2 + a, 0.2], [0.2, 0.2 + b]])

    def self_similarity(self, wave) -> float:
        return 0.7

    def text_emb(self, prompts: list[str]) -> torch.Tensor:
        return torch.from_numpy(self._embed(prompts))


@pytest.fixture()
def wired(tmp_path: Path):
    conn = connect(tmp_path / "runs" / "vocab.db")
    seed_audio_vocab(conn)
    paths = RunPaths(tmp_path / "runs" / "a")
    yield conn, paths, FakeAudioGenerator(), FakeClapJudge()
    conn.close()


def make_evolve(conn, paths: RunPaths) -> EvolutionaryProposer:
    return EvolutionaryProposer(
        conn,
        np.random.default_rng(SEED),
        FakeEmbedder(),
        tasks=[TIME_REVERSE.name],
        cfg=SearchConfig(
            clusters=4,
            children_per_slot=8,
            surrogate=SurrogateConfig(min_train=8, window=2),
        ),
        screening_seed=0,
        guidance_scale=7.0,
        num_inference_steps=100,
        state_dir=paths.root,
    )


def rows_of(paths: RunPaths) -> list[dict]:
    out = []
    for p in paths.all_scores():
        out += [json.loads(line) for line in p.read_text().splitlines() if line]
    return out


# -- registration and vocabulary --------------------------------------------


def test_time_reverse_is_a_registered_symmetric_two_view_task() -> None:
    task = get_task("time_reverse")
    assert task is TIME_REVERSE
    assert task.n_views == 2 and task.symmetric and task.ref_slot is None
    assert task.slot_names == ["forward", "reverse"]
    # It is resolvable, but not one of the papers' example types.
    assert "time_reverse" not in TASKS and "time_reverse" in all_tasks()
    register_task(TIME_REVERSE)  # idempotent
    with pytest.raises(ValueError):
        register_task(replace(TIME_REVERSE, citation="something else"))


def test_audio_vocab_seeds_every_authored_prompt_once(tmp_path: Path) -> None:
    conn = connect(tmp_path / "vocab.db")
    added = seed_audio_vocab(conn)
    authored = STEP1_PROMPTS + DECAY_PROMPTS + SWELL_PROMPTS + SOURCE_PROMPTS
    expected = len(authored) + 1
    assert added == expected
    assert seed_audio_vocab(conn) == 0
    arms = list_arms(conn, "time_reverse", "subject")
    assert {a.source for a in arms} == {"authored"}
    assert all(a.citation for a in arms if a.word in STEP1_PROMPTS)
    assert len({a.word for a in arms}) == expected - 1
    conn.close()


def test_audio_and_image_vocabularies_share_a_database(tmp_path: Path) -> None:
    from ava.vocab import seed_author_vocab

    conn = connect(tmp_path / "vocab.db")
    seed_author_vocab(conn)
    seed_audio_vocab(conn)
    assert list_arms(conn, "flip", "subject")
    assert list_arms(conn, "time_reverse", "subject")
    conn.close()


# -- translation ------------------------------------------------------------


def test_search_spec_translates_to_the_generator_spec() -> None:
    config = AudioLoopConfig(run_id="x", duration_s=5.0, negative_prompt="noisy")
    spec = CandidateSpec(
        "time_reverse",
        ("a bell", "a swell"),
        "",
        seed=3,
        guidance_scale=7.0,
        num_inference_steps=50,
    )
    audio = to_audio_spec(spec, config)
    assert audio.prompt_forward == "a bell" and audio.prompt_reverse == "a swell"
    assert audio.seed == 3 and audio.duration_s == 5.0
    assert audio.num_inference_steps == 50 and audio.negative_prompt == "noisy"
    with pytest.raises(ValueError):
        to_audio_spec(CandidateSpec("flip", ("a", "b")), config)


def test_verdict_from_the_clap_matrix_matches_the_two_view_arithmetic() -> None:
    spec = CandidateSpec("time_reverse", ("a bell", "a swell"), "")
    matrix = torch.tensor([[0.30, 0.20], [0.20, 0.28]])
    v = verdict_for_wave(spec, matrix, 33.0, 0.6)
    assert v.slots == ["forward", "reverse"]
    assert v.sep == pytest.approx([0.10, 0.08])
    assert v.holds == [True, True] and v.diagnose() == "ok"
    assert v.j == pytest.approx(1 / (1 + np.exp(-33.0 * 0.08)), abs=1e-6)
    assert v.extra["self_similarity"] == 0.6


# -- the loop with each proposer ----------------------------------------------


def test_evolve_run_writes_wavs_scores_archive_and_audition(wired) -> None:
    conn, paths, engine, judge = wired
    config = AudioLoopConfig(run_id="a", rounds=3, k=6, proposer="evolve")
    run_loop(config, paths, conn, make_evolve(conn, paths), engine, judge)

    rows = rows_of(paths)
    assert len(rows) == config.rounds * config.k
    assert (paths.root / ARCHIVE_FILENAME).exists()
    assert (paths.root / "audition.md").exists()
    assert paths.components.exists()
    for r in rows:
        assert r["task"] == "time_reverse"
        assert len(r["prompts"]) == 2 and r["prompts"][0] != r["prompts"][1]
        for direction in ("forward", "reverse"):
            path = Path(r["wav_paths"][direction])
            assert path.exists()
            with wave.open(str(path)) as w:
                assert w.getnchannels() == 2 and w.getframerate() == RATE
        assert r["self_similarity"] == pytest.approx(0.7)
        assert "seconds" in r and "extra" in r
    origins = {r["origin"] for r in rows}
    assert "bootstrap" in origins and ("evolve" in origins or "race" in origins)


def test_reverse_wav_is_the_forward_wav_played_backwards(wired) -> None:
    conn, paths, engine, judge = wired
    config = AudioLoopConfig(run_id="a", rounds=1, k=2, proposer="evolve")
    run_loop(config, paths, conn, make_evolve(conn, paths), engine, judge)
    row = rows_of(paths)[0]

    def pcm(path: str) -> np.ndarray:
        with wave.open(path) as w:
            data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        return data.reshape(-1, 2).T

    fwd, rev = pcm(row["wav_paths"]["forward"]), pcm(row["wav_paths"]["reverse"])
    assert np.array_equal(rev, fwd[:, ::-1])


def test_bandit_run_assigns_credit_to_the_audio_arms(wired) -> None:
    conn, paths, engine, judge = wired
    config = AudioLoopConfig(run_id="a", rounds=2, k=6, proposer="bandit")
    proposer = BanditProposer(
        conn,
        np.random.default_rng(SEED),
        tasks=["time_reverse"],
        mix=config.mix(),
        guidance_scale=7.0,
        num_inference_steps=100,
    )
    run_loop(config, paths, conn, proposer, engine, judge)
    tried = [a for a in list_arms(conn, "time_reverse") if a.n_trials > 0]
    assert tried
    assert all(s.duration_s == config.duration_s for s in engine.calls)
    assert all(s.num_inference_steps == 100 for s in engine.calls)
    written = paths.config.read_text()
    assert "proposer: bandit" in written and "duration_s: 5.0" in written


def test_audition_lists_held_candidates_best_first_and_counts_the_rest(
    tmp_path,
) -> None:
    rows = [
        {
            "j": 0.9,
            "sep_min": 0.05,
            "prompts": ["a", "b"],
            "seed": 0,
            "origin": "x",
            "wav_paths": {"forward": "f0", "reverse": "r0"},
            "self_similarity": 0.5,
        },
        {
            "j": 0.99,
            "sep_min": 0.12,
            "prompts": ["c", "d"],
            "seed": 1,
            "origin": "x",
            "wav_paths": {"forward": "f1", "reverse": "r1"},
            "self_similarity": 0.6,
        },
        {
            "j": 0.1,
            "sep_min": -0.05,
            "prompts": ["e", "f"],
            "seed": 0,
            "origin": "x",
            "wav_paths": {"forward": "f2", "reverse": "r2"},
            "self_similarity": 0.9,
        },
    ]
    text = write_audition(rows, tmp_path / "audition.md", min_j=0.5).read_text()
    assert "2 of 3 candidates listed" in text and "1 hidden" in text
    assert text.index("| c | d |") < text.index("| a | b |")
    assert "| e | f |" not in text
    everything = write_audition(rows, tmp_path / "all.md", min_j=None).read_text()
    assert "| e | f |" in everything


# ---- which latent the anagram is sampled in ----------------------------------


def test_backend_is_a_run_setting_with_stable_audio_as_default() -> None:
    from dataclasses import asdict

    from ava.audio.loop import BACKENDS, AudioLoopConfig

    config = AudioLoopConfig(run_id="x")
    assert config.backend == "stable_audio"
    assert asdict(config)["backend"] == "stable_audio"
    assert asdict(config)["model_id"] is None
    assert "audioldm2" in BACKENDS


def test_unknown_backend_is_rejected_before_anything_loads() -> None:
    from ava.audio.loop import AudioLoopConfig, build_engine

    with pytest.raises(ValueError, match="backend"):
        AudioLoopConfig(run_id="x", backend="audiogen")
    with pytest.raises(ValueError, match="backend"):
        build_engine("audiogen", "cpu")


def test_backend_flag_reaches_the_config() -> None:
    from ava.audio.loop import build_parser

    args = build_parser().parse_args(
        ["--backend", "audioldm2", "--model-id", "cvssp/audioldm2-large"]
    )
    assert args.backend == "audioldm2"
    assert args.model_id == "cvssp/audioldm2-large"
    assert build_parser().parse_args([]).backend == "stable_audio"


def test_backend_does_not_enter_the_candidate_uid() -> None:
    """Runs on different backends are compared by config.yaml, not by uid."""
    from ava.audio.loop import AudioLoopConfig, to_audio_spec

    spec = CandidateSpec(
        task=TIME_REVERSE.name, prompts=("a hit that decays", "a swell that stops")
    )
    a = to_audio_spec(spec, AudioLoopConfig(run_id="a", backend="stable_audio"))
    b = to_audio_spec(spec, AudioLoopConfig(run_id="b", backend="audioldm2"))
    assert a.uid() == b.uid()
