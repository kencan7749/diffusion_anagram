"""The CLAP judge: axis ordering without a model, and the Step 0a result with one.

The fast tests pin the thing that would be silently wrong forever if it were
wrong -- which axis of the score matrix is the view and which is the prompt.
Getting that backwards produces plausible numbers that mean the opposite of
what they say.

The slow test pins the measured outcome of Step 0a, so that a change of model,
of transformers version, or of the probe signals cannot quietly move the
conclusion the audio track is being built on.
"""

import numpy as np
import pytest
import torch

from ava.audio.judge import ClapJudge
from ava.audio.signals import SAMPLE_RATE, percussive_burst, symmetric_burst


class StubJudge(ClapJudge):
    """Embeddings chosen so every cell of the matrix is distinguishable.

    forward audio -> e0, reversed audio -> e1, and the two prompts sit on the
    same two axes, so S must come out as the identity if the ordering is right.
    """

    def text_emb(self, prompts: list[str]) -> torch.Tensor:
        basis = {"FWD": [1.0, 0.0], "REV": [0.0, 1.0]}
        return torch.tensor([basis[p] for p in prompts])

    def audio_emb(self, waves: list[np.ndarray]) -> torch.Tensor:
        # The reversed view is the one whose first sample is the original's last.
        return torch.tensor([[1.0, 0.0] if w[0] == 0.0 else [0.0, 1.0] for w in waves])


def test_score_matrix_axes_are_view_by_prompt() -> None:
    wave = np.array([0.0, 0.5, 1.0])
    s = StubJudge(device="cpu").score_matrix(wave, "FWD", "REV")

    assert s.shape == (2, 2)
    # Diagonal is each view against its own prompt.
    assert s[0, 0] == pytest.approx(1.0)
    assert s[1, 1] == pytest.approx(1.0)
    assert s[0, 1] == pytest.approx(0.0)
    assert s[1, 0] == pytest.approx(0.0)


def test_self_similarity_compares_a_signal_with_its_reverse() -> None:
    wave = np.array([0.0, 0.5, 1.0])
    assert StubJudge(device="cpu").self_similarity(wave) == pytest.approx(0.0)


@pytest.mark.slow
def test_clap_hears_envelope_direction_but_not_pitch_direction() -> None:
    """The Step 0a result, pinned. Needs CUDA and the CLAP weights."""
    judge = ClapJudge(sample_rate=SAMPLE_RATE)

    # A reversal nobody can hear must leave the embedding where it was.
    assert judge.self_similarity(symmetric_burst()) > 0.99

    # A decay turned into a swell must move it a long way.
    assert judge.self_similarity(percussive_burst()) < 0.70

    # And the movement must point at the right words, in both views at once.
    s = judge.score_matrix(
        percussive_burst(),
        "a percussive attack with a decay",
        "a reversed sound building up",
    )
    assert float(s[0, 0] - s[0, 1]) > 0.0
    assert float(s[1, 1] - s[1, 0]) > 0.0
