"""CLAP scoring for the reversal anagram.

Structurally identical to `ava/judge.py`: build a 2x2 matrix of
cosine similarities, S[view][prompt], with both axes ordered the same way, and
and hand it to `ava.metric.scores_to_probs`, shared with the image track:
turning a score matrix into two probabilities and their minimum is a two-way
softmax and has nothing to do with pixels.

What this measures is narrower than it looks, and the distinction matters:

  self_similarity  does the embedding move at all under reversal?
  score_matrix     does it move *towards the right words*?

The first is a precondition for the second and is far more diagnostic. CLAP's
audio tower pools over time, so it is entirely possible for a sound and its
reverse to land in nearly the same place -- in which case no choice of prompt
can rescue the score, and the failure is in the encoder, not the wording.
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ava.audio.features import to_mono
from ava.audio.perceive import forward_view, reverse_view
from ava.audio.resample import resample

CLAP_ID = "laion/clap-htsat-unfused"
# CLAP refuses audio at any other rate, and it is mono.
CLAP_SAMPLE_RATE = 48_000

__all__ = ["CLAP_ID", "ClapJudge"]


class ClapJudge:
    """Scores audio against text with CLAP.

    Runs in float32. The model is ~150 M parameters, so half precision buys
    nothing here and would only put the cosine margins -- the quantity the
    whole experiment reads -- at the mercy of rounding.
    """

    def __init__(
        self,
        device: str = "cuda",
        clap_id: str = CLAP_ID,
        sample_rate: int = CLAP_SAMPLE_RATE,
    ) -> None:
        """`sample_rate` is the rate of the audio that will be handed in.

        Anything other than CLAP's own 48 kHz is converted here rather than
        pushed back onto the caller: the generator's rate is the generator's
        business, and 44.1 kHz is what Stable Audio produces.
        """
        self.device = device
        self.clap_id = clap_id
        self.sample_rate = sample_rate

        self._model: Any = None
        self._proc: Any = None
        self._text_cache: dict[str, torch.Tensor] = {}

    # ---- lazy model loading ----------------------------------------------

    def _ensure_clap(self) -> None:
        if self._model is None:
            from transformers import ClapModel, ClapProcessor

            model = ClapModel.from_pretrained(self.clap_id)
            self._model = model.to(self.device).eval()
            self._proc = ClapProcessor.from_pretrained(self.clap_id)

    @property
    def logit_scale(self) -> float:
        """CLAP's own softmax temperature, for the audio-text pairing."""
        self._ensure_clap()
        return float(self._model.logit_scale_a.exp())

    # ---- feature extraction ----------------------------------------------

    @torch.no_grad()
    def text_emb(self, prompts: list[str]) -> torch.Tensor:
        """Normalized text embeddings (N, D), cached per string."""
        self._ensure_clap()
        missing = [p for p in prompts if p not in self._text_cache]
        if missing:
            tok = self._proc(
                text=missing, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            emb = F.normalize(self._model.get_text_features(**tok).float(), dim=-1)
            for prompt, vector in zip(missing, emb, strict=True):
                self._text_cache[prompt] = vector
        return torch.stack([self._text_cache[p] for p in prompts])

    @torch.no_grad()
    def audio_emb(self, waves: list[np.ndarray]) -> torch.Tensor:
        """Normalized audio embeddings (N, D).

        Each signal is downmixed and rate-converted first. CLAP is mono at
        48 kHz and raises rather than adapting.
        """
        self._ensure_clap()
        prepared = [
            resample(to_mono(w), self.sample_rate, CLAP_SAMPLE_RATE).astype(np.float32)
            for w in waves
        ]
        inputs = self._proc(
            audios=prepared,
            sampling_rate=CLAP_SAMPLE_RATE,
            return_tensors="pt",
        ).to(self.device)
        emb = self._model.get_audio_features(**inputs)
        return F.normalize(emb.float(), dim=-1)

    # ---- evaluation -------------------------------------------------------

    @torch.no_grad()
    def self_similarity(self, wave: np.ndarray) -> float:
        """Cosine between a signal's embedding and its reverse's.

        The prompt-free precondition. A value at 1.0 means CLAP cannot
        represent the difference at all; the control signal says what 1.0
        looks like when the sounds really are perceptually the same.
        """
        emb = self.audio_emb([forward_view(wave), reverse_view(wave)])
        return float(emb[0] @ emb[1])

    @torch.no_grad()
    def score_matrix(
        self, wave: np.ndarray, prompt_forward: str, prompt_reverse: str
    ) -> torch.Tensor:
        """S[view][prompt], both axes ordered (forward, reverse).

        The diagonal is what the anagram needs: the forward view should prefer
        the forward prompt and the reversed view the reversed prompt.
        """
        audio = self.audio_emb([forward_view(wave), reverse_view(wave)])
        text = self.text_emb([prompt_forward, prompt_reverse])
        return audio @ text.T
