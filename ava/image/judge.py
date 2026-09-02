"""Judge: measures J, the degree to which the illusion holds.

J is NOT a measure of how interesting an image is. It is a necessary condition
for the illusion working at all: it says whether every prompt is legible in its
own view, nothing more.

J orders results and drives the diagnosis that steers the next round's
proposals. It does not discard anything. Every candidate is scored, recorded
and drawn, so the line between good and bad stays a human's to draw and can be
redrawn later from the saved scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ava.image.views import ViewSet
from ava.metric import alignment, concealment, multiway_probs
from ava.spec import CandidateSpec, Verdict

CLIP_ID = "openai/clip-vit-large-patch14"
BLIP_ID = "Salesforce/blip-image-captioning-large"


def load_image(path: str | Path) -> torch.Tensor:
    """Read a PNG as (C, H, W) float in [0, 1]."""
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def to_pil(img: torch.Tensor) -> Image.Image:
    """(C, H, W) float in [0, 1] -> PIL."""
    arr = img.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((arr * 255).round().astype(np.uint8))


class Judge(Protocol):
    def evaluate(
        self, img: torch.Tensor, spec: CandidateSpec, viewset: ViewSet
    ) -> Verdict: ...

    def views(self, img: torch.Tensor, viewset: ViewSet) -> list[torch.Tensor]:
        """The images, one per slot, that evaluate() scores.

        Part of the protocol because the loop persists them alongside the
        scores; that is what lets figures be redrawn without a recompute.
        """
        ...


def build_verdict(
    spec: CandidateSpec,
    viewset: ViewSet,
    s: torch.Tensor,
    logit_scale: float,
    captions: list[str],
    extra: dict[str, Any],
) -> Verdict:
    """Turn an NxN score matrix into a Verdict. Model-free, so it is testable."""
    p, j = multiway_probs(s, logit_scale)
    return Verdict(
        uid=spec.uid(),
        task=spec.task,
        slots=viewset.task.slot_names,
        scores=[[float(x) for x in row] for row in s.tolist()],
        p=p,
        j=j,
        alignment=alignment(s),
        concealment=concealment(s, logit_scale),
        captions=captions,
        extra={"logit_scale": logit_scale, **extra},
    )


class ClipBlipJudge:
    """Scores with CLIP, records evidence text with BLIP.

    BLIP captions never enter the score. CLIP's text-to-text similarity is
    weak, so turning caption/prompt similarity into a number would not be
    trustworthy. Captions are kept as report evidence and as material to hand
    to a VLM later.
    """

    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        clip_id: str = CLIP_ID,
        blip_id: str = BLIP_ID,
        use_blip: bool = True,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.clip_id = clip_id
        self.blip_id = blip_id
        self.use_blip = use_blip

        self._clip: Any = None
        self._clip_proc: Any = None
        self._blip: Any = None
        self._blip_proc: Any = None
        self._text_cache: dict[str, torch.Tensor] = {}

    # ---- lazy model loading ----------------------------------------------

    def _ensure_clip(self) -> None:
        if self._clip is None:
            from transformers import CLIPModel, CLIPProcessor

            self._clip = (
                CLIPModel.from_pretrained(self.clip_id, torch_dtype=self.dtype)
                .to(self.device)
                .eval()
            )
            self._clip_proc = CLIPProcessor.from_pretrained(self.clip_id)

    def _ensure_blip(self) -> None:
        if self._blip is None:
            from transformers import BlipForConditionalGeneration, BlipProcessor

            self._blip = (
                BlipForConditionalGeneration.from_pretrained(
                    self.blip_id, torch_dtype=self.dtype
                )
                .to(self.device)
                .eval()
            )
            self._blip_proc = BlipProcessor.from_pretrained(self.blip_id)

    # ---- feature extraction ----------------------------------------------

    @torch.no_grad()
    def text_embeddings(self, prompts: list[str]) -> torch.Tensor:
        """Normalized text embeddings (N, D), cached per string.

        Public because the search layer reuses these embeddings as features;
        computing them here means the judge and the proposer see one CLIP.
        """
        self._ensure_clip()
        missing = [p for p in prompts if p not in self._text_cache]
        if missing:
            tok = self._clip_proc(
                text=missing, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            emb = self._clip.get_text_features(**tok)
            emb = F.normalize(emb.float(), dim=-1)
            for p, e in zip(missing, emb, strict=True):
                self._text_cache[p] = e
        return torch.stack([self._text_cache[p] for p in prompts])

    @torch.no_grad()
    def image_embeddings(self, pils: list[Image.Image]) -> torch.Tensor:
        """Normalized image embeddings (N, D)."""
        self._ensure_clip()
        px = self._clip_proc(images=pils, return_tensors="pt")["pixel_values"]
        px = px.to(self.device, self.dtype)
        emb = self._clip.get_image_features(pixel_values=px)
        return F.normalize(emb.float(), dim=-1)

    @torch.no_grad()
    def caption(self, pil: Image.Image) -> str:
        self._ensure_blip()
        inputs = self._blip_proc(images=pil, return_tensors="pt").to(
            self.device, self.dtype
        )
        out = self._blip.generate(**inputs, max_new_tokens=30, num_beams=3)
        return str(self._blip_proc.decode(out[0], skip_special_tokens=True)).strip()

    @property
    def logit_scale(self) -> float:
        self._ensure_clip()
        return float(self._clip.logit_scale.exp())

    # ---- evaluation -------------------------------------------------------

    def views(self, img: torch.Tensor, viewset: ViewSet) -> list[torch.Tensor]:
        return viewset.perceive(img)

    @torch.no_grad()
    def evaluate(
        self, img: torch.Tensor, spec: CandidateSpec, viewset: ViewSet
    ) -> Verdict:
        """img: (C, H, W) float in [0, 1]."""
        pils = [to_pil(v) for v in self.views(img, viewset)]

        # S[view][prompt]; both axes in slot order
        im_emb = self.image_embeddings(pils)
        tx_emb = self.text_embeddings(spec.full_prompts)
        s = (im_emb @ tx_emb.T).cpu()

        captions = [self.caption(p) for p in pils] if self.use_blip else []
        return build_verdict(
            spec,
            viewset,
            s,
            self.logit_scale,
            captions,
            extra={"size": int(img.shape[-1]), "view_seed": viewset.seed},
        )
