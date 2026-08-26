"""Judge: measures J, the degree to which the illusion holds.

J is NOT a measure of how interesting an image is. It is a necessary condition
for the illusion working at all: it says whether both prompts are legible in
their respective views, nothing more.

J orders results and drives the diagnosis that steers the next round's
proposals. It does not discard anything. Every candidate is scored, recorded
and drawn, so the line between good and bad stays a human's to draw and can be
redrawn later from the saved scores.
"""

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .perceive import far_view, far_view_resize, near_view
from .spec import CandidateSpec, Verdict

CLIP_ID = "openai/clip-vit-large-patch14"
BLIP_ID = "Salesforce/blip-image-captioning-large"


def load_image(path: str | Path) -> torch.Tensor:
    """Read a PNG as (C, H, W) float in [0, 1]."""
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def scores_to_probs(
    s: torch.Tensor, logit_scale: float
) -> tuple[float, float, float]:
    """Turn a CLIP score matrix into (p_far, p_near, J).

    `s` is S[view][prompt], both axes ordered (low, high). Each row is a
    two-way choice, so chance level is 0.5. The softmax temperature is CLIP's
    own logit_scale.

    J is the minimum, not the sum. A sum would reward a candidate that renders
    the low-frequency side perfectly and the high-frequency side not at all;
    an illusion only exists when both sides hold.

    Kept separate from the model so it can be tested without loading CLIP.
    """
    if s.shape[-2:] != (2, 2):
        raise ValueError(f"expected a (2, 2) score matrix, got {tuple(s.shape)}")
    p = (logit_scale * s.float()).softmax(dim=-1)
    p_far = float(p[0, 0])  # far view picks prompt_low
    p_near = float(p[1, 1])  # near view picks prompt_high
    return p_far, p_near, min(p_far, p_near)


def to_pil(img: torch.Tensor) -> Image.Image:
    """(C, H, W) float in [0, 1] -> PIL."""
    arr = img.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((arr * 255).round().astype(np.uint8))


class Judge(Protocol):
    def evaluate(self, img: torch.Tensor, spec: CandidateSpec) -> Verdict: ...

    def views(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The (far, near) images that evaluate() scores.

        Part of the protocol because the loop persists them alongside the
        scores; that is what lets figures be redrawn without a recompute.
        """
        ...


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
        far_mode: str = "blur",  # "blur" | "resize"
    ) -> None:
        if far_mode not in ("blur", "resize"):
            raise ValueError(f"far_mode must be 'blur' or 'resize', got {far_mode!r}")
        self.device = device
        self.dtype = dtype
        self.clip_id = clip_id
        self.blip_id = blip_id
        self.use_blip = use_blip
        self.far_mode = far_mode

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
    def _text_emb(self, prompts: list[str]) -> torch.Tensor:
        """Normalized text embeddings (N, D), cached per string."""
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
    def _image_emb(self, pils: list[Image.Image]) -> torch.Tensor:
        """Normalized image embeddings (N, D)."""
        self._ensure_clip()
        px = self._clip_proc(images=pils, return_tensors="pt")["pixel_values"]
        px = px.to(self.device, self.dtype)
        emb = self._clip.get_image_features(pixel_values=px)
        return F.normalize(emb.float(), dim=-1)

    @torch.no_grad()
    def _caption(self, pil: Image.Image) -> str:
        self._ensure_blip()
        inputs = self._blip_proc(images=pil, return_tensors="pt").to(
            self.device, self.dtype
        )
        out = self._blip.generate(**inputs, max_new_tokens=30, num_beams=3)
        return str(self._blip_proc.decode(out[0], skip_special_tokens=True)).strip()

    # ---- evaluation -------------------------------------------------------

    def views(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        far_fn = far_view if self.far_mode == "blur" else far_view_resize
        return far_fn(img), near_view(img)

    @torch.no_grad()
    def evaluate(self, img: torch.Tensor, spec: CandidateSpec) -> Verdict:
        """img: (C, H, W) float in [0, 1]."""
        far, near = self.views(img)
        pil_far, pil_near = to_pil(far), to_pil(near)

        # S[view][prompt]; both axes ordered (low, high)
        im_emb = self._image_emb([pil_far, pil_near])
        tx_emb = self._text_emb(spec.prompts)
        s = im_emb @ tx_emb.T

        logit_scale = float(self._clip.logit_scale.exp())
        p_far, p_near, j = scores_to_probs(s, logit_scale)

        cap_far = cap_near = ""
        if self.use_blip:
            cap_far = self._caption(pil_far)
            cap_near = self._caption(pil_near)

        return Verdict(
            uid=spec.uid(),
            s_far_low=float(s[0, 0]),
            s_far_high=float(s[0, 1]),
            s_near_low=float(s[1, 0]),
            s_near_high=float(s[1, 1]),
            p_far=p_far,
            p_near=p_near,
            j=j,
            caption_far=cap_far,
            caption_near=cap_near,
            extra={
                "far_mode": self.far_mode,
                "size": int(img.shape[-1]),
                "logit_scale": logit_scale,
            },
        )
