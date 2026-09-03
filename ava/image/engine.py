"""Resident generator for multi-view illusions on DeepFloyd IF.

Why this exists rather than shelling out to `generate.py`:

  1. `generate.py` reloads every pipeline on each invocation; T5-XXL alone is
     11.5 GB in fp16. A search loop needs the pipelines resident.
  2. `views.get_views` only forwards `--view_args` to five view types, so the
     hybrid sigma cannot be set through the CLI at all, and the triple / motion
     views would lose their parameters silently. `ava.image.tasks` builds the
     View objects directly instead.

Both papers share one sampler (`visual_anagrams.samplers`): the views decide
whether it behaves as Visual Anagrams (transform the noisy image, average the
estimates) or as Factorized Diffusion (decompose the estimates, sum them). The
task carries the matching `reduction`, and this module only wires it through.

Prompt embeddings are cached per string, which matters because a recombination
search re-encounters the same prompt many times.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image
from visual_anagrams.samplers import sample_stage_1, sample_stage_2

from ava.image.views import ViewSet
from ava.spec import CandidateSpec

STAGE_1_ID = "DeepFloyd/IF-I-M-v1.0"
STAGE_2_ID = "DeepFloyd/IF-II-M-v1.0"
STAGE_3_ID = "stabilityai/stable-diffusion-x4-upscaler"

# What the generator is told about a slot that is pinned to a reference image.
# The paper conditions that component on the empty prompt (FD readme, inverse).
REF_SLOT_PROMPT = ""


def to_01(image: torch.Tensor) -> torch.Tensor:
    """Sampler output (1, 3, H, W) in [-1, 1] -> (3, H, W) in [0, 1]."""
    return (image[0] / 2.0 + 0.5).clamp(0, 1)


def load_reference(path: str | Path) -> torch.Tensor:
    """Read a reference image as (3, H, W) in [-1, 1], the sampler's range."""
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0


def generator_prompts(spec: CandidateSpec, viewset: ViewSet) -> list[str]:
    """The prompts the model is conditioned on, in slot order.

    Identical to `spec.full_prompts` except for a reference slot, whose prompt
    describes the reference image for the judge and is replaced by the empty
    prompt for generation.
    """
    prompts = spec.full_prompts
    if viewset.task.ref_slot is not None:
        prompts[viewset.task.ref_slot] = REF_SLOT_PROMPT
    return prompts


class Generator(Protocol):
    """What the loop needs from a generator.

    Engine is the real implementation; the protocol exists so the loop can be
    tested end to end without loading DeepFloyd or holding a GPU.
    """

    def generate(
        self, spec: CandidateSpec, viewset: ViewSet
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def upscale_1024(
        self, spec: CandidateSpec, image_256: torch.Tensor, viewset: ViewSet
    ) -> torch.Tensor: ...


class Engine:
    """Holds the DeepFloyd pipelines and generates one candidate at a time.

    `sample_stage_1` hardcodes `batch_size = 1`, so candidates are generated
    sequentially; roughly 30 s each at 30 steps.
    """

    def __init__(
        self, device: str = "cuda", dtype: torch.dtype = torch.float16
    ) -> None:
        self.device = device
        self.dtype = dtype
        self._stage_1: Any = None
        self._stage_2: Any = None
        self._stage_3: Any = None
        # prompt string -> (prompt_embeds, negative_prompt_embeds)
        self._embed_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._reference_cache: dict[str, torch.Tensor] = {}

    # ---- lazy pipeline loading -------------------------------------------

    @property
    def stage_1(self) -> Any:
        if self._stage_1 is None:
            from diffusers import DiffusionPipeline

            pipe = DiffusionPipeline.from_pretrained(
                STAGE_1_ID, variant="fp16", torch_dtype=self.dtype
            )
            pipe.enable_model_cpu_offload()
            self._stage_1 = pipe.to(self.device)
        return self._stage_1

    @property
    def stage_2(self) -> Any:
        if self._stage_2 is None:
            from diffusers import DiffusionPipeline

            pipe = DiffusionPipeline.from_pretrained(
                STAGE_2_ID, text_encoder=None, variant="fp16", torch_dtype=self.dtype
            )
            pipe.enable_model_cpu_offload()
            self._stage_2 = pipe.to(self.device)
        return self._stage_2

    @property
    def stage_3(self) -> Any:
        """x4 upscaler. Loaded only when harvesting, never during screening."""
        if self._stage_3 is None:
            from diffusers import DiffusionPipeline

            pipe = DiffusionPipeline.from_pretrained(STAGE_3_ID, torch_dtype=self.dtype)
            pipe.enable_model_cpu_offload()
            self._stage_3 = pipe.to(self.device)
        return self._stage_3

    # ---- prompt encoding --------------------------------------------------

    @torch.no_grad()
    def encode(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one prompt, caching by string to avoid repeat T5 calls."""
        if prompt not in self._embed_cache:
            pos, neg = self.stage_1.encode_prompt(prompt)
            self._embed_cache[prompt] = (pos, neg)
        return self._embed_cache[prompt]

    @torch.no_grad()
    def encode_spec(
        self, spec: CandidateSpec, viewset: ViewSet
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack the prompt embeddings in view order."""
        pairs = [self.encode(p) for p in generator_prompts(spec, viewset)]
        pos = torch.cat([p for p, _ in pairs])
        neg = torch.cat([n for _, n in pairs])  # null embeddings
        return pos, neg

    def reference(self, spec: CandidateSpec) -> torch.Tensor | None:
        if spec.ref_image is None:
            return None
        if spec.ref_image not in self._reference_cache:
            self._reference_cache[spec.ref_image] = load_reference(spec.ref_image)
        return self._reference_cache[spec.ref_image]

    # ---- generation -------------------------------------------------------

    @torch.no_grad()
    def generate(
        self, spec: CandidateSpec, viewset: ViewSet
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate one illusion.

        Returns (image_64, image_256), both (3, H, W) float in [0, 1].
        """
        task = viewset.task
        if spec.task != task.name:
            raise ValueError(f"spec is for {spec.task!r}, views are for {task.name!r}")
        if spec.n_views != task.n_views:
            raise ValueError(
                f"{task.name} has {task.n_views} views, got {spec.n_views} prompts"
            )
        if (task.ref_slot is None) != (spec.ref_image is None):
            raise ValueError(f"{task.name}: reference image and ref_slot must agree")

        pos, neg = self.encode_spec(spec, viewset)
        ref_im = self.reference(spec)
        generator = torch.manual_seed(spec.seed)

        image_64 = sample_stage_1(
            self.stage_1,
            pos,
            neg,
            viewset.views,
            ref_im=ref_im,
            num_inference_steps=spec.num_inference_steps,
            guidance_scale=spec.guidance_scale,
            reduction=task.reduction,
            generator=generator,
        )
        image_256 = sample_stage_2(
            self.stage_2,
            image_64,
            pos,
            neg,
            viewset.views,
            ref_im=ref_im,
            num_inference_steps=spec.num_inference_steps,
            guidance_scale=spec.guidance_scale,
            reduction=task.reduction,
            generator=generator,
        )
        return to_01(image_64), to_01(image_256)

    @torch.no_grad()
    def upscale_1024(
        self, spec: CandidateSpec, image_256: torch.Tensor, viewset: ViewSet
    ) -> torch.Tensor:
        """Naive x4 upscale conditioned on one prompt.

        The upscaler is a latent model and knows nothing about views, so it is
        conditioned on the task's `sr_slot`: the identity view for Visual
        Anagrams, the finest component for Factorized Diffusion (App. A.4).
        Harvest-time only.
        """
        generator = torch.manual_seed(spec.seed)
        out = self.stage_3(
            prompt=spec.full_prompt(viewset.task.sr_slot),
            image=(image_256 * 2 - 1)[None],
            noise_level=0,
            output_type="pt",
            generator=generator,
        ).images
        return out[0].clamp(0, 1)


def save_sample(img_01: torch.Tensor, out_dir: Path) -> Path:
    """Save one image as sample_<size>.png.

    `visual_anagrams.utils.save_illusion` is deliberately not used: it also
    writes sample_<size>.views.png, whose content depends on the view type and
    is regenerated by the judge anyway.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"sample_{img_01.shape[-1]}.png"
    save_image(img_01, path, padding=0)
    return path
