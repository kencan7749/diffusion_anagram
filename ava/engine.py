"""Resident generator for Factorized Diffusion hybrid images.

Why this exists rather than shelling out to `generate.py`:

  1. `generate.py` reloads every pipeline on each invocation; T5-XXL alone is
     11.5 GB in fp16. A search loop needs the pipelines resident.
  2. `views.get_views` only forwards `--view_args` to five view types, so the
     hybrid sigma cannot be set through the CLI at all, and future triple /
     motion views would lose their parameters silently. This module builds the
     View objects directly instead.

Prompt embeddings are cached per string, which matters because a recombination
search re-encounters the same prompt many times.
"""

from pathlib import Path
from typing import Any, Protocol

import torch
from torchvision.utils import save_image
from visual_anagrams.samplers import sample_stage_1, sample_stage_2
from visual_anagrams.views.view_hybrid import HybridHighPassView, HybridLowPassView

from .spec import KERNEL_SIZE, REDUCTION, SIGMA, CandidateSpec

STAGE_1_ID = "DeepFloyd/IF-I-M-v1.0"
STAGE_2_ID = "DeepFloyd/IF-II-M-v1.0"
STAGE_3_ID = "stabilityai/stable-diffusion-x4-upscaler"


def hybrid_views(sigma: float = SIGMA, kernel_size: int = KERNEL_SIZE) -> list[Any]:
    """Build the (low_pass, high_pass) view pair.

    Both views must share sigma and kernel_size, otherwise lp(e1) + hp(e2) is
    not an identity decomposition of the epsilon prediction.
    """
    return [
        HybridLowPassView(sigma=sigma, kernel_size=kernel_size),
        HybridHighPassView(sigma=sigma, kernel_size=kernel_size),
    ]


def to_01(image: torch.Tensor) -> torch.Tensor:
    """Sampler output (1, 3, H, W) in [-1, 1] -> (3, H, W) in [0, 1]."""
    return (image[0] / 2.0 + 0.5).clamp(0, 1)


class Generator(Protocol):
    """What the loop needs from a generator.

    Engine is the real implementation; the protocol exists so the loop can be
    tested end to end without loading DeepFloyd or holding a GPU.
    """

    def generate(
        self, spec: CandidateSpec, sigma: float = SIGMA, kernel_size: int = KERNEL_SIZE
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def upscale_1024(
        self, spec: CandidateSpec, image_256: torch.Tensor
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
    def encode_spec(self, spec: CandidateSpec) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack the (low, high) prompt embeddings in view order."""
        pairs = [self.encode(p) for p in spec.prompts]
        pos = torch.cat([p for p, _ in pairs])
        neg = torch.cat([n for _, n in pairs])  # null embeddings
        return pos, neg

    # ---- generation -------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        spec: CandidateSpec,
        sigma: float = SIGMA,
        kernel_size: int = KERNEL_SIZE,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate one hybrid image.

        Returns (image_64, image_256), both (3, H, W) float in [0, 1].

        `sigma` is exposed only so Step 0 can generate deliberately broken
        examples. The search loop always leaves it at SIGMA.
        """
        views = hybrid_views(sigma, kernel_size)
        pos, neg = self.encode_spec(spec)
        generator = torch.manual_seed(spec.seed)

        image_64 = sample_stage_1(
            self.stage_1,
            pos,
            neg,
            views,
            num_inference_steps=spec.num_inference_steps,
            guidance_scale=spec.guidance_scale,
            reduction=REDUCTION,
            generator=generator,
        )
        image_256 = sample_stage_2(
            self.stage_2,
            image_64,
            pos,
            neg,
            views,
            num_inference_steps=spec.num_inference_steps,
            guidance_scale=spec.guidance_scale,
            reduction=REDUCTION,
            generator=generator,
        )
        return to_01(image_64), to_01(image_256)

    @torch.no_grad()
    def upscale_1024(
        self, spec: CandidateSpec, image_256: torch.Tensor
    ) -> torch.Tensor:
        """Naive x4 upscale conditioned on prompt_low only.

        The upscaler knows nothing about the other view, so the high-frequency
        side can degrade. Harvest-time only; see the upstream readme.
        """
        generator = torch.manual_seed(spec.seed)
        out = self.stage_3(
            prompt=spec.full_low,
            image=(image_256 * 2 - 1)[None],
            noise_level=0,
            output_type="pt",
            generator=generator,
        ).images
        return out[0].clamp(0, 1)


def save_sample(img_01: torch.Tensor, out_dir: Path) -> Path:
    """Save one image as sample_<size>.png.

    `visual_anagrams.utils.save_illusion` is deliberately not used: it also
    writes sample_<size>.views.png, which is meaningless for hybrid views
    because `view()` is the identity there.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"sample_{img_01.shape[-1]}.png"
    save_image(img_01, path, padding=0)
    return path
