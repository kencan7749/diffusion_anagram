"""The reversal anagram sampler for Stable Audio Open.

At every denoising step the epsilon prediction is the average of two branches:

    eps = 0.5 * (eps(z, p_forward) + flip(eps(flip(z), p_reverse)))

which is Visual Anagrams' construction with time reversal as the view. The
view is an involution, so there is no separate inverse to derive, and Step 0b
established that flipping the latent's time axis really does decode to the
reversed waveform (0.6 dB above the codec's own floor on a probe where doing
nothing would have cost 49.9 dB).

Written against the pipeline's internals rather than calling it, for the same
reason `ava/image/engine.py` does not call generate.py: the view has to be
applied inside the loop, and `__call__` offers no way in.

Two details of this model shape the implementation.

**The latent is always 1024 frames.** `audio_end_in_s` does not shorten it; it
conditions the model and then crops the decoded waveform. So flipping the whole
latent would move a 10 s sound to the far end of a 47 s span, and the crop
would return the wrong part of it. The flip is therefore applied only to the
frames the audio actually occupies, and the tail is left alone.

**Both branches fit in one forward pass.** The batch is laid out as
[z, z, flip(z), flip(z)] against [uncond, forward, uncond, reverse], so one
call to the transformer yields both branches and their guidance terms.
"""

import math
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch

from ava.audio.permute import (
    JIGSAW_BLOCKS,
    JIGSAW_PERM,
    block_slices,
    inverse_permutation,
)
from ava.audio.spec import AudioCandidateSpec

MODEL_ID = "stabilityai/stable-audio-open-1.0"


def flip_window(latent: torch.Tensor, frames: int, dim: int = -1) -> torch.Tensor:
    """Reverse the first `frames` along the latent's time axis, leaving the rest.

    Reversing the whole axis would relocate the audio rather than reverse it
    when the occupied region is a prefix of a much longer latent, as it is for
    Stable Audio. `dim` names the time axis: the last one for Stable Audio's
    (B, C, T), the second to last for AudioLDM 2's (B, C, T, mel).
    """
    if frames >= latent.shape[dim]:
        return torch.flip(latent, dims=[dim])
    index: list[slice] = [slice(None)] * latent.ndim
    index[dim] = slice(0, frames)
    window = tuple(index)
    out = latent.clone()
    out[window] = torch.flip(latent[window], dims=[dim])
    return out


def permute_blocks(
    latent: torch.Tensor, frames: int, perm: tuple[int, ...], dim: int = -1
) -> torch.Tensor:
    """Cut the first `frames` along `dim` into equal blocks and reorder them.

    Position i of the result holds source block `perm[i]`. Frames that do not
    divide evenly, and everything past `frames`, stay where they are -- the
    same rule `ava.audio.permute.block_slices` applies to the waveform.
    """
    axis = dim % latent.ndim
    slices = block_slices(frames, len(perm))
    out = latent.clone()
    for position, source in enumerate(perm):
        target: list[slice] = [slice(None)] * latent.ndim
        origin: list[slice] = [slice(None)] * latent.ndim
        target[axis], origin[axis] = slices[position], slices[source]
        out[tuple(target)] = latent[tuple(origin)]
    return out


class LatentView(Protocol):
    """A linear, invertible rearrangement of a latent's time axis.

    Visual Anagrams' condition: the view must be orthogonal so the noise stays
    Gaussian; permutations of frames are, and a flip is one of them.
    """

    @property
    def name(self) -> str: ...

    def apply(
        self, latent: torch.Tensor, frames: int, dim: int = -1
    ) -> torch.Tensor: ...

    def invert(
        self, latent: torch.Tensor, frames: int, dim: int = -1
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class TimeReverse:
    """Play it backwards. An involution: its own inverse."""

    name: str = "time_reverse"

    def apply(self, latent: torch.Tensor, frames: int, dim: int = -1) -> torch.Tensor:
        return flip_window(latent, frames, dim)

    def invert(self, latent: torch.Tensor, frames: int, dim: int = -1) -> torch.Tensor:
        return flip_window(latent, frames, dim)


@dataclass(frozen=True)
class BlockPermute:
    """Cut time into equal blocks and play them in another order."""

    perm: tuple[int, ...] = JIGSAW_PERM
    name: str = f"jigsaw_{JIGSAW_BLOCKS}"

    def apply(self, latent: torch.Tensor, frames: int, dim: int = -1) -> torch.Tensor:
        return permute_blocks(latent, frames, self.perm, dim)

    def invert(self, latent: torch.Tensor, frames: int, dim: int = -1) -> torch.Tensor:
        return permute_blocks(latent, frames, inverse_permutation(self.perm), dim)


TIME_REVERSE_VIEW = TimeReverse()
JIGSAW_VIEW = BlockPermute()
LATENT_VIEWS: dict[str, LatentView] = {
    TIME_REVERSE_VIEW.name: TIME_REVERSE_VIEW,
    JIGSAW_VIEW.name: JIGSAW_VIEW,
}


def anagram_epsilon(
    prediction: torch.Tensor,
    guidance_scale: float,
    frames: int,
    dim: int = -1,
    view: LatentView = TIME_REVERSE_VIEW,
) -> torch.Tensor:
    """Fold the batch-of-4 transformer output into a single epsilon.

    Rows arrive as [uncond, forward, uncond, viewed]: guidance is applied
    within each branch, then the two are averaged. The second branch predicted
    noise for a transformed latent, so its prediction is transformed back
    first -- without that the two branches would be added in different time
    frames and the result would be neither sound.

    Split out from the sampling loop so the arithmetic can be checked on
    hand-written tensors, without a 4 GB model and a GPU. `dim` is the time
    axis, as for `flip_window`; the AudioLDM 2 engine shares this function.
    """
    uncond_f, cond_f, uncond_r, cond_r = prediction.chunk(4)
    eps_forward = uncond_f + guidance_scale * (cond_f - uncond_f)
    eps_viewed = uncond_r + guidance_scale * (cond_r - uncond_r)
    return 0.5 * (eps_forward + view.invert(eps_viewed, frames, dim))


class AudioAnagramEngine:
    """Holds the Stable Audio pipeline and samples one anagram at a time.

    The pipeline stays resident because a search re-enters it for every
    candidate, and loading it costs several GB.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        view: LatentView = TIME_REVERSE_VIEW,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.view = view
        # diffusers' randn_tensor inspects `device.type`, so the pipeline
        # helpers need a torch.device rather than the string used elsewhere.
        self.torch_device = torch.device(device)
        self.dtype = dtype
        self._pipe: Any = None
        # Prompt string -> conditioning embedding. A search re-encounters the
        # same prompt many times.
        self._embed_cache: dict[str, torch.Tensor] = {}

    # ---- lazy loading -----------------------------------------------------

    @property
    def pipe(self) -> Any:
        if self._pipe is None:
            from diffusers import StableAudioPipeline

            # low_cpu_mem_usage=False is mandatory here; see ava/audio/codec.py
            # for why no version of torch or diffusers removes the need.
            pipe = StableAudioPipeline.from_pretrained(
                self.model_id, torch_dtype=self.dtype, low_cpu_mem_usage=False
            )
            self._pipe = pipe.to(self.device)
        return self._pipe

    @property
    def sample_rate(self) -> int:
        return int(self.pipe.vae.sampling_rate)

    @property
    def downsample_ratio(self) -> int:
        return int(np.prod(self.pipe.vae.config.downsampling_ratios))

    def latent_frames(self, duration_s: float) -> int:
        """How many latent frames a given duration of audio occupies."""
        samples = duration_s * self.sample_rate
        return min(
            int(self.pipe.transformer.config.sample_size),
            int(math.ceil(samples / self.downsample_ratio)),
        )

    # ---- conditioning -----------------------------------------------------

    @torch.no_grad()
    def _prompt_emb(self, prompt: str) -> torch.Tensor:
        """Conditional text embedding for one prompt, cached per string."""
        if prompt not in self._embed_cache:
            self._embed_cache[prompt] = self.pipe.encode_prompt(
                prompt, self.device, False
            )
        return self._embed_cache[prompt]

    @torch.no_grad()
    def _conditioning(
        self, spec: AudioCandidateSpec
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (encoder_hidden_states, global_hidden_states) for the batch of 4.

        Rows are [uncond, forward, uncond, reverse]: the two branches share the
        unconditional term but not the prompt.
        """
        pipe = self.pipe
        start, end = pipe.encode_duration(0.0, spec.duration_s, self.device, False, 1)

        def with_duration(text_emb: torch.Tensor) -> torch.Tensor:
            return torch.cat([text_emb, start, end], dim=1)

        forward = with_duration(self._prompt_emb(spec.prompt_forward))
        reverse = with_duration(self._prompt_emb(spec.prompt_reverse))
        # The pipeline substitutes zeros when no negative prompt is given; an
        # explicit one is encoded like any other.
        uncond = (
            with_duration(self._prompt_emb(spec.negative_prompt))
            if spec.negative_prompt
            else torch.zeros_like(forward)
        )

        encoder_states = torch.cat([uncond, forward, uncond, reverse], dim=0)
        global_states = torch.cat([start, end], dim=2).repeat(4, 1, 1)
        return encoder_states, global_states

    # ---- sampling ---------------------------------------------------------

    @torch.no_grad()
    def generate(self, spec: AudioCandidateSpec) -> np.ndarray:
        """Sample one anagram. Returns (channels, n) float in roughly [-1, 1]."""
        from diffusers.models.embeddings import get_1d_rotary_pos_embed

        pipe = self.pipe
        encoder_states, global_states = self._conditioning(spec)

        pipe.scheduler.set_timesteps(spec.num_inference_steps, device=self.device)
        timesteps = pipe.scheduler.timesteps

        generator = torch.Generator(self.device).manual_seed(spec.seed)
        latents = pipe.prepare_latents(
            1,
            pipe.transformer.config.in_channels,
            int(pipe.transformer.config.sample_size),
            encoder_states.dtype,
            self.torch_device,
            generator,
            None,
            None,
            1,
            audio_channels=pipe.vae.config.audio_channels,
        )

        rotary = get_1d_rotary_pos_embed(
            pipe.rotary_embed_dim,
            latents.shape[2] + global_states.shape[1],
            use_real=True,
            repeat_interleave_real=False,
        )
        frames = self.latent_frames(spec.duration_s)

        for t in timesteps:
            viewed = self.view.apply(latents, frames)
            model_input = torch.cat([latents, latents, viewed, viewed])
            model_input = pipe.scheduler.scale_model_input(model_input, t)

            prediction = pipe.transformer(
                model_input,
                t.unsqueeze(0),
                encoder_hidden_states=encoder_states,
                global_hidden_states=global_states,
                rotary_embedding=rotary,
                return_dict=False,
            )[0]

            eps = anagram_epsilon(
                prediction, spec.guidance_scale, frames, view=self.view
            )
            latents = pipe.scheduler.step(eps, t, latents).prev_sample

        audio = pipe.vae.decode(latents).sample
        # The whole span the view acted on, not the requested duration: the
        # occupied frames cover a little more than `duration_s`, and cropping
        # to the request would shift the viewed branch by the difference (15 ms
        # at 5 s). A listener's view cuts the waveform at the same instants
        # the latent view cut the frames only if the two have the same length.
        end_sample = frames * self.downsample_ratio
        return audio[0, :, :end_sample].float().cpu().numpy()
