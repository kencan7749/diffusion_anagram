"""Latent codecs, behind one interface, so the same experiment runs on any of them.

The reversal anagram works in latent space: the sampler flips a latent and asks
the model to denoise the flip towards a second prompt. That only means
something if flipping the latent's time axis *is* reversing the audio, and
whether it is depends entirely on the autoencoder.

There is no reason to expect it for free. A 1D convolution satisfies

    conv(reverse(x), k) = reverse(conv(x, reverse(k)))

so reversal commutes with a conv layer only when the kernel is symmetric, which
a learned kernel is not, and the mismatch compounds through the stack. Striding
adds a second problem: reversal maps sample n to N-1-n, which need not land on
the decimation grid, so a fractional offset can creep in. Causal padding, if any
is used, breaks the symmetry outright.

Hence this interface: `encode`, `decode` and `flip_time`, with nothing else, so
Step 0b can be pointed at Stable Audio's waveform autoencoder or AudioLDM 2's
mel autoencoder without the measurement code knowing which it has.
"""

from typing import Any, Protocol

import numpy as np
import torch


class LatentCodec(Protocol):
    """What the view-validity measurement needs from an autoencoder."""

    # A property, not a bare annotation: an implementation reads this off the
    # loaded model, and a plain attribute would oblige it to be settable.
    @property
    def sample_rate(self) -> int: ...

    def encode(self, wave: np.ndarray) -> torch.Tensor: ...

    def decode(self, latent: torch.Tensor) -> np.ndarray: ...

    def flip_time(self, latent: torch.Tensor) -> torch.Tensor: ...


class StableAudioCodec:
    """AutoencoderOobleck, the waveform autoencoder of Stable Audio Open.

    Loaded on its own rather than through StableAudioPipeline: the question
    here is about the autoencoder, and pulling in T5 and the DiT to ask it
    would cost several GB of weights that never get used.

    Runs in float32 on purpose. The measurement *is* a reconstruction error,
    and half precision would mix its own rounding into the number.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-audio-open-1.0",
        device: str = "cuda",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._vae: Any = None

    def _ensure_vae(self) -> None:
        if self._vae is None:
            from diffusers import AutoencoderOobleck

            # low_cpu_mem_usage=False is required, not an optimisation choice.
            # The default path builds the model on the meta device and streams the
            # weights in, but Oobleck wraps every conv in torch.nn.utils.weight_norm,
            # which computes the weight during construction -- and torch 2.1.1 has no
            # meta kernel for aten::_weight_norm_interface. Newer torch would fix it;
            # so would parametrizations.weight_norm upstream. Neither is worth
            # disturbing the pinned torch that the image pipeline reproduces on.
            vae = AutoencoderOobleck.from_pretrained(
                self.model_id, subfolder="vae", low_cpu_mem_usage=False
            )
            self._vae = vae.to(self.device).eval()

    @property
    def sample_rate(self) -> int:
        self._ensure_vae()
        return int(self._vae.config.sampling_rate)

    @property
    def channels(self) -> int:
        self._ensure_vae()
        return int(self._vae.config.audio_channels)

    def _to_batch(self, wave: np.ndarray) -> torch.Tensor:
        """(n,) or (channels, n) -> (1, channels, n), duplicating mono if needed."""
        array = wave[None, :] if wave.ndim == 1 else wave
        if array.shape[0] == 1 and self.channels == 2:
            array = np.repeat(array, 2, axis=0)
        return torch.from_numpy(array.astype(np.float32))[None].to(self.device)

    @torch.no_grad()
    def encode(self, wave: np.ndarray) -> torch.Tensor:
        """Waveform -> latent (1, C, T).

        Takes the distribution's mode rather than a sample: a stochastic
        encoder would put noise into a comparison whose whole point is to
        measure a difference of a few dB.
        """
        self._ensure_vae()
        return self._vae.encode(self._to_batch(wave)).latent_dist.mode()

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> np.ndarray:
        """Latent (1, C, T) -> waveform (channels, n)."""
        self._ensure_vae()
        audio = self._vae.decode(latent).sample
        return audio[0].float().cpu().numpy()

    def flip_time(self, latent: torch.Tensor) -> torch.Tensor:
        """Reverse the latent's time axis, which is the last one for (B, C, T)."""
        return torch.flip(latent, dims=[-1])
