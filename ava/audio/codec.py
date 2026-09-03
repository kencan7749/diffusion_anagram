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
mel autoencoder without the measurement code knowing which it has. Both are
below; `CODECS` maps the command-line names to them.
"""

from typing import Any, Protocol

import numpy as np
import torch

from ava.audio.features import to_mono
from ava.audio.mel import AUDIOLDM2_MEL, MelConfig, log_mel_spectrogram


class LatentCodec(Protocol):
    """What the view-validity measurement needs from an autoencoder."""

    # A property, not a bare annotation: an implementation reads this off the
    # loaded model, and a plain attribute would oblige it to be settable.
    @property
    def sample_rate(self) -> int: ...

    # The latent's time axis, and the waveform samples one frame stands for.
    @property
    def time_dim(self) -> int: ...

    @property
    def samples_per_frame(self) -> int: ...

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

    @property
    def time_dim(self) -> int:
        return -1

    @property
    def samples_per_frame(self) -> int:
        self._ensure_vae()
        return int(np.prod(self._vae.config.downsampling_ratios))

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


class AudioLDM2Codec:
    """AudioLDM 2's mel autoencoder and vocoder, as one waveform codec.

    Two models rather than one, because the latent is a mel spectrogram's:
    `encode` runs the front end in `ava.audio.mel` and then the KL-VAE, and
    `decode` runs the VAE and then HiFi-GAN, which invents the phase. The
    pipeline's text encoders and UNet are not loaded, for the reason the
    Stable Audio codec gives.

    The latent is (1, 8, T/4, 16): time is the second-to-last axis, and
    `flip_time` flips that one. What a flip means in the decoded waveform is
    reversal plus a fixed shift of one hop less a sample (159 samples, 10 ms),
    because the STFT's frame grid is anchored at sample 0 and reversal maps
    sample 0 to sample N-1. The shift is the same for every signal and is not
    the thing Step 0b measures, but it is in the numbers, so it is stated here.

    Runs in float32, like the other codec: the measurement is a reconstruction
    error and should not carry half-precision rounding.
    """

    def __init__(
        self,
        model_id: str = "cvssp/audioldm2",
        device: str = "cuda",
        mel: MelConfig = AUDIOLDM2_MEL,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.mel = mel
        self._vae: Any = None
        self._vocoder: Any = None

    def _ensure_models(self) -> None:
        if self._vae is None:
            from diffusers import AutoencoderKL
            from transformers import SpeechT5HifiGan

            vae = AutoencoderKL.from_pretrained(self.model_id, subfolder="vae")
            vocoder = SpeechT5HifiGan.from_pretrained(
                self.model_id, subfolder="vocoder"
            )
            self._vae = vae.to(self.device).eval()
            self._vocoder = vocoder.to(self.device).eval()
            if int(vocoder.config.sampling_rate) != self.mel.sample_rate:
                raise ValueError(
                    f"vocoder runs at {vocoder.config.sampling_rate} Hz, "
                    f"front end at {self.mel.sample_rate} Hz"
                )

    @property
    def sample_rate(self) -> int:
        return self.mel.sample_rate

    @property
    def time_dim(self) -> int:
        return -2

    @property
    def samples_per_frame(self) -> int:
        return self.mel.samples_per_latent_frame

    @torch.no_grad()
    def encode(self, wave: np.ndarray) -> torch.Tensor:
        """Waveform (n,) or (channels, n) at 16 kHz -> latent (1, 8, T/4, 16).

        Stereo is averaged: the model is mono. Takes the distribution's mode,
        as the Stable Audio codec does, for the same reason.
        """
        self._ensure_models()
        mel = log_mel_spectrogram(to_mono(wave), self.mel)
        batch = torch.from_numpy(mel.astype(np.float32))[None, None].to(self.device)
        return self._vae.encode(batch).latent_dist.mode()

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> np.ndarray:
        """Latent (1, 8, T/4, 16) -> waveform (1, n), n = T * hop."""
        self._ensure_models()
        mel = self._vae.decode(latent).sample[0, 0]
        return self.vocode(mel)

    @torch.no_grad()
    def vocode(self, log_mel: np.ndarray | torch.Tensor) -> np.ndarray:
        """Log-mel (T, 64) -> waveform (1, n), straight through HiFi-GAN.

        Public so the front end can be checked on its own: if
        `vocode(log_mel_spectrogram(x))` does not play back as x, the recipe in
        `ava.audio.mel` is wrong and no VAE measurement downstream means much.
        """
        self._ensure_models()
        mel = torch.as_tensor(
            np.asarray(log_mel) if not torch.is_tensor(log_mel) else log_mel
        )
        mel = mel.to(self.device, torch.float32)
        wave = self._vocoder(mel[None])
        return wave.float().cpu().numpy().reshape(1, -1)

    def flip_time(self, latent: torch.Tensor) -> torch.Tensor:
        """Reverse the time axis, the second to last for (B, C, T, mel)."""
        return torch.flip(latent, dims=[-2])


CODECS: dict[str, type[StableAudioCodec] | type[AudioLDM2Codec]] = {
    "stable_audio": StableAudioCodec,
    "audioldm2": AudioLDM2Codec,
}
