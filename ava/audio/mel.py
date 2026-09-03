"""AudioLDM 2's mel front end, reproduced so a waveform can be encoded.

diffusers ships AudioLDM 2 as a decoder only: the VAE decodes a latent to a
mel spectrogram and HiFi-GAN turns that into a waveform. Nothing in the
pipeline goes the other way, because text-to-audio never needs to. The
view-validity measurement (Step 0b) does -- it has to encode a known waveform,
flip the latent and listen to what comes back -- so the front end the model was
trained on is written out here.

The recipe is AudioLDM's `TacotronSTFT` (audioldm/audio/stft.py): a 1024-point
STFT with a Hann window, hop 160 at 16 kHz (10 ms per frame), reflect padding
of 512 samples on each side, *magnitude* (not power) projected onto 64 Slaney
mel filters between 0 and 8 kHz, then a natural log floored at 1e-5. The
training loader also normalised every waveform to a peak of 0.5 before this
(audioldm/audio/tools.py, `read_wav_file`); `peak_normalize` does the same, and
whether the whole recipe is right is checked empirically by handing the result
to the vocoder and measuring what it plays back.

Why this front end makes time reversal a good latent view: the Hann window is
symmetric, so the magnitude spectrogram of a reversed signal is the reversed
magnitude spectrogram of the signal, up to a sub-hop misalignment of the frame
grid. Mel projection and the log act per frame and do not care about time
direction. What is left to measure in Step 0b is only the VAE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from transformers.audio_utils import mel_filter_bank, spectrogram, window_function


@dataclass(frozen=True)
class MelConfig:
    """The constants AudioLDM 2 was trained with. Not meant to be changed."""

    sample_rate: int = 16_000
    n_fft: int = 1024
    hop_length: int = 160
    win_length: int = 1024
    n_mels: int = 64
    f_min: float = 0.0
    f_max: float = 8_000.0
    log_floor: float = 1e-5
    # The training loader's peak level.
    peak: float = 0.5
    # The VAE downsamples time by this factor, so a spectrogram it encodes must
    # have a multiple of this many frames.
    latent_stride: int = 4

    @property
    def frames_per_second(self) -> float:
        return self.sample_rate / self.hop_length

    @property
    def samples_per_latent_frame(self) -> int:
        return self.hop_length * self.latent_stride


AUDIOLDM2_MEL = MelConfig()


def peak_normalize(wave: np.ndarray, peak: float = AUDIOLDM2_MEL.peak) -> np.ndarray:
    """Remove the mean and scale to the given peak, as AudioLDM's loader did.

    Silence stays silence rather than becoming NaN.
    """
    centred = wave.astype(np.float64) - wave.mean()
    top = float(np.abs(centred).max())
    if top < 1e-8:
        return np.zeros_like(centred)
    return centred / top * peak


def mel_frames(n_samples: int, config: MelConfig = AUDIOLDM2_MEL) -> int:
    """Frames the front end yields for a signal, rounded up to the latent grid.

    AudioLDM's loader pads the waveform to a whole number of frames and crops
    the spectrogram to that count; the same convention is used here so that a
    signal of `duration_s` seconds occupies `ceil(duration_s * 100)` frames,
    then rounded up to a multiple of the VAE's stride.
    """
    frames = int(np.ceil(n_samples / config.hop_length))
    stride = config.latent_stride
    return int(np.ceil(frames / stride)) * stride


def log_mel_spectrogram(
    wave: np.ndarray, config: MelConfig = AUDIOLDM2_MEL, normalize: bool = True
) -> np.ndarray:
    """Mono waveform at `config.sample_rate` -> log-mel of shape (frames, n_mels).

    `frames` is `mel_frames(len(wave))`: the waveform is zero-padded to that
    many hops and the extra frame the centred STFT produces at the far end is
    dropped, exactly as `wav_to_fbank` does upstream.
    """
    if wave.ndim != 1:
        raise ValueError(f"expected a mono (n,) waveform, got shape {wave.shape}")
    signal = peak_normalize(wave, config.peak) if normalize else wave.astype(np.float64)

    frames = mel_frames(signal.shape[0], config)
    padded = np.zeros(frames * config.hop_length, dtype=np.float64)
    padded[: signal.shape[0]] = signal

    filters = mel_filter_bank(
        num_frequency_bins=config.n_fft // 2 + 1,
        num_mel_filters=config.n_mels,
        min_frequency=config.f_min,
        max_frequency=config.f_max,
        sampling_rate=config.sample_rate,
        norm="slaney",
        mel_scale="slaney",
    )
    # power=1.0: AudioLDM projects magnitudes, not power, onto the mel filters.
    # center=True with reflect padding is the 512-sample reflect pad upstream.
    mel = spectrogram(
        padded,
        window=window_function(config.win_length, "hann"),
        frame_length=config.n_fft,
        hop_length=config.hop_length,
        power=1.0,
        center=True,
        pad_mode="reflect",
        mel_filters=filters,
        mel_floor=0.0,
        dtype=np.float64,
    )
    log_mel = np.log(np.maximum(mel, config.log_floor))
    # (n_mels, frames + 1) -> (frames, n_mels)
    return np.ascontiguousarray(log_mel.T[:frames])
