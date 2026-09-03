"""AudioLDM 2's mel front end, checked without any model.

The property the whole AudioLDM 2 track rests on is that time reversal is an
(almost) exact operation on the mel spectrogram. That can be checked on
synthetic signals with nothing loaded, so it is checked here, with tolerances
set from measurement rather than hope: on the percussive burst the flipped mel
differs from the mel of the reversed signal by 0.16 (natural-log units, RMS)
against a null of 6.09 if the flip did nothing; on the chirp 0.44 against 3.81.
The residual is the frame grid, which is anchored at sample 0 and so is not
itself reversal-symmetric.
"""

import numpy as np
import pytest

from ava.audio.mel import (
    AUDIOLDM2_MEL,
    log_mel_spectrogram,
    mel_frames,
    peak_normalize,
)
from ava.audio.signals import percussive_burst, rising_chirp

RATE = AUDIOLDM2_MEL.sample_rate


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).mean()))


def test_frame_rate_is_one_hundred_per_second() -> None:
    mel = log_mel_spectrogram(percussive_burst(sample_rate=RATE, duration=5.0))
    assert mel.shape == (500, 64)
    assert AUDIOLDM2_MEL.frames_per_second == 100.0


def test_frames_round_up_to_the_latent_grid() -> None:
    """The VAE halves time twice, so frame counts are multiples of four."""
    assert mel_frames(80_000) == 500
    assert mel_frames(80_003) == 504
    assert mel_frames(160) == 4
    assert mel_frames(1) == 4
    assert mel_frames(AUDIOLDM2_MEL.samples_per_latent_frame) == 4


def test_odd_lengths_are_padded_not_truncated() -> None:
    wave = percussive_burst(sample_rate=RATE, duration=5.0)[:79_500]
    mel = log_mel_spectrogram(wave)
    assert mel.shape == (500, 64)


@pytest.mark.parametrize(
    ("signal", "limit"),
    [(percussive_burst, 0.3), (rising_chirp, 0.6)],
)
def test_flipping_the_mel_is_reversing_the_audio(signal, limit: float) -> None:
    """flip(mel(x)) ~ mel(reverse(x)); the null (mel(x) itself) is far away."""
    wave = signal(sample_rate=RATE, duration=5.0)
    mel = log_mel_spectrogram(wave)
    mel_reversed = log_mel_spectrogram(wave[::-1].copy())
    measured = _rms(mel[::-1], mel_reversed)
    null = _rms(mel, mel_reversed)
    assert measured < limit
    assert measured < 0.15 * null


def test_log_floor_bounds_silence() -> None:
    mel = log_mel_spectrogram(np.zeros(16_000))
    assert np.allclose(mel, np.log(AUDIOLDM2_MEL.log_floor))


def test_peak_normalize_centres_and_scales() -> None:
    wave = np.array([1.0, 3.0, 1.0, 3.0])
    out = peak_normalize(wave)
    assert out.mean() == pytest.approx(0.0)
    assert np.abs(out).max() == pytest.approx(0.5)
    assert np.all(peak_normalize(np.zeros(8)) == 0.0)


def test_level_does_not_change_the_normalised_mel() -> None:
    """The loader's peak normalisation makes the front end gain-invariant."""
    wave = percussive_burst(sample_rate=RATE, duration=2.0)
    assert np.allclose(log_mel_spectrogram(wave), log_mel_spectrogram(0.1 * wave))


def test_mono_only() -> None:
    with pytest.raises(ValueError):
        log_mel_spectrogram(np.zeros((2, 16_000)))
