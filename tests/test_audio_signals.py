"""The probe signals must actually isolate what they claim to isolate.

The whole reading of Step 0a rests on this. If `rising_chirp` turned out to
also swing its level, a direction-sensitive score would no longer tell us
whether CLAP heard the trajectory or the envelope, and the experiment would
answer a different question than the one asked. So the separation is asserted
here rather than trusted from the docstrings.

Thresholds come from the measured values, with margin:

    signal            envelope symmetry   centroid 1st -> 2nd half
    rising_chirp            +0.98             1150 -> 3050 Hz
    percussive_burst        -0.16             6216 -> 6192 Hz
    symmetric_burst         +1.00             6190 -> 6188 Hz
"""

import numpy as np
import pytest

from ava.audio.features import envelope_symmetry, rms_profile, spectral_centroid_halves
from ava.audio.perceive import reverse_view
from ava.audio.signals import (
    DURATION,
    PROBES,
    SAMPLE_RATE,
    percussive_burst,
    rising_chirp,
    symmetric_burst,
)


@pytest.mark.parametrize("name", sorted(PROBES))
def test_probes_are_deterministic(name: str) -> None:
    """A result is only reproducible if the inputs are."""
    first, second = PROBES[name](), PROBES[name]()
    assert np.array_equal(first, second)


@pytest.mark.parametrize("name", sorted(PROBES))
def test_probes_have_the_declared_shape_and_level(name: str) -> None:
    wave = PROBES[name]()
    assert wave.ndim == 1
    assert wave.size == int(round(SAMPLE_RATE * DURATION))
    assert np.max(np.abs(wave)) == pytest.approx(0.95, abs=1e-6)


def test_chirp_moves_its_spectrum_and_holds_its_level() -> None:
    """Trajectory probe: the spectrum goes somewhere, the envelope does not."""
    wave = rising_chirp()
    first, second = spectral_centroid_halves(wave, SAMPLE_RATE)
    assert second > 2.0 * first
    assert envelope_symmetry(wave) > 0.95


def test_reversing_the_chirp_reverses_the_sweep() -> None:
    reversed_wave = reverse_view(rising_chirp())
    first, second = spectral_centroid_halves(reversed_wave, SAMPLE_RATE)
    assert second < 0.5 * first


def test_percussive_burst_moves_its_level_and_holds_its_spectrum() -> None:
    """Envelope probe: the level decays, the spectrum stays put."""
    wave = percussive_burst()
    first, second = spectral_centroid_halves(wave, SAMPLE_RATE)
    assert abs(second / first - 1.0) < 0.05

    profile = rms_profile(wave)
    assert profile[:5].mean() > 100.0 * profile[-5:].mean()
    assert envelope_symmetry(wave) < 0.2


def test_control_is_unchanged_in_every_way_reversal_could_change_it() -> None:
    """Control: reversal alters every sample and nothing audible."""
    wave = symmetric_burst()
    first, second = spectral_centroid_halves(wave, SAMPLE_RATE)
    assert abs(second / first - 1.0) < 0.05
    assert envelope_symmetry(wave) > 0.95

    # Samples really do change -- otherwise the control would be vacuous.
    assert not np.allclose(wave, reverse_view(wave))
