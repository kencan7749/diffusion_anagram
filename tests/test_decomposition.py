"""The hybrid views must form an identity decomposition of the epsilon prediction.

lp(e) + hp(e) == e holds only when both views share sigma and kernel_size.
This is the invariant that justifies REDUCTION == "sum"; if it broke, every
score downstream would be measuring the wrong thing.
"""

import pytest
import torch

from ava.image.engine import hybrid_views
from ava.spec import KERNEL_SIZE, SIGMA

SEED = 0


@pytest.mark.parametrize("size", [64, 256])
def test_low_plus_high_reconstructs_input(size: int) -> None:
    torch.manual_seed(SEED)
    # 6 channels mimics the sampler's layout: 3 noise + 3 variance channels.
    noise = torch.randn(6, size, size)

    low, high = hybrid_views(SIGMA, KERNEL_SIZE)
    # inverse_view mutates noise[:3] in place, so each view gets its own copy.
    lp = low.inverse_view(noise.clone())
    hp = high.inverse_view(noise.clone())

    torch.testing.assert_close(lp[:3] + hp[:3], noise[:3], rtol=1e-4, atol=1e-4)


def test_mismatched_sigma_breaks_the_decomposition() -> None:
    """Guards the reason sigma is a shared module constant rather than per-view."""
    torch.manual_seed(SEED)
    noise = torch.randn(6, 64, 64)

    low, _ = hybrid_views(SIGMA, KERNEL_SIZE)
    _, high_other = hybrid_views(SIGMA * 3, KERNEL_SIZE)

    lp = low.inverse_view(noise.clone())
    hp = high_other.inverse_view(noise.clone())

    assert not torch.allclose(lp[:3] + hp[:3], noise[:3], rtol=1e-3, atol=1e-3)


def test_inverse_view_leaves_trailing_channels_untouched() -> None:
    """Only the first three channels are decomposed; variance channels pass through."""
    torch.manual_seed(SEED)
    noise = torch.randn(6, 64, 64)

    low, _ = hybrid_views(SIGMA, KERNEL_SIZE)
    lp = low.inverse_view(noise.clone())

    torch.testing.assert_close(lp[3:], noise[3:])
