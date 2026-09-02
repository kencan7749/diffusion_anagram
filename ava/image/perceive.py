"""Perceptual views: what a human actually sees, not the algebraic decomposition.

This is deliberately separate from Factorized Diffusion's own view classes.
In FD, `view(im)` is the identity and the decomposition happens in
`inverse_view(noise)`, applied to the epsilon prediction (see view_hybrid.py).
Two consequences:

  - `sample_*.views.png` is just the same image N times for hybrid views, so it
    carries no evaluation signal.
  - The high-frequency component the sampler uses (`im - blur`) is a zero-mean
    residual; feeding it to CLIP collapses the scores.

So the evaluation views are defined here independently.

Perception of a hybrid image:
    far / squinting -> low frequencies dominate  -> blurred image
    near            -> high frequencies dominate -> the image itself
"""

import torch
import torchvision.transforms.functional as TF

from ava.spec import KERNEL_SIZE, SIGMA


def blur_params(size: int) -> tuple[int, float]:
    """Scale the generation-time blur to the evaluation resolution.

    Uses the same rule as `inverse_view` in view_hybrid.py:
        factor = h // 64
        k      = kernel_size * factor + ((factor + 1) % 2)
        sigma  = sigma * factor
    The ((factor + 1) % 2) term keeps the kernel size odd.
    """
    factor = max(1, size // 64)
    k = KERNEL_SIZE * factor + ((factor + 1) % 2)
    sigma = SIGMA * factor
    assert k % 2 == 1, f"gaussian_blur requires an odd kernel: k={k}"
    return k, sigma


def far_view(img: torch.Tensor) -> torch.Tensor:
    """Far / squinted view: Gaussian blur matched to the generation sigma.

    img: (C, H, W) or (B, C, H, W), values in [0, 1].
    """
    size = img.shape[-1]
    k, sigma = blur_params(size)
    return TF.gaussian_blur(img, kernel_size=k, sigma=sigma)


def far_view_resize(img: torch.Tensor, factor: int = 12) -> torch.Tensor:
    """Alternative far view: downsample, then upsample back.

    factor=12 is the default `resize_factor` used by
    `view_hybrid.make_frame_hybrid` for hybrid animations. Step 0 checks that
    this and `far_view` lead to the same conclusion, so that the gate does not
    depend on one particular blur implementation.
    """
    size = img.shape[-1]
    small = max(1, size // factor)
    down = TF.resize(img, [small, small], antialias=True)
    return TF.resize(down, [size, size], antialias=True)


def near_view(img: torch.Tensor) -> torch.Tensor:
    """Near view: the image itself.

    Do not return (img - blur) here. That is the sampler's algebraic
    high-frequency component, not what a human sees.
    """
    return img
