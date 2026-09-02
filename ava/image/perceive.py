"""Perceptual views: what a human actually sees, not the algebraic decomposition.

The two papers differ in where the transformation lives, and that decides what
the judge has to look at:

  Visual Anagrams   `view(x_t)` transforms the noisy image, so the transformed
                    image IS what the viewer sees. `va_transform` applies the
                    upstream view to the finished picture.
  Factorized        `view(im)` is the identity and the decomposition happens in
  Diffusion         `inverse_view(noise)`, on the epsilon prediction. The
                    component the sampler uses (`im - blur`, `im - gray`) is a
                    zero-mean residual that means nothing to CLIP. So the views a
                    person sees are defined here, independently: blurred /
                    as-is, grayscale / as-is, motion-blurred / as-is.

Every function accepts (C, H, W) or (B, C, H, W) in [0, 1] and returns the same
shape. Both shapes are handled explicitly at the top of each function; a helper
that silently treats a batch axis as a channel axis returns wrong numbers, not
an exception, and that has happened three times in this repository.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from ava.spec import KERNEL_SIZE, MOTION_SIZE, SIGMA


def _check_image(img: torch.Tensor) -> None:
    if img.ndim not in (3, 4):
        raise ValueError(f"expected (C, H, W) or (B, C, H, W), got {tuple(img.shape)}")


def scale_blur_params(sigma_64: float, kernel_64: int, size: int) -> tuple[int, float]:
    """Scale a 64 px blur to the evaluation resolution.

    Uses the same rule as `inverse_view` in view_hybrid.py:
        factor = h // 64
        k      = kernel_size * factor + ((factor + 1) % 2)
        sigma  = sigma * factor
    The ((factor + 1) % 2) term keeps the kernel size odd.
    """
    factor = max(1, size // 64)
    k = kernel_64 * factor + ((factor + 1) % 2)
    sigma = sigma_64 * factor
    assert k % 2 == 1, f"gaussian_blur requires an odd kernel: k={k}"
    return k, sigma


def blur_params(size: int) -> tuple[int, float]:
    """The hybrid-image blur at `size`; kept for the Step 0 scripts."""
    return scale_blur_params(SIGMA, KERNEL_SIZE, size)


def blur(
    img: torch.Tensor, sigma_64: float = SIGMA, kernel_64: int = KERNEL_SIZE
) -> torch.Tensor:
    """Gaussian blur matched to a generation-time sigma given at the 64 px stage."""
    _check_image(img)
    k, sigma = scale_blur_params(sigma_64, kernel_64, img.shape[-1])
    return TF.gaussian_blur(img, kernel_size=k, sigma=sigma)


def compound_sigma(sigma_1: float, sigma_2: float) -> float:
    """Sigma of two Gaussian blurs applied in sequence.

    The triple hybrid's low band is `G_sigma2(G_sigma1(x))`; a viewer far
    enough away to see only that band sees a blur of this width.
    """
    return math.sqrt(sigma_1**2 + sigma_2**2)


def far_view(img: torch.Tensor) -> torch.Tensor:
    """Far / squinted view of a hybrid image: blur matched to SIGMA."""
    return blur(img, SIGMA, KERNEL_SIZE)


def far_view_resize(img: torch.Tensor, factor: int = 12) -> torch.Tensor:
    """Alternative far view: downsample, then upsample back.

    factor=12 is the default `resize_factor` used by
    `view_hybrid.make_frame_hybrid` for hybrid animations. Step 0 checks that
    this and `far_view` lead to the same conclusion, so that the gate does not
    depend on one particular blur implementation. Factorized Diffusion's own
    metric sweeps this factor over [1, 8] and keeps the best (App. D).
    """
    _check_image(img)
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


def grayscale(img: torch.Tensor) -> torch.Tensor:
    """What a colour hybrid looks like in dim light: the channel mean, replicated.

    The same linear grayscale the sampler decomposes with (view_color.py),
    not a luma-weighted conversion, so the judge sees the component the
    prompt was attached to.
    """
    _check_image(img)
    return img.mean(dim=-3, keepdim=True).expand_as(img).clone()


def motion_blur(img: torch.Tensor, size_64: int = MOTION_SIZE) -> torch.Tensor:
    """What a motion hybrid looks like while moving: the diagonal line blur.

    Reproduces `MotionBlurView.save_view`: a `size x size` identity kernel over
    `size`, scaled from the 64 px stage and kept odd, applied per channel.
    """
    _check_image(img)
    batched = img.ndim == 4
    x = img if batched else img[None]
    b, c, h, w = x.shape
    factor = max(1, h // 64)
    size = size_64 * factor + ((factor + 1) % 2)
    kernel = (torch.eye(size, dtype=x.dtype, device=x.device) / size)[None, None]
    flat = x.reshape(b * c, 1, h, w)
    out = F.conv2d(flat, kernel, padding=size // 2).reshape(b, c, h, w)
    return out if batched else out[0]


def va_transform(view: Any, img: torch.Tensor) -> torch.Tensor:
    """Apply a Visual Anagrams view to a finished image.

    Upstream views act on the model's [-1, 1] range and on a single (C, H, W)
    tensor, so the image is rescaled around it and a batch is handled one
    image at a time. Negation therefore comes out as `1 - img`.
    """
    _check_image(img)
    if img.ndim == 4:
        return torch.stack([va_transform(view, im) for im in img])
    out = view.view(img * 2.0 - 1.0)
    return ((out + 1.0) / 2.0).clamp(0.0, 1.0)
