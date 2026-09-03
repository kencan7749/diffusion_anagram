"""A low-pass filter, as a map on AudioLDM 2's latent.

Factorized Diffusion builds a hybrid by splitting the noise estimate into
components that sum back to the whole -- for a hybrid image, a blur and its
residual -- and steering each with its own prompt. The audio hybrid wants the
same split by frequency: what reaches a listener through a wall (the low
band) and what is heard in the room (everything). The sampler works on the
latent, so the split has to be a map on the latent.

Zeroing the latent's frequency rows is not that map. The latent of the KL-VAE
is not a spectrogram: rows set to zero decode to something further from a
low-passed signal than the untouched latent is (measured before this module
was written: 45-66 dB against a do-nothing baseline of 8-50 dB). What does
work is a *fitted* affine map on each latent frame,

    z_low = z W + b

with W and b chosen by least squares from pairs (Enc(x), Enc(lowpass(x))) over
many clips. On clips held out of the fit, decoding `z_low` lands within the
codec's own error of `lowpass(Dec(z))` and drops the energy above the cutoff
by 20-40 dB. The bias is essential (without it the fit collapses) but plays
no part in the sampler: the two components of the noise estimate are

    low(eps)  = eps W           high(eps) = eps - eps W

which sum to eps exactly, as Factorized Diffusion requires, and the bias only
enters when the low component of a *sample* is decoded to be listened to.

`W` depends on the clips it was fitted on, so it is an analysis artefact
(Step 0c), persisted with its provenance and named explicitly by every run
that uses it -- there is no default projector in the code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


def to_frames(latent: torch.Tensor) -> torch.Tensor:
    """(1, C, T, F) -> (T, C*F): one row per latent time frame."""
    if latent.ndim != 4 or latent.shape[0] != 1:
        raise ValueError(f"expected a (1, C, T, F) latent, got {tuple(latent.shape)}")
    _, channels, frames, bins = latent.shape
    return latent[0].permute(1, 0, 2).reshape(frames, channels * bins)


def from_frames(frames: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Inverse of `to_frames`, with the layout of `like`."""
    _, channels, n_frames, bins = like.shape
    return frames.reshape(n_frames, channels, bins).permute(1, 0, 2)[None]


@dataclass(frozen=True)
class LowpassProjector:
    """The fitted map, plus where it came from.

    `weight` is (D, D) in the row-vector convention (`z @ weight`), `bias`
    is (D,), with D = channels * frequency bins of the latent (128 for
    AudioLDM 2). `meta` records the fit: cutoff, clips and their digests,
    ridge, residual, codec -- everything needed to refit it.
    """

    weight: np.ndarray
    bias: np.ndarray
    cutoff_hz: float
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.weight.ndim != 2 or self.weight.shape[0] != self.weight.shape[1]:
            raise ValueError(f"weight must be square, got {self.weight.shape}")
        if self.bias.shape != (self.weight.shape[0],):
            raise ValueError(
                f"bias must be ({self.weight.shape[0]},), got {self.bias.shape}"
            )

    @property
    def dim(self) -> int:
        return int(self.weight.shape[0])

    # ---- on latents -------------------------------------------------------

    def _tensors(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        # The map is applied in float32 whatever the latent's dtype: a 128x128
        # product in half precision would put its rounding into every step,
        # and CPU tensors have no half-precision matmul at all.
        weight = torch.as_tensor(self.weight, dtype=torch.float32, device=device)
        bias = torch.as_tensor(self.bias, dtype=torch.float32, device=device)
        return weight, bias

    def low(self, latent: torch.Tensor) -> torch.Tensor:
        """The low-band component of a sample: `z W + b`, frame by frame."""
        frames = to_frames(latent).float()
        weight, bias = self._tensors(frames.device)
        return from_frames(frames @ weight + bias, latent).to(latent.dtype)

    def split_epsilon(self, eps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(low, high) components of a noise estimate; they sum to `eps`.

        Linear, no bias: the constant would not cancel between the two
        branches' different estimates, and the decomposition must be exact.
        """
        frames = to_frames(eps).float()
        weight, _ = self._tensors(frames.device)
        low = from_frames(frames @ weight, eps).to(eps.dtype)
        return low, eps - low

    # ---- persistence ------------------------------------------------------

    def save(self, path: Path) -> Path:
        """Write `<path>.npz` (arrays) and `<path>.json` (provenance)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path.with_suffix(".npz"), weight=self.weight, bias=self.bias)
        path.with_suffix(".json").write_text(
            json.dumps(
                {"cutoff_hz": self.cutoff_hz, "dim": self.dim, **self.meta},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return path.with_suffix(".npz")

    @classmethod
    def load(cls, path: Path) -> LowpassProjector:
        arrays = np.load(path.with_suffix(".npz"))
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        cutoff = float(meta.pop("cutoff_hz"))
        meta.pop("dim", None)
        return cls(
            weight=arrays["weight"].astype(np.float64),
            bias=arrays["bias"].astype(np.float64),
            cutoff_hz=cutoff,
            meta=meta,
        )


def fit_lowpass_projector(
    latents: list[np.ndarray],
    lowpassed: list[np.ndarray],
    cutoff_hz: float,
    ridge: float = 1e-2,
    meta: dict[str, Any] | None = None,
) -> LowpassProjector:
    """Least squares for `z_low = z W + b` over frames pooled from many clips.

    `latents[i]` and `lowpassed[i]` are (T_i, D) frame matrices of the same
    clip, encoded as is and after low-pass filtering. Ridge-regularised so a
    direction the clips never excite gets a small weight rather than an
    arbitrary one. The training residual is recorded in `meta`; the
    generalisation to held-out clips is Step 0c's job, not this function's.
    """
    z = np.concatenate([np.asarray(a, dtype=np.float64) for a in latents])
    z_low = np.concatenate([np.asarray(a, dtype=np.float64) for a in lowpassed])
    if z.shape != z_low.shape:
        raise ValueError(f"frame matrices differ: {z.shape} vs {z_low.shape}")
    design = np.hstack([z, np.ones((z.shape[0], 1))])
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    solution = np.linalg.solve(gram, design.T @ z_low)
    weight, bias = solution[:-1], solution[-1]
    residual = float(np.linalg.norm(design @ solution - z_low) / np.linalg.norm(z_low))
    return LowpassProjector(
        weight=weight,
        bias=bias,
        cutoff_hz=cutoff_hz,
        meta={
            **(meta or {}),
            "ridge": ridge,
            "frames": int(z.shape[0]),
            "train_relative_residual": residual,
        },
    )
