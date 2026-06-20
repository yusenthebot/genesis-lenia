"""Dimension-agnostic continuous cellular automaton (Lenia-family).

The world is a real-valued field ``A`` of arbitrary spatial rank. One update is:

    U = K * A            (circular convolution with a normalised radial kernel)
    A = clip(A + dt * G(U), 0, 1)

with growth ``G`` a centred Gaussian bump mapped to [-1, 1]. With the right
parameters this produces *autopoietic* structures — localised patterns that
maintain and move themselves — without any of that behaviour being coded.

The kernel and convolution are built from ``shape`` alone, so the identical code
runs in 1D, 2D or 3D. Convolution is done in the Fourier domain (rfftn/irfftn)
on a full-grid centred kernel, which keeps the implementation rank-independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _bump(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Unit-height Gaussian bump centred at ``mu``."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


@dataclass(frozen=True)
class LeniaParams:
    """Rules of a world. Defaults are the canonical Orbium neighbourhood.

    The genome of a creature/world lives entirely here, so evolution later just
    perturbs these numbers.
    """

    R: float = 13.0                       # kernel radius in cells
    mu_k: float = 0.5                     # kernel ring centre (normalised radius)
    sigma_k: float = 0.15                 # kernel ring width
    mu_g: float = 0.15                    # growth centre
    sigma_g: float = 0.015               # growth width
    dt: float = 0.1                       # time step (1/T)
    kernel_peaks: tuple[float, ...] = (1.0,)  # relative heights of concentric rings

    def replace(self, **kw) -> "LeniaParams":
        from dataclasses import replace as _r

        return _r(self, **kw)


def kernel_shell(r: np.ndarray, params: LeniaParams) -> np.ndarray:
    """Radial kernel profile on normalised radius ``r`` (0 at centre, 1 at R).

    Supports Bert Chan's multi-ring kernel: the band [0,1) is split into one
    sub-band per entry of ``kernel_peaks``; each sub-band is a Gaussian ring of
    the given relative height. A single peak gives the standard one-ring kernel.
    """
    peaks = np.asarray(params.kernel_peaks, dtype=np.float64)
    n = len(peaks)
    out = np.zeros_like(r)
    mask = (r >= 0.0) & (r < 1.0)
    rm = r[mask]
    band = np.minimum((rm * n).astype(int), n - 1)
    frac = (rm * n) % 1.0  # position within the band, in [0,1)
    out[mask] = peaks[band] * _bump(frac, params.mu_k, params.sigma_k)
    return out


class World:
    """An N-dimensional continuous-CA world.

    Parameters
    ----------
    shape : tuple[int, ...]
        Spatial extent. ``len(shape)`` is the world's dimensionality.
    params : LeniaParams
        The update rule.
    """

    def __init__(self, shape, params: LeniaParams, dtype=np.float64):
        self.shape = tuple(int(s) for s in shape)
        if not self.shape:
            raise ValueError("shape must have at least one dimension")
        self.ndim = len(self.shape)
        self.params = params
        self.dtype = dtype
        self.A = np.zeros(self.shape, dtype=dtype)
        self.t = 0
        self._build_kernel()

    # -- construction -------------------------------------------------------
    def _build_kernel(self) -> None:
        idx = np.indices(self.shape, dtype=np.float64)           # (ndim, *shape)
        center = np.array([s // 2 for s in self.shape], dtype=np.float64)
        center = center.reshape((self.ndim,) + (1,) * self.ndim)
        dist = np.sqrt(((idx - center) ** 2).sum(axis=0))         # (*shape,)
        r = dist / self.params.R
        K = kernel_shell(r, self.params)
        total = K.sum()
        if total <= 0:
            raise ValueError(
                "Kernel is empty — check R/mu_k/sigma_k relative to shape."
            )
        self.K = K / total
        # centre the kernel at the origin so the convolution introduces no shift
        self.K_fft = np.fft.rfftn(np.fft.ifftshift(self.K))

    # -- dynamics -----------------------------------------------------------
    def growth(self, U: np.ndarray) -> np.ndarray:
        """Growth map G(U) in [-1, 1]; +1 at the preferred neighbourhood sum."""
        return 2.0 * _bump(U, self.params.mu_g, self.params.sigma_g) - 1.0

    def potential(self, A: np.ndarray | None = None) -> np.ndarray:
        """Neighbourhood sum U = K * A via circular FFT convolution."""
        field = self.A if A is None else A
        axes = tuple(range(self.ndim))
        AF = np.fft.rfftn(field, axes=axes)
        return np.fft.irfftn(AF * self.K_fft, s=self.shape, axes=axes)

    def step(self) -> np.ndarray:
        U = self.potential()
        self.A = np.clip(self.A + self.params.dt * self.growth(U), 0.0, 1.0)
        self.t += 1
        return self.A

    def run(self, steps: int, record: bool = False):
        """Advance ``steps`` updates. If ``record``, return stacked history."""
        if not record:
            for _ in range(steps):
                self.step()
            return None
        hist = np.empty((steps,) + self.shape, dtype=self.dtype)
        for i in range(steps):
            hist[i] = self.step()
        return hist

    # -- state --------------------------------------------------------------
    @property
    def mass(self) -> float:
        return float(self.A.sum())

    def seed_blob(self, center=None, radius=None, amp=1.0, asymmetry=0.0,
                  rng=None) -> None:
        """Place a localised Gaussian blob — the only thing we ever hand-place.

        ``asymmetry`` skews the blob along the last axis so a symmetric kernel
        can still impart net motion (a self-propelled creature instead of a
        stationary breather).
        """
        if center is None:
            center = tuple(s // 2 for s in self.shape)
        if radius is None:
            radius = self.params.R
        idx = np.indices(self.shape, dtype=np.float64)
        c = np.array(center, dtype=np.float64).reshape((self.ndim,) + (1,) * self.ndim)
        d = idx - c
        dist2 = (d ** 2).sum(axis=0)
        blob = amp * np.exp(-0.5 * dist2 / radius ** 2)
        if asymmetry:
            # multiply by a sigmoid ramp along the last axis -> skewed blob
            ramp = 1.0 / (1.0 + np.exp(-asymmetry * d[-1] / radius))
            blob = blob * ramp
        if rng is not None:
            blob = blob * (0.85 + 0.3 * rng.random(self.shape))
        self.A = np.clip(self.A + blob, 0.0, 1.0)
