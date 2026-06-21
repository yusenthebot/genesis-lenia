"""Multi-channel Flow-Lenia — the path to a MOBILE creature (round 29+).

Round 28 diagnosed why single-channel Flow-Lenia can't move: F = grad(G) is a gradient flow
(curl-free) -> it only relaxes to a stationary equilibrium. The fix (Plantec et al.): MULTIPLE
channels, each field advected by ITS OWN affinity gradient, COUPLED through cross-channel kernels.
Channel c's affinity depends on the OTHER channels' fields, so the channels chase each other; the
coupled system is NOT a single gradient and can sustain a travelling wave -> a glider.

Each channel still conserves its own mass (mass-conserving advection), so a found mover persists.
Pure numpy, dimension-agnostic, additive (single-channel flowlenia.py untouched).
"""

from __future__ import annotations

import numpy as np

from genesis.world import LeniaParams, kernel_shell
from genesis.flowlenia import advect


def _kernel_fft(shape, R, mu_k, sigma_k, asym=0.0, phi=0.0, offset=(0, 0)):
    """A radial kernel, optionally asymmetric (angular tilt) and/or spatially OFFSET.

    A spatial offset shifts the neighbourhood a channel senses -> a directional coupling,
    which is what lets coupled channels translate.
    """
    idx = np.indices(shape, dtype=np.float64)
    centre = np.array([s // 2 for s in shape], dtype=np.float64)
    centre = centre.reshape((len(shape),) + (1,) * len(shape))
    r = np.sqrt(((idx - centre) ** 2).sum(0)) / R
    K = kernel_shell(r, LeniaParams(R=R, mu_k=mu_k, sigma_k=sigma_k))
    if asym > 0.0 and len(shape) >= 2:
        theta = np.arctan2(idx[0] - centre[0], idx[1] - centre[1])
        K = K * np.clip(1.0 + asym * np.cos(theta - phi), 0.0, None)
    K /= K.sum() + 1e-12
    if len(shape) >= 2 and (offset[0] or offset[1]):
        K = np.roll(np.roll(K, int(offset[0]), 0), int(offset[1]), 1)
    return np.fft.rfftn(np.fft.ifftshift(K))


class FlowWorldMC:
    """Multi-channel Flow-Lenia. `conn` is a list of interaction dicts:
       {src, dst, R, mu_k, sigma_k, mu_g, sigma_g, weight, offset, asym, phi}.
    Each interaction adds weight*growth(K_offset * A[src]) to channel dst's affinity.
    """

    def __init__(self, shape, n_channels, conn, flow_clip=0.9, dt=0.2):
        self.shape = tuple(int(s) for s in shape)
        self.ndim = len(self.shape)
        self.C = n_channels
        self.conn = conn
        self.flow_clip = flow_clip
        self.dt = dt
        self.A = [np.zeros(self.shape) for _ in range(n_channels)]
        for k in conn:
            k["_fft"] = _kernel_fft(self.shape, k["R"], k["mu_k"], k["sigma_k"],
                                    k.get("asym", 0.0), k.get("phi", 0.0),
                                    k.get("offset", (0, 0)))
        self.t = 0

    def _growth(self, U, mu_g, sigma_g):
        return np.exp(-0.5 * ((U - mu_g) / sigma_g) ** 2) * 2.0 - 1.0

    def step(self):
        ax = tuple(range(self.ndim))
        Afft = [np.fft.rfftn(a, axes=ax) for a in self.A]
        affinity = [np.zeros(self.shape) for _ in range(self.C)]
        wsum = [1e-9] * self.C
        for k in self.conn:
            U = np.fft.irfftn(Afft[k["src"]] * k["_fft"], s=self.shape, axes=ax)
            g = self._growth(U, k["mu_g"], k["sigma_g"])
            w = k.get("weight", 1.0)
            affinity[k["dst"]] += w * g
            wsum[k["dst"]] += abs(w)
        new = []
        for c in range(self.C):
            G = affinity[c] / wsum[c]
            F = np.array(np.gradient(G))
            mag = np.sqrt((F ** 2).sum(0)) + 1e-12
            F = F * np.clip(self.flow_clip / mag, None, 1.0)
            new.append(advect(self.A[c], F, self.dt))
        self.A = new
        self.t += 1

    def total(self):
        return np.array([a.sum() for a in self.A])

    def combined(self):
        return sum(self.A)

    def centroid(self):
        tot = sum(a.sum() for a in self.A) + 1e-9
        idx = np.indices(self.shape)
        comb = self.combined()
        return np.array([float((comb * idx[i]).sum() / tot) for i in range(self.ndim)])


def two_channel_conn(R=12.0, mu_k=0.5, sigma_k=0.15, mu_g=0.15, sigma_g=0.018,
                     cross_off=4, asym=0.0):
    """A canonical 2-channel coupling with an OFFSET cross-interaction (directional)."""
    base = dict(R=R, mu_k=mu_k, sigma_k=sigma_k, mu_g=mu_g, sigma_g=sigma_g)
    return [
        dict(src=0, dst=0, weight=1.0, offset=(0, 0), asym=asym, **base),
        dict(src=1, dst=1, weight=1.0, offset=(0, 0), asym=asym, **base),
        dict(src=1, dst=0, weight=0.8, offset=(cross_off, 0), **base),   # ch0 senses ch1 ahead
        dict(src=0, dst=1, weight=0.8, offset=(-cross_off, 0), **base),  # ch1 senses ch0 behind
    ]
