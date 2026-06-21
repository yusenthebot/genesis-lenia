"""Flow-Lenia — mass-conserving Lenia (Plantec et al. 2022), in pure numpy.

Plain Lenia updates A += dt*G(U) and clips — growth ADDS and REMOVES mass, so mass is not
conserved (a moving body dissipates; R25's mobile 3D candidates were diffuse for this reason).
Flow-Lenia instead CONSERVES mass: the growth defines an AFFINITY, mass FLOWS up the affinity
gradient, and is moved (not grown) by mass-conserving advection. The total stays fixed forever;
a "creature" is a self-organised clump of the conserved mass that can hold together and travel.

This is the same engine pieces (radial kernel, FFT potential, Gaussian growth) plus:
  affinity G = growth(U);  flow F = clip(grad G);  A <- upwind-advect(A, F)  (mass-conserving).

Dimension-agnostic (len(shape) is the dimension), like the rest of the project. Additive module:
plain Lenia (world.py) is untouched.
"""

from __future__ import annotations

import numpy as np

from genesis.world import LeniaParams, kernel_shell


def _build_kernel_fft(shape, params: LeniaParams, asym=0.0, phi=0.0):
    """Radial Lenia kernel, optionally made ASYMMETRIC (ring stronger on one side).

    asym in [0,1) tilts the kernel by (1 + asym*cos(theta - phi)) in the first two axes,
    breaking rotational symmetry -> a persistent directional bias in the flow -> motion.
    """
    idx = np.indices(shape, dtype=np.float64)
    centre = np.array([s // 2 for s in shape], dtype=np.float64)
    centre = centre.reshape((len(shape),) + (1,) * len(shape))
    r = np.sqrt(((idx - centre) ** 2).sum(0)) / params.R
    K = kernel_shell(r, params)
    if asym > 0.0 and len(shape) >= 2:
        theta = np.arctan2(idx[0] - centre[0], idx[1] - centre[1])
        K = K * np.clip(1.0 + asym * np.cos(theta - phi), 0.0, None)
    K /= K.sum()
    return np.fft.rfftn(np.fft.ifftshift(K))


def advect(A, F, dt):
    """Mass-conserving upwind advection of field A by flow F (ndim, *shape), periodic.

    Solves dA/dt + div(A F) = 0 with first-order upwind fluxes. Total mass is conserved
    because every face flux that leaves one cell enters its neighbour (telescoping sum).
    """
    out = A.copy()
    ndim = A.ndim
    for d in range(ndim):
        v = F[d]
        vface = 0.5 * (v + np.roll(v, -1, d))            # velocity on the +d face of cell i
        flux = np.where(vface >= 0, A, np.roll(A, -1, d)) * vface  # upwind donor mass
        out = out - dt * (flux - np.roll(flux, 1, d))    # gain left face, lose right face
    return np.clip(out, 0.0, None)


class FlowWorld:
    """A Flow-Lenia world: total mass is conserved; structure self-organises and moves."""

    def __init__(self, shape, params: LeniaParams, flow_clip=1.0, dt=None,
                 asym=0.0, phi=0.0):
        self.shape = tuple(int(s) for s in shape)
        self.ndim = len(self.shape)
        self.params = params
        self.K_fft = _build_kernel_fft(self.shape, params, asym=asym, phi=phi)
        self.flow_clip = flow_clip
        self.dt = params.dt if dt is None else dt
        self.A = np.zeros(self.shape)
        self.t = 0

    def potential(self, A=None):
        field = self.A if A is None else A
        ax = tuple(range(self.ndim))
        return np.fft.irfftn(np.fft.rfftn(field, axes=ax) * self.K_fft,
                             s=self.shape, axes=ax)

    def growth(self, U):
        return np.exp(-0.5 * ((U - self.params.mu_g) / self.params.sigma_g) ** 2) * 2.0 - 1.0

    def step(self):
        U = self.potential()
        G = self.growth(U)                                # affinity: +1 at the ideal neighbourhood
        F = np.array(np.gradient(G))                      # flow up the affinity gradient
        # clip flow magnitude (CFL stability + keeps creatures from rocketing off)
        mag = np.sqrt((F ** 2).sum(0)) + 1e-12
        scale = np.clip(self.flow_clip / mag, None, 1.0)
        F = F * scale
        self.A = advect(self.A, F, self.dt)
        self.t += 1
        return self.A

    def mass(self):
        return float(self.A.sum())
