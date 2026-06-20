"""A world with food — the step from locomotion to AGENCY.

A ``ForagingWorld`` adds a scalar food field ``F`` to the Lenia substrate and couples
it two ways:

  sense  — food near the creature's body boosts growth *there*, so the creature grows
           up the local food gradient. Directed motion toward food (chemotaxis) is not
           coded; it emerges from this coupling and is then selected for.
  eat    — food under the creature's body is consumed (removed from F, tallied).

Optionally a small metabolic ``decay`` makes the creature slowly lose mass unless it
keeps eating, so foraging becomes survival, not just reward. The base rule + seed come
from an evolved glider; round 3 evolves the sensing/steering on top.
"""

from __future__ import annotations

import numpy as np

from genesis.world import World, LeniaParams


class ForagingWorld(World):
    def __init__(self, shape, params: LeniaParams, sense_sigma: float,
                 gamma: float, eta: float = 0.20, decay: float = 0.0,
                 feed: float = 0.0, dtype=np.float64):
        super().__init__(shape, params, dtype=dtype)
        self.sense_sigma = float(sense_sigma)
        self.gamma = float(gamma)        # sensorimotor gain (sense -> drift)
        self.eta = float(eta)            # consumption rate
        self.decay = float(decay)        # metabolic cost per step
        self.feed = float(feed)          # eaten food -> body mass (sustenance)
        self.F = np.zeros(self.shape, dtype=dtype)
        self.eaten = 0.0
        self._drift = np.zeros(self.ndim)   # accumulated sub-cell displacement
        self._build_sense_kernel(self.sense_sigma)

    def _build_sense_kernel(self, sigma: float) -> None:
        idx = np.indices(self.shape, dtype=np.float64)
        c = np.array([s // 2 for s in self.shape], dtype=np.float64)
        c = c.reshape((self.ndim,) + (1,) * self.ndim)
        d2 = ((idx - c) ** 2).sum(axis=0)
        Ks = np.exp(-0.5 * d2 / max(sigma, 1e-6) ** 2)
        # peak-normalise (NOT sum-normalise): S stays order-1 near food, so the
        # gradient the creature senses is meaningful rather than vanishingly small.
        Ks /= Ks.max()
        self.Ks_fft = np.fft.rfftn(np.fft.ifftshift(Ks))

    def sense(self) -> np.ndarray:
        """Food smoothed by the sensing kernel: how much food is *near* each cell."""
        axes = tuple(range(self.ndim))
        FF = np.fft.rfftn(self.F, axes=axes)
        return np.fft.irfftn(FF * self.Ks_fft, s=self.shape, axes=axes)

    def step(self) -> np.ndarray:
        U = self.potential()                       # neighbourhood sum K*A
        # base Lenia growth, minus a metabolic cost paid wherever there is body
        self.A = self.A + self.params.dt * (self.growth(U) - self.decay * (self.A > 0))
        # eat food under the body; eaten food feeds the body (sustenance)
        bite = np.clip(self.eta * self.A, 0.0, 1.0) * self.F
        self.F = self.F - bite
        self.A = self.A + self.feed * bite
        self.A = np.clip(self.A, 0.0, 1.0)
        self.eaten += float(bite.sum())
        # SENSORIMOTOR REFLEX: sense the food gradient over the body and rigidly
        # translate the creature toward it. np.roll is an exact permutation, so this
        # moves the body without creating or destroying any mass (no blow-up). The
        # body is still an emergent Lenia glider; only the sense->drift gain is evolved.
        if self.gamma:
            gS = np.gradient(self.sense())
            tot = self.A.sum() + 1e-9
            for ax in range(self.ndim):
                grad_ax = float((self.A * gS[ax]).sum() / tot)  # body-averaged gradient
                self._drift[ax] += self.gamma * grad_ax
                shift = int(round(self._drift[ax]))
                if shift:
                    self.A = np.roll(self.A, shift, axis=ax)
                    self._drift[ax] -= shift
        self.t += 1
        return self.A

    def add_food_blob(self, center, radius: float, amp: float = 1.0) -> None:
        idx = np.indices(self.shape, dtype=np.float64)
        c = np.array(center, dtype=np.float64).reshape((self.ndim,) + (1,) * self.ndim)
        d2 = ((idx - c) ** 2).sum(axis=0)
        self.F = self.F + amp * np.exp(-0.5 * d2 / radius ** 2)


def random_food_layout(world: ForagingWorld, rng, n_clusters=1, dist_frac=0.30,
                       radius=10.0, amp=1.0):
    """Place food cluster(s) at random angle(s), a fixed distance from the centre.

    Random *direction* each episode is what forces taxis: a ballistic glider only
    hits food when it happens to be dead ahead; a steering creature turns to it.
    """
    H = world.shape[0]
    center = np.array([s / 2 for s in world.shape])
    dist = dist_frac * min(world.shape)
    for _ in range(n_clusters):
        ang = rng.uniform(0, 2 * np.pi)
        off = np.array([np.cos(ang), np.sin(ang)]) * dist
        # jitter the offset a little so clusters are not all on one ring
        off = off * rng.uniform(0.8, 1.15)
        pos = (center + off) % np.array(world.shape)
        world.add_food_blob(tuple(pos), radius=radius, amp=amp)
