"""Flow-Lenia invariants: the defining property is EXACT mass conservation (2D + 3D)."""

import numpy as np

from genesis.world import LeniaParams
from genesis.flowlenia import FlowWorld, advect


def _blob(shape, sigma=8.0, amp=0.5):
    idx = np.indices(shape, dtype=float)
    c = [s // 2 for s in shape]
    d2 = sum((idx[i] - c[i]) ** 2 for i in range(len(shape)))
    return amp * np.exp(-0.5 * d2 / sigma ** 2)


def test_advect_conserves_mass():
    rng = np.random.default_rng(0)
    A = rng.random((40, 40))
    F = rng.normal(0, 0.3, (2, 40, 40))
    out = advect(A, F, dt=0.2)
    assert abs(out.sum() - A.sum()) < 1e-9        # advection conserves total mass


def test_mass_conserved_2d():
    p = LeniaParams(R=12.0, mu_k=0.5, sigma_k=0.15, mu_g=0.15, sigma_g=0.018, dt=0.2)
    w = FlowWorld((80, 80), p, flow_clip=0.85)
    w.A = _blob((80, 80))
    m0 = w.mass()
    for _ in range(60):
        w.step()
    assert abs(w.mass() / m0 - 1.0) < 1e-6        # the defining Flow-Lenia property


def test_mass_conserved_and_compact_3d():
    """3D: mass is conserved AND the seed stays a compact body (no explosion/dissipation)."""
    p = LeniaParams(R=8.0, mu_k=0.5, sigma_k=0.15, mu_g=0.15, sigma_g=0.02, dt=0.2)
    N = 40
    w = FlowWorld((N, N, N), p, flow_clip=0.85)
    w.A = _blob((N, N, N), sigma=6.0)
    m0 = w.mass()
    for _ in range(50):
        w.step()
    assert abs(w.mass() / m0 - 1.0) < 1e-6
    assert w.A.max() > 0.1                          # still a real body, not dissipated
    assert (w.A > 0.05).mean() < 0.25              # and localized, not space-filling
