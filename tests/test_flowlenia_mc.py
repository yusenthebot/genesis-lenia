"""Multi-channel Flow-Lenia invariants: PER-CHANNEL mass conservation + coupled creatures form."""

import numpy as np

from genesis.flowlenia_mc import FlowWorldMC, two_channel_conn


def _seed(w, N):
    yy, xx = np.indices((N, N)); c = N // 2
    w.A[0] = 0.5 * np.exp(-0.5 * ((yy - c) ** 2 + (xx - c) ** 2) / 7.0 ** 2)
    w.A[1] = 0.5 * np.exp(-0.5 * ((yy - c - 3) ** 2 + (xx - c) ** 2) / 6.0 ** 2)


def test_per_channel_mass_conserved():
    N = 80
    w = FlowWorldMC((N, N), 2, two_channel_conn(R=11.0), flow_clip=0.9)
    _seed(w, N)
    m0 = w.total()
    for _ in range(60):
        w.step()
    m1 = w.total()
    assert np.allclose(m1, m0, rtol=1e-5)          # EACH channel conserves its own mass


def test_coupled_creature_forms_and_localizes():
    N = 80
    w = FlowWorldMC((N, N), 2, two_channel_conn(R=11.0), flow_clip=0.9)
    _seed(w, N)
    for _ in range(80):
        w.step()
    comb = w.combined()
    assert comb.max() > 0.1                          # a real body
    assert (comb > 0.05).mean() < 0.3               # localized, not space-filling
    assert w.A[0].max() > 0.1 and w.A[1].max() > 0.1  # both channels persist (coupled)
