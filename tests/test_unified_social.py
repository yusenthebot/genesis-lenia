"""Unified-social invariants: full yield needs BOTH communication and coordination."""

import numpy as np

from genesis.unified_social import (
    N, K, NPARAMS, _unpack, yield_on, team_yield, evolve_es, PATTERNS,
)


def test_unpack_and_yield():
    sc, fo = _unpack(np.zeros(NPARAMS))
    assert len(sc) == 4 and len(fo) == 4
    y = yield_on(np.zeros(NPARAMS), PATTERNS[0], rng=np.random.default_rng(0))
    assert 0.0 <= y <= 1.0


def test_both_faculties_load_bearing():
    """Like round 21: ablate EITHER communication OR coordination and the team does worse."""
    theta, _ = evolve_es(gens=180, seed=0)
    full = team_yield(theta, seed=555)
    no_comm = team_yield(theta, ablate_comm=True, seed=555)
    no_coord = team_yield(theta, ablate_coord=True, seed=555)
    assert full > 0.9                       # full team covers the rich sites
    assert full > no_comm + 0.25            # communication is load-bearing
    assert full > no_coord + 0.25           # coordination is load-bearing
