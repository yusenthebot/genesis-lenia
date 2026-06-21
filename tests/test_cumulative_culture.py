"""Cumulative-culture invariants: the ratchet needs BOTH transmission and innovation."""

import numpy as np

from genesis.cumulative_culture import (
    N, TARGET, quality, lifetime, random_design, run_lineage, GENS,
)


def test_quality_and_lifetime():
    rng = np.random.default_rng(0)
    assert quality(TARGET) > 0.95                      # the target itself is high quality
    d = random_design(rng)
    improved = lifetime(d, 80, rng)
    assert quality(improved) >= quality(d)             # a lifetime never makes it worse (hill-climb)


def test_ratchet_needs_transmission_and_innovation():
    cum = run_lineage("cumulative", seed=0)[0][-1]
    ind = run_lineage("individual", seed=0)[0][-1]
    tra = run_lineage("transmit", seed=0)[0][-1]
    assert cum > 0.7                    # cumulative ratchets to a high-quality artifact
    assert cum > ind + 0.4              # ... far beyond what individual (restart) reaches
    assert cum > tra + 0.4              # ... and beyond transmit-only (no innovation)
    assert ind < 0.3 and tra < 0.2      # neither alone accumulates
