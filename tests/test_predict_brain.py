"""Predictive-brain invariants: packing, sequence run, and prediction needs a model."""

import numpy as np

from genesis.predict_brain import (
    H, OBS, NPARAMS, unpack, run_seq, evolve_es,
)


def test_param_pack_unpack_shapes():
    theta = np.arange(NPARAMS, dtype=float)
    Wh, Wi, bh, wo = unpack(theta)
    assert Wh.shape == (H, H) and Wi.shape == (H, OBS)
    assert bh.shape == (H,) and wo.shape == (H,)
    assert NPARAMS == H * H + H * OBS + H + H


def test_run_seq_returns_valid_accuracy():
    rng = np.random.default_rng(0)
    theta = rng.normal(0, 0.5, NPARAMS)
    acc, acc_trans = run_seq(theta, recurrent=True, n_periods=20)
    assert 0.0 <= acc <= 1.0 and 0.0 <= acc_trans <= 1.0


def test_recurrent_predicts_ambiguous_world_feedforward_cannot():
    theta_r, _ = evolve_es(recurrent=True, gens=45, seed=0)
    theta_f, _ = evolve_es(recurrent=False, gens=45, seed=0)
    acc_r, trans_r = run_seq(theta_r, True, n_periods=80)
    acc_f, _ = run_seq(theta_f, False, n_periods=80)
    assert acc_r > 0.9                         # models the world -> predicts the flips
    assert trans_r > 0.9                       # gets the hard transition steps right
    assert acc_f < 0.65                        # reactive -> chance on an ambiguous pattern


def test_ablating_recurrence_destroys_prediction():
    """A trained recurrent net run WITHOUT recurrence loses its predictive power."""
    theta_r, _ = evolve_es(recurrent=True, gens=45, seed=1)
    acc_ablated, _ = run_seq(theta_r, recurrent=False, n_periods=80)
    assert acc_ablated < 0.7
