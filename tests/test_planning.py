"""Planning invariants: packing, planner beats pursuit, recurrence enables learning."""

import numpy as np

import genesis.planning as P


def test_param_pack_unpack_shapes():
    theta = np.arange(P.NPARAMS, dtype=float)
    Wh, Wi, bh, Wo, bo = P.unpack(theta)
    assert Wh.shape == (P.H, P.H) and Wi.shape == (P.H, P.IN)
    assert Wo.shape == (P.OUT, P.H) and bo.shape == (P.OUT,)


def test_episode_returns_valid_catch_time():
    t = P.episode(P.pursuit, seed=0)
    assert 1 <= t <= P.TMAX


def test_planner_beats_pursuit():
    """The model-based interceptor catches the circling target faster than pure pursuit."""
    seeds = range(200, 240)
    t_pursuit = P.mean_catch_time("pursuit", seeds=seeds)
    t_planner = P.mean_catch_time("planner", seeds=seeds)
    assert t_planner < t_pursuit                    # acting on prediction is faster
    assert t_planner < 0.8 * t_pursuit              # and clearly so


def test_recurrence_lets_a_controller_learn_to_anticipate():
    """A recurrent controller (can infer motion) evolves to catch more than feedforward."""
    rec, _ = P.evolve_es(recurrent=True, gens=45, seed=0)
    ff, _ = P.evolve_es(recurrent=False, gens=45, seed=0)
    r_rec = P.catch_rate("rnn", rec, True, range(200, 240))
    r_ff = P.catch_rate("rnn", ff, False, range(200, 240))
    assert r_rec > r_ff                             # recurrence helps
    assert r_rec > 0.2                              # and the recurrent one really catches
