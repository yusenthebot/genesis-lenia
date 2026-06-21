"""Planning — acting on foresight: intercepting a moving target.

The mind can predict (round 16); the last verb is INTEND — using a model of the future to
CHOOSE actions that steer toward a goal. The test is interception: a target moves on a
predictable path, and an agent (slightly faster) must catch it. A REACTIVE pursuer always
heads to where the target IS and perpetually lags (a tail-chase). A PLANNER predicts where
the target WILL be and heads to the intercept point. We show: (1) a model-based planner
crushes reactive pursuit; (2) an EVOLVED RECURRENT controller — given only the relative
position each step — DISCOVERS interception (it infers the target's motion from history and
leads it), while a feedforward controller can only pursue. Acting on prediction, learned.
"""

from __future__ import annotations

import numpy as np

L = 100.0
CIRCLE_R = 30.0   # the target orbits a centre on a circle (predictable, periodic, bounded)
OMEGA = 0.05      # target angular speed -> tangential speed = CIRCLE_R * OMEGA
VA = 1.55         # agent speed (~ the target's tangential speed): a pursuer tail-chases
VT = CIRCLE_R * OMEGA
TMAX = 260
CATCH_R = 4.5

# recurrent controller: obs [rel_x, rel_y, bias] -> heading (2D)
H = 8
IN = 3
OUT = 2
NPARAMS = H * H + H * IN + H + OUT * H + OUT


def unpack(theta):
    i = 0
    Wh = theta[i:i + H * H].reshape(H, H); i += H * H
    Wi = theta[i:i + H * IN].reshape(H, IN); i += H * IN
    bh = theta[i:i + H]; i += H
    Wo = theta[i:i + OUT * H].reshape(OUT, H); i += OUT * H
    bo = theta[i:i + OUT]
    return Wh, Wi, bh, Wo, bo


def _unit(d):
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else np.zeros(2)


def pursuit(pa, pt, vt):
    """Reactive: head straight at the target's CURRENT position."""
    return _unit(pt - pa)


def planner(pa, pt, vt):
    """Model-based: solve for the interception point using the target's velocity."""
    r = pt - pa
    a = vt @ vt - VA ** 2
    b = 2 * (r @ vt)
    c = r @ r
    tau = None
    if abs(a) < 1e-9:
        if abs(b) > 1e-9:
            tau = -c / b
    else:
        disc = b * b - 4 * a * c
        if disc >= 0:
            roots = [(-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a)]
            pos = [t for t in roots if t > 1e-6]
            if pos:
                tau = min(pos)
    aim = pt + vt * tau if (tau and tau > 0) else pt
    return _unit(aim - pa)


def _init(rng):
    """Target orbits a centre on a circle; agent starts somewhere off the circle."""
    c = np.array([0.5 * L, 0.5 * L]) + rng.uniform(-0.1 * L, 0.1 * L, 2)
    th0 = rng.uniform(0, 2 * np.pi)
    direction = rng.choice([-1.0, 1.0])
    pa = rng.uniform(0.15 * L, 0.85 * L, 2)
    return c, th0, direction, pa


def _target(c, th, direction):
    pt = c + CIRCLE_R * np.array([np.cos(th), np.sin(th)])
    vt = direction * CIRCLE_R * OMEGA * np.array([-np.sin(th), np.cos(th)])  # tangent
    return pt, vt


def episode(controller, seed=0, return_traj=False):
    """Hand-coded controller(pa, pt, vt) -> heading. Returns catch time (TMAX if missed)."""
    rng = np.random.default_rng(seed)
    c, th, direction, pa = _init(rng)
    pt, vt = _target(c, th, direction)
    ta, tt = [pa.copy()], [pt.copy()]
    for t in range(TMAX):
        head = controller(pa.copy(), pt.copy(), vt.copy())
        pa = pa + VA * head
        th += direction * OMEGA
        pt, vt = _target(c, th, direction)
        ta.append(pa.copy()); tt.append(pt.copy())
        if np.linalg.norm(pa - pt) < CATCH_R:
            return (t + 1, np.array(ta), np.array(tt)) if return_traj else t + 1
    return (TMAX, np.array(ta), np.array(tt)) if return_traj else TMAX


def episode_rnn(theta, recurrent, seed=0, return_traj=False):
    """Evolved controller: sees ONLY relative position (must infer motion to plan).

    Returns (catch_time, min_distance) — or (catch_time, traj_a, traj_t) if return_traj.
    """
    rng = np.random.default_rng(seed)
    c, th, direction, pa = _init(rng)
    pt, vt = _target(c, th, direction)
    Wh, Wi, bh, Wo, bo = unpack(theta)
    if not recurrent:
        Wh = np.zeros_like(Wh)
    h = np.zeros(H)
    ta, tt = [pa.copy()], [pt.copy()]
    min_d = np.linalg.norm(pa - pt)
    for t in range(TMAX):
        rel = (pt - pa) / L
        h = np.tanh(Wh @ h + Wi @ np.array([rel[0], rel[1], 1.0]) + bh)
        head = _unit(Wo @ h + bo)
        pa = pa + VA * head
        th += direction * OMEGA
        pt, vt = _target(c, th, direction)
        ta.append(pa.copy()); tt.append(pt.copy())
        d = np.linalg.norm(pa - pt)
        min_d = min(min_d, d)
        if d < CATCH_R:
            return (t + 1, np.array(ta), np.array(tt)) if return_traj else (t + 1, min_d)
    return (TMAX, np.array(ta), np.array(tt)) if return_traj else (TMAX, min_d)


def fitness_rnn(theta, recurrent, seeds):
    """Dense shaped reward: credit for closing distance, bonus for catching fast."""
    rs = []
    for s in seeds:
        ct, md = episode_rnn(theta, recurrent, seed=s)
        if ct < TMAX:
            rs.append(0.5 + 0.5 * (TMAX - ct) / TMAX)     # caught -> 0.5..1.0
        else:
            rs.append(0.5 * max(0.0, 1.0 - md / L))        # not caught -> 0..0.5 by closeness
    return float(np.mean(rs))


def evolve_es(recurrent=True, gens=70, pop=36, sigma=0.25, lr=0.2, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 0.4, NPARAMS)
    curve = []
    train = tuple(range(8))
    for _ in range(gens):
        eps = rng.normal(0, 1, (pop, NPARAMS))
        fits = np.array([fitness_rnn(theta + sigma * eps[i], recurrent, train)
                         for i in range(pop)])
        adv = (fits - fits.mean()) / (fits.std() + 1e-9)
        theta = theta + lr / (pop * sigma) * (eps.T @ adv)
        curve.append(fitness_rnn(theta, recurrent, train))
    return theta, np.array(curve)


def mean_catch_time(kind, theta=None, recurrent=True, seeds=range(50)):
    if kind == "pursuit":
        return float(np.mean([episode(pursuit, seed=s) for s in seeds]))
    if kind == "planner":
        return float(np.mean([episode(planner, seed=s) for s in seeds]))
    return float(np.mean([episode_rnn(theta, recurrent, seed=s)[0] for s in seeds]))


def catch_rate(kind, theta=None, recurrent=True, seeds=range(50)):
    """Fraction of episodes where the target is caught within TMAX."""
    if kind in ("pursuit", "planner"):
        fn = pursuit if kind == "pursuit" else planner
        return float(np.mean([episode(fn, seed=s) < TMAX for s in seeds]))
    return float(np.mean([episode_rnn(theta, recurrent, seed=s)[0] < TMAX for s in seeds]))
