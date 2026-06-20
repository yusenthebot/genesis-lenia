"""Emergence metrics — how we *measure*, rather than assume, that a world is alive.

Given a recorded history (T, *shape) we ask four questions an autopoietic
structure must answer well:

  alive       — does mass survive (neither vanish nor saturate the whole world)?
  localized   — is the mass concentrated in a small support (a body, not soup)?
  persistent  — is the mass roughly steady over the late phase (homeostasis)?
  locomotion  — does the centre of mass travel (a creature, not a fixed breather)?

These are deliberately substrate-level and cheap; richer notions of intelligence
(agency, prediction) build on top in later rounds.
"""

from __future__ import annotations

import numpy as np


def circular_centroid(field: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Mass-weighted centroid per axis, correct under periodic boundaries.

    Uses the circular mean (project positions onto a circle, average, project
    back) so a structure straddling the wrap edge is not torn in half.
    """
    w = np.where(field > threshold, field, 0.0)
    total = w.sum()
    if total <= 0:
        return np.array([s / 2 for s in field.shape], dtype=np.float64)
    out = np.empty(field.ndim, dtype=np.float64)
    for ax in range(field.ndim):
        n = field.shape[ax]
        # marginal mass along this axis
        other = tuple(a for a in range(field.ndim) if a != ax)
        marg = w.sum(axis=other) if field.ndim > 1 else w
        theta = 2.0 * np.pi * np.arange(n) / n
        vec = (marg * np.exp(1j * theta)).sum()
        ang = np.angle(vec) % (2.0 * np.pi)
        out[ax] = ang * n / (2.0 * np.pi)
    return out


def _wrap_delta(delta: np.ndarray, shape) -> np.ndarray:
    """Wrap a per-axis displacement into [-n/2, n/2] (shortest periodic path)."""
    shape = np.asarray(shape, dtype=np.float64)
    return (delta + shape / 2.0) % shape - shape / 2.0


def _circular_path_length(centroids: np.ndarray, shape) -> float:
    """Total centroid travel summed over steps, respecting periodic wrap."""
    if len(centroids) < 2:
        return 0.0
    diffs = _wrap_delta(np.diff(centroids, axis=0), shape)
    return float(np.sqrt((diffs ** 2).sum(axis=1)).sum())


def _circular_net_displacement(centroids: np.ndarray, shape) -> float:
    """Straight-line start->end distance, wrap-aware (drift, not oscillation)."""
    if len(centroids) < 2:
        return 0.0
    net = _wrap_delta(centroids[-1] - centroids[0], shape)
    return float(np.sqrt((net ** 2).sum()))


def window_drift(history: np.ndarray, lo: float, hi: float,
                 threshold: float = 0.1) -> dict:
    """Net centroid drift over the fractional window [lo, hi] of the run.

    Lets us show that 1D motion is *transient*: large early drift, ~0 late drift.
    """
    T = history.shape[0]
    spatial = history.shape[1:]
    a, b = int(T * lo), max(int(T * hi), int(T * lo) + 2)
    centroids = np.stack([circular_centroid(history[i], threshold)
                          for i in range(a, b)])
    net = _circular_net_displacement(centroids, spatial)
    span = max(spatial)
    n = len(centroids) - 1
    return {"net_widths": float(net / span),
            "drift_speed": float(net / (n + 1e-9))}


def analyze_history(history: np.ndarray, threshold: float = 0.1,
                    tail: float = 0.5) -> dict:
    """Compute emergence metrics from a (T, *shape) history array."""
    T = history.shape[0]
    spatial = history.shape[1:]
    ncells = int(np.prod(spatial))
    t0 = int(T * (1.0 - tail))

    mass = history.reshape(T, -1).sum(axis=1)
    tail_mass = mass[t0:]
    mean_mass = float(tail_mass.mean())
    mass_cv = float(tail_mass.std() / (tail_mass.mean() + 1e-9))

    final = history[-1]
    support_fraction = float((final > threshold).mean())

    centroids = np.stack([circular_centroid(history[i], threshold)
                          for i in range(t0, T)])
    n_steps = len(centroids) - 1
    path = _circular_path_length(centroids, spatial)
    net = _circular_net_displacement(centroids, spatial)
    span = max(spatial)
    drift_speed = net / (n_steps + 1e-9)               # net cells per step
    osc_speed = path / (n_steps + 1e-9)                # total path per step
    straightness = net / (path + 1e-9)                 # 1=glider, ~0=breather
    travel = net / span                                # net travel in world-widths

    exploded = support_fraction > 0.6 or mean_mass > 0.9 * ncells
    dead = mean_mass < 0.5 or support_fraction < 1e-4
    alive = bool(not exploded and not dead)

    localized = float(max(0.0, 1.0 - support_fraction / 0.5)) if alive else 0.0
    persistent = float(max(0.0, 1.0 - mass_cv / 0.2)) if alive else 0.0
    # locomotion rewards *directed* drift, not breathing in place
    locomotion = (float(min(1.0, drift_speed / 0.10) * min(1.0, straightness / 0.6))
                  if alive else 0.0)

    score = 0.0
    if alive:
        # weight homeostasis and a real body most; movement is a bonus
        score = 0.45 * persistent + 0.35 * localized + 0.20 * locomotion

    return {
        "alive": alive,
        "exploded": bool(exploded),
        "dead": bool(dead),
        "mean_mass": mean_mass,
        "mass_cv": mass_cv,
        "support_fraction": support_fraction,
        "drift_speed": float(drift_speed),
        "osc_speed": float(osc_speed),
        "straightness": float(straightness),
        "centroid_speed": float(drift_speed),  # back-compat alias
        "travel_widths": float(travel),
        "localized": localized,
        "persistent": persistent,
        "locomotion": locomotion,
        "score": float(score),
    }
