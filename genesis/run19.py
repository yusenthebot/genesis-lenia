"""Round 19 — planning: acting on foresight to intercept a moving target.

The mind reacts, remembers, and predicts; the capstone verb is INTEND — using a model of
the future to choose actions that steer toward a goal. A target orbits on a circle. A
REACTIVE pursuer always heads at where the target IS and spirals along behind it; a PLANNER
heads where the target WILL be and cuts across to intercept. We show (1) the hand-coded
planner catches ~2x faster than reactive pursuit, with the classic pursuit-curve-vs-
interception trajectories; (2) an EVOLVED RECURRENT controller — given only the relative
position — LEARNS to anticipate (it infers the circular motion), far outperforming a
feedforward (reaction-only) controller. Acting on prediction, both proven and learned.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

import genesis.planning as P  # noqa: E402

GENS = 80


def figure(out: Path):
    rec, _ = P.evolve_es(recurrent=True, gens=GENS, seed=0)
    ff, _ = P.evolve_es(recurrent=False, gens=GENS, seed=0)
    test = range(200, 280)
    t_pur = P.mean_catch_time("pursuit", seeds=test)
    t_pln = P.mean_catch_time("planner", seeds=test)
    r_rec = P.catch_rate("rnn", rec, True, test)
    r_ff = P.catch_rate("rnn", ff, False, test)

    fig = plt.figure(figsize=(13.5, 4.6), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1])

    # trajectories on a representative seed
    ax0 = fig.add_subplot(gs[0, 0])
    _, pa_p, pt = P.episode(P.pursuit, seed=205, return_traj=True)
    _, pa_l, _ = P.episode(P.planner, seed=205, return_traj=True)
    ax0.plot(pt[:, 0], pt[:, 1], color="#bbb", lw=1.0, label="target (orbiting)")
    ax0.plot(pa_p[:, 0], pa_p[:, 1], color="#b04030", lw=1.8, label="reactive pursuit (tail-chase)")
    ax0.plot(pa_l[:, 0], pa_l[:, 1], color="#2a9d2a", lw=1.8, label="planner (cuts across)")
    ax0.plot(*pa_p[0], "ko", ms=5); ax0.plot(*pa_p[-1], "x", color="#b04030", ms=10, mew=2)
    ax0.plot(*pa_l[-1], "x", color="#2a9d2a", ms=10, mew=2)
    ax0.set_aspect("equal"); ax0.set_xticks([]); ax0.set_yticks([])
    ax0.set_title("Reaction spirals behind; planning cuts across")
    ax0.legend(fontsize=8, loc="upper left")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.bar(["reactive\npursuit", "planner\n(lead)"], [t_pur, t_pln],
            color=["#b04030", "#2a9d2a"])
    for i, v in enumerate([t_pur, t_pln]):
        ax1.text(i, v + 1, f"{v:.0f}", ha="center", fontsize=11)
    ax1.set_ylabel("steps to intercept (lower = better)")
    ax1.set_title("Planning catches ~2x faster")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.bar(["recurrent\n(can model)", "feedforward\n(reaction only)"], [r_rec, r_ff],
            color=["#2a9d2a", "#b04030"])
    for i, v in enumerate([r_rec, r_ff]):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=11)
    ax2.set_ylim(0, 1.0); ax2.set_ylabel("catch rate (evolved from scratch)")
    ax2.set_title("A creature LEARNS to anticipate")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return dict(pursuit_time=t_pur, planner_time=t_pln,
                recurrent_rate=r_rec, feedforward_rate=r_ff)


def make_gif(out: Path, seed=205):
    _, pa_p, pt = P.episode(P.pursuit, seed=seed, return_traj=True)
    _, pa_l, _ = P.episode(P.planner, seed=seed, return_traj=True)
    n = max(len(pa_p), len(pa_l))
    lo = pt.min() - 8; hi = pt.max() + 8
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.plot(pt[:, 0], pt[:, 1], color="#ddd", lw=1.0)
    (tgt,) = ax.plot([], [], "o", color="#2060d0", ms=10)
    (pur,) = ax.plot([], [], "-", color="#b04030", lw=1.6)
    (pln,) = ax.plot([], [], "-", color="#2a9d2a", lw=1.6)
    ttl = ax.set_title("", fontsize=9)

    def upd(k):
        i = min(k, len(pt) - 1)
        tgt.set_data([pt[i, 0]], [pt[i, 1]])
        ip = min(k, len(pa_p) - 1); il = min(k, len(pa_l) - 1)
        pur.set_data(pa_p[:ip + 1, 0], pa_p[:ip + 1, 1])
        pln.set_data(pa_l[:il + 1, 0], pa_l[:il + 1, 1])
        ttl.set_text(f"t={k}   blue=target  red=pursuit (chases)  green=planner (intercepts)")
        return tgt, pur, pln, ttl

    FuncAnimation(fig, upd, frames=n, blit=False).save(
        out, writer=PillowWriter(fps=18))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 19: planning / interception")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("== planning: intercept a circling target (act on prediction) ==")
    res = figure(outdir / "round19_planning.png")
    print(f"  hand-coded catch time: reactive pursuit {res['pursuit_time']:.0f} vs "
          f"planner {res['planner_time']:.0f}")
    print(f"  evolved catch rate: recurrent {res['recurrent_rate']:.2f} vs "
          f"feedforward {res['feedforward_rate']:.2f}")
    print(f"wrote {outdir/'round19_planning.png'}")
    if args.gif:
        make_gif(outdir / "round19_planning.gif")
        print(f"wrote {outdir/'round19_planning.gif'}")

    (outdir / "round19_planning.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {outdir/'round19_planning.json'}")


if __name__ == "__main__":
    main()
