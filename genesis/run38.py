"""Round 38 — harder theory of mind: reading intent when behaviour MISLEADS.

The actor must DETOUR around a central obstacle to reach goals behind it, so its early motion points
AWAY from the goal. A position oracle (nearest goal to the actor's current position) is FOOLED while
the actor is on the wrong side; only an observer that has learned the obstacle + the go-around policy
infers the true goal mid-detour. Result: at the misleading moment (~3/4 through, actor still detouring)
the observer is ~0.99 while the position oracle is ~0.69; the oracle only catches up once the actor
finally arrives. The observer BEATS the position oracle on average — the genuine ToM R35 lacked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

import genesis.tom_obstacle as TM  # noqa: E402

COLORS = ["#d04030", "#2a9d2a", "#2060d0"]


def _draw_obstacle(ax):
    ax.add_patch(Circle((0, 0), TM.RO, color="#999", alpha=0.55, zorder=1))
    for k in range(TM.K):
        ax.plot(*TM.GOALS[k], "*", color=COLORS[k], ms=16, mec="k", zorder=4)
    ax.plot(*TM.START, "ko", ms=7, zorder=4)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-38, 38); ax.set_ylim(-40, 40)


def figure(theta, out: Path):
    acc = TM.accuracy_by_step(theta, seed=777, n=400)
    naive = TM.naive_accuracy_by_step(seed=777, n=400)
    abl = TM.accuracy_by_step(theta, ablate=True, seed=777, n=400)
    steps = np.arange(1, TM.T + 1)
    mid = 3 * TM.T // 4

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(13.8, 4.5), facecolor="white")
    a0.plot(steps, acc, color="#2a9d2a", lw=2.7, label="observer (models the detour)")
    a0.plot(steps, naive, color="#b04030", lw=2.4, label="position oracle (fooled by detour)")
    a0.plot(steps, abl, color="#888", lw=1.8, ls="--", label="ablated (random motion)")
    a0.axhline(1.0 / TM.K, color="k", ls=":", lw=1.0, label=f"chance ({1.0/TM.K:.2f})")
    a0.axvline(mid, color="#666", lw=1.0, alpha=0.5)
    a0.annotate("mid-detour:\nobserver right,\noracle fooled", (mid, 0.45), fontsize=8,
                ha="center", color="#333")
    a0.set_xlabel("steps of behaviour observed"); a0.set_ylabel("goal-inference accuracy")
    a0.set_ylim(0, 1.05); a0.set_title("Observer reads intent THROUGH the detour")
    a0.legend(fontsize=7.5, loc="lower right")

    # trajectories: the detour around the obstacle (why position misleads)
    rng = np.random.default_rng(3)
    for _ in range(12):
        g = rng.integers(TM.K)
        _, path = TM.actor_trajectory(g, rng)
        a1.plot(path[:, 0], path[:, 1], color=COLORS[g], lw=1.2, alpha=0.8, zorder=3)
    _draw_obstacle(a1)
    a1.set_title("actors DETOUR around the obstacle\n(early motion points away from the goal)")

    # snapshot at the misleading moment: actor mid-detour, observer right, oracle wrong
    rng = np.random.default_rng(6)
    g = 1
    steps_v, path = TM.actor_trajectory(g, rng)
    bel = TM.observer_beliefs(theta, steps_v, rng=rng)
    obs_guess = bel[mid - 1].argmax()
    naive_guess = np.argmin(np.linalg.norm(TM.GOALS - path[mid], axis=1))
    a2.plot(path[:mid + 1, 0], path[:mid + 1, 1], color="#333", lw=1.8, zorder=3)
    a2.plot(*path[mid], "o", color="#000", ms=9, zorder=5)
    _draw_obstacle(a2)
    a2.plot(*TM.GOALS[obs_guess], "o", mfc="none", mec="#2a9d2a", ms=24, mew=2.5, zorder=6)
    a2.plot(*TM.GOALS[naive_guess], "x", color="#b04030", ms=16, mew=3, zorder=6)
    a2.set_title(f"mid-detour: observer (o) -> goal {obs_guess} (true {g})\n"
                 f"oracle (x) -> goal {naive_guess} (fooled)")

    fig.suptitle("Round 38 — harder theory of mind: reading intent when behaviour MISLEADS "
                 "(a detour fools the position oracle)", fontsize=11, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=116)
    plt.close(fig)
    return dict(observer_mean=float(acc.mean()), naive_mean=float(naive.mean()),
                observer_mid=float(acc[mid - 1]), naive_mid=float(naive[mid - 1]),
                observer_final=float(acc[-1]), ablated=float(abl[-1]), chance=1.0 / TM.K)


def make_gif(theta, out: Path):
    rng = np.random.default_rng(9)
    g = 1
    steps_v, path = TM.actor_trajectory(g, rng)
    bel = TM.observer_beliefs(theta, steps_v, rng=rng)
    fig, (axw, axb) = plt.subplots(1, 2, figsize=(8.8, 4.6))
    _draw_obstacle(axw); axw.set_title("actor detours around obstacle", fontsize=10)
    (trail,) = axw.plot([], [], color="#333", lw=2, zorder=3)
    (dot,) = axw.plot([], [], "o", color="#000", ms=9, zorder=5)
    axb.set_ylim(0, 1); axb.set_xticks(range(TM.K)); axb.set_ylabel("belief")
    axb.set_title("observer's belief (true goal = green)", fontsize=10)
    bars = axb.bar(range(TM.K), [1.0 / TM.K] * TM.K, color=COLORS)

    def upd(t):
        trail.set_data(path[:t + 1, 0], path[:t + 1, 1])
        dot.set_data([path[t, 0]], [path[t, 1]])
        b = bel[min(t, TM.T - 1)] if t > 0 else np.full(TM.K, 1.0 / TM.K)
        for bar, v in zip(bars, b):
            bar.set_height(v)
        return [trail, dot] + list(bars)

    FuncAnimation(fig, upd, frames=TM.T + 1, blit=False).save(out, writer=PillowWriter(fps=6))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 38: harder theory of mind")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)

    print("== harder theory of mind: reading intent when behaviour misleads ==")
    theta, _ = TM.evolve_es(gens=170, seed=0)
    res = figure(theta, outdir / "round38_tom_obstacle.png")
    print(f"  mid-detour: observer {res['observer_mid']:.2f} vs oracle {res['naive_mid']:.2f}; "
          f"mean observer {res['observer_mean']:.2f} vs oracle {res['naive_mean']:.2f}; "
          f"ablated {res['ablated']:.2f}")
    print(f"wrote {outdir/'round38_tom_obstacle.png'}")
    if args.gif:
        make_gif(theta, outdir / "round38_tom_obstacle.gif")
        print(f"wrote {outdir/'round38_tom_obstacle.gif'}")
    (outdir / "round38_tom_obstacle.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
