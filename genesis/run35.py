"""Round 35 — theory of mind: inferring a hidden goal from behaviour.

An ACTOR moves toward one of K hidden goals with heavy noise; an OBSERVER sees only the actor's
step-by-step MOTION (displacements, no absolute position) and must infer WHICH goal. The observer
is a recurrent net (evolved) that INTEGRATES the noisy motion into a belief over goals. Its belief
SHARPENS as it watches (chance -> confident), and ablating the observation (random motion) drops it
to chance. Mentalising intent from behaviour — a social-intelligence axis distinct from communication.
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

import genesis.theory_of_mind as TOM  # noqa: E402

COLORS = ["#d04030", "#2a9d2a", "#2060d0", "#d0a020"]


def figure(theta, out: Path):
    acc = TOM.accuracy_by_step(theta, seed=777, n=400)
    abl = TOM.accuracy_by_step(theta, ablate=True, seed=777, n=400)
    steps = np.arange(1, TOM.T + 1)

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(13.5, 4.4), facecolor="white")
    a0.plot(steps, acc, color="#2a9d2a", lw=2.6, label="observer (reads motion)")
    a0.plot(steps, abl, color="#b04030", lw=2.2, label="ablated (random motion)")
    a0.axhline(1.0 / TOM.K, color="k", ls=":", lw=1.2, label=f"chance ({1.0/TOM.K:.2f})")
    a0.set_xlabel("steps of behaviour observed"); a0.set_ylabel("goal-inference accuracy")
    a0.set_ylim(0, 1.0); a0.set_title("Intent read progressively from behaviour")
    a0.legend(fontsize=8, loc="center right")

    # belief sharpening, AVERAGED over many episodes (robust, not a cherry-picked run):
    # mean belief assigned to the TRUE goal vs the best WRONG goal, after each step.
    rng = np.random.default_rng(5)
    ptrue = np.zeros(TOM.T); pother = np.zeros(TOM.T); n = 300
    for _ in range(n):
        g = rng.integers(TOM.K)
        sv, _ = TOM.actor_trajectory(g, rng)
        bel = TOM.observer_beliefs(theta, sv, rng=rng)
        ptrue += bel[:, g]
        mask = np.ones(TOM.K, bool); mask[g] = False
        pother += bel[:, mask].max(1)
    ptrue /= n; pother /= n
    a1.plot(steps, ptrue, color="#2a9d2a", lw=2.6, label="belief in the TRUE goal")
    a1.plot(steps, pother, color="#b04030", lw=2.2, label="belief in the best WRONG goal")
    a1.axhline(1.0 / TOM.K, color="k", ls=":", lw=1.0)
    a1.set_xlabel("steps observed"); a1.set_ylabel("mean belief"); a1.set_ylim(0, 1.0)
    a1.set_title("belief SHARPENS on the true goal\n(averaged over episodes)"); a1.legend(fontsize=8)

    # a few actor trajectories, coloured by whether the observer inferred correctly
    rng = np.random.default_rng(2)
    for _ in range(8):
        gg = rng.integers(TOM.K)
        sv, _ = TOM.actor_trajectory(gg, rng)
        pos = np.cumsum(sv, axis=0)
        pos = np.vstack([[0, 0], pos])
        inferred = TOM.observer_beliefs(theta, sv, rng=rng)[-1].argmax()
        col = "#2a9d2a" if inferred == gg else "#b04030"
        a2.plot(pos[:, 0], pos[:, 1], color=col, lw=1.2, alpha=0.8)
    for k in range(TOM.K):
        a2.plot(*TOM.TARGETS[k], "*", color=COLORS[k], ms=15, mec="k")
    a2.plot(0, 0, "ko", ms=6); a2.set_aspect("equal"); a2.set_xticks([]); a2.set_yticks([])
    a2.set_title("trajectories (green=inferred right)")

    fig.suptitle("Round 35 — theory of mind: an observer infers another agent's hidden goal "
                 "from its behaviour", fontsize=11, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=118)
    plt.close(fig)
    return dict(final_accuracy=float(acc[-1]), ablated=float(abl[-1]),
                chance=1.0 / TOM.K, accuracy_step5=float(acc[4]))


def make_gif(theta, out: Path):
    rng = np.random.default_rng(11)
    g = 2
    sv, _ = TOM.actor_trajectory(g, rng)
    pos = np.vstack([[0, 0], np.cumsum(sv, axis=0)])
    bel = TOM.observer_beliefs(theta, sv, rng=rng)
    fig, (axw, axb) = plt.subplots(1, 2, figsize=(8.6, 4.4))
    lim = TOM.D + 8
    axw.set_xlim(-lim, lim); axw.set_ylim(-lim, lim); axw.set_aspect("equal")
    axw.set_xticks([]); axw.set_yticks([]); axw.set_title("the actor moves", fontsize=10)
    for k in range(TOM.K):
        axw.plot(*TOM.TARGETS[k], "*", color=COLORS[k], ms=15, mec="k")
    (path,) = axw.plot([], [], color="#444", lw=2)
    (dot,) = axw.plot([], [], "o", color="#222", ms=8)
    axb.set_ylim(0, 1); axb.set_xticks(range(TOM.K)); axb.set_ylabel("belief")
    axb.set_title("observer's belief over goals", fontsize=10)
    bars = axb.bar(range(TOM.K), [0.25] * TOM.K, color=COLORS)

    def upd(t):
        path.set_data(pos[:t + 1, 0], pos[:t + 1, 1])
        dot.set_data([pos[t, 0]], [pos[t, 1]])
        b = bel[min(t, TOM.T - 1)] if t > 0 else np.full(TOM.K, 0.25)
        for bar, v in zip(bars, b):
            bar.set_height(v)
        return [path, dot] + list(bars)

    FuncAnimation(fig, upd, frames=TOM.T + 1, blit=False).save(
        out, writer=PillowWriter(fps=4))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 35: theory of mind")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)

    print("== theory of mind: infer a hidden goal from behaviour ==")
    theta, _ = TOM.evolve_es(gens=160, seed=0)
    res = figure(theta, outdir / "round35_theory_of_mind.png")
    print(f"  goal-inference accuracy: {res['final_accuracy']:.2f} (after 5 steps "
          f"{res['accuracy_step5']:.2f}) vs ablated {res['ablated']:.2f} (chance {res['chance']:.2f})")
    print(f"wrote {outdir/'round35_theory_of_mind.png'}")
    if args.gif:
        make_gif(theta, outdir / "round35_theory_of_mind.gif")
        print(f"wrote {outdir/'round35_theory_of_mind.gif'}")
    (outdir / "round35_theory_of_mind.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
