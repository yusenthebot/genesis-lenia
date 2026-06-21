"""Round 9 — within-lifetime learning: a creature that learns, then re-learns.

Two food sources, one nutritious at a time, the rule REVERSING partway through life.
A creature with a plastic value-learning brain tracks which source pays and, crucially,
RE-LEARNS each time the rule flips — adaptation within a single lifetime, not across
generations. The same creature with plasticity off (lr=0) cannot. Visualised as the
accuracy-over-life curve (learner re-adapts at every reversal; ablated stays at chance)
and a GIF of the agent following the rewarding source as it flips.
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

from genesis.learning import run_episode, windowed_accuracy, summary  # noqa: E402

L = 120
SOURCES = ((60, 32), (60, 88))
FLIP = 320
T = 2560


def figure(out: Path, seeds=8):
    lr_acc, ab_acc = [], []
    for s in range(seeds):
        lt, _ = run_episode(L=L, sources=SOURCES, flip=FLIP, T=T, lr=0.25, seed=s)
        at, _ = run_episode(L=L, sources=SOURCES, flip=FLIP, T=T, lr=0.0, seed=s)
        lr_acc.append(summary(lt)["accuracy"]); ab_acc.append(summary(at)["accuracy"])
    # representative curves (seed 0)
    lt, _ = run_episode(L=L, sources=SOURCES, flip=FLIP, T=T, lr=0.25, seed=0)
    at, _ = run_episode(L=L, sources=SOURCES, flip=FLIP, T=T, lr=0.0, seed=0)
    lx, la = windowed_accuracy(lt, T)
    ax_, aa = windowed_accuracy(at, T)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor="white",
                                 gridspec_kw={"width_ratios": [2.2, 1]})
    for f in range(FLIP, T, FLIP):
        a0.axvline(f, color="#ccc", lw=1)
    a0.plot(lx, la, color="#2a9d2a", lw=1.8, label="learner (plastic brain)")
    a0.plot(ax_, aa, color="#b04030", lw=1.6, label="ablated (no learning)")
    a0.axhline(0.5, color="#bbb", ls=":", lw=1)
    a0.set_ylim(0, 1.05); a0.set_xlabel("time step (grey lines = rule reversals)")
    a0.set_ylabel("rolling accuracy (tastes that hit the active source)")
    a0.set_title("Within-lifetime learning: re-learns after every reversal")
    a0.legend(fontsize=9, loc="lower right")

    a1.bar(["learner", "ablated"], [np.mean(lr_acc), np.mean(ab_acc)],
           color=["#2a9d2a", "#b04030"])
    for i, v in enumerate([np.mean(lr_acc), np.mean(ab_acc)]):
        a1.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=11)
    a1.axhline(0.5, color="#bbb", ls=":", lw=1)
    a1.set_ylim(0, 1.05); a1.set_ylabel(f"mean accuracy ({seeds} lives)")
    a1.set_title("Learning vs none")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return float(np.mean(lr_acc)), float(np.mean(ab_acc))


def make_gif(out: Path, seed=0):
    _, frames = run_episode(L=L, sources=SOURCES, flip=FLIP, T=T, lr=0.25,
                            seed=seed, record=True)
    src = np.array(SOURCES, dtype=float)
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    circles = [plt.Circle(src[i][::-1], 9, color="#2a9d2a") for i in range(len(src))]
    for c in circles:
        ax.add_patch(c)
    (dot,) = ax.plot([], [], "o", ms=12, color="#d030a0", mec="k")
    ttl = ax.set_title("", fontsize=10)

    def upd(k):
        p, active, V, t = frames[k]
        for i, c in enumerate(circles):
            c.set_color("#39e639" if i == active else "#19451a")  # active = bright
        dot.set_data([p[1]], [p[0]])
        pref = int(np.argmax(V))
        ttl.set_text(f"t={t}   nutritious source: {active}   "
                     f"brain prefers: {pref}")
        return (dot, ttl, *circles)

    FuncAnimation(fig, upd, frames=len(frames), blit=False).save(
        out, writer=PillowWriter(fps=20))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 9: within-lifetime learning")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("== reversal-learning task: 2 sources, rule flips every "
          f"{FLIP} steps ==")
    lr_acc, ab_acc = figure(outdir / "round9_learning.png")
    print(f"  learner accuracy {lr_acc:.2f} vs ablated {ab_acc:.2f} "
          f"(chance 0.50) -> the plastic brain tracks reversals")
    print(f"wrote {outdir/'round9_learning.png'}")
    if args.gif:
        make_gif(outdir / "round9_learning.gif")
        print(f"wrote {outdir/'round9_learning.gif'}")

    (outdir / "round9_learning.json").write_text(json.dumps(
        {"sources": SOURCES, "flip": FLIP, "T": T,
         "learner_accuracy": lr_acc, "ablated_accuracy": ab_acc}, indent=2))
    print(f"wrote {outdir/'round9_learning.json'}")


if __name__ == "__main__":
    main()
