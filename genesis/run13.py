"""Round 13 — learning under selection: does knowing more make a creature win?

Round 12 showed learners KNOW more about the world (0.69 bits). Here learners (plastic
brains) and fixed-reflex creatures (a non-plastic inherited strategy) COMPETE in one
world, with 'is_learner' heritable. The answer is conditional and clean:

  STABLE world  -> the learner fraction FALLS: fixed reflexes win, learning's startup
                   cost buys nothing when the world never changes.
  CHANGING world -> the learner fraction RISES to ~1: learners adapt within life while the
                    fixed strategy, tuned only across generations, can't keep up.

Sweeping the world's change-rate traces WHEN learning pays. Knowing is only worth its cost
in a world that changes within a lifetime — the evolution-of-learning (Baldwin) result.
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

from genesis.evolve import Genome  # noqa: E402
from genesis.evo_learning import EvoLearning  # noqa: E402

T = 2600
STABLE = 100_000        # effectively no reversal
CHANGING = 300


def load_glider(path="outputs/round2_genome.json"):
    d = json.load(open(path))
    rule = Genome(**{k: d["rule"][k] for k in
                     ["mu_g", "sigma_g", "mu_k", "sigma_k", "R", "dt"]})
    return rule, np.array(d["patch"])


def run(rule, patch, flip, seed, record=False):
    eco = EvoLearning((150, 150), rule, patch, n0=20, learner_frac=0.5, flip=flip,
                      spawn=10, food_decay=0.02, max_pop=36, seed=seed)
    frames = []
    for _ in range(T):
        eco.step()
        if record and eco.t % 28 == 0:
            frames.append((eco.composite(), eco.learner_fraction(), eco.t))
    h = np.array(eco.history)
    frac = h[:, 1] / (h[:, 1] + h[:, 2] + 1e-9)
    return h[:, 0], frac, frames


def figure(rule, patch, out: Path):
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor="white")
    finals = {}
    for flip, color, label in [(STABLE, "#d06010", "stable world"),
                               (CHANGING, "#2060d0", "changing world")]:
        fr_runs = []
        for s in range(2):
            t, frac, _ = run(rule, patch, flip, seed=s)
            a0.plot(t, frac, color=color, lw=1.0, alpha=0.4)
            fr_runs.append(frac)
        L = min(len(f) for f in fr_runs)
        mean = np.mean([f[:L] for f in fr_runs], axis=0)
        a0.plot(t[:L], mean, color=color, lw=2.4, label=label)
        finals[flip] = float(mean[-1])
    a0.axhline(0.5, color="#bbb", ls=":", lw=1)
    a0.set_ylim(0, 1.02); a0.set_xlabel("time step"); a0.set_ylabel("fraction of population that LEARNS")
    a0.set_title("Learning is selected — only in a changing world")
    a0.legend(fontsize=9, loc="center right")

    # when does learning pay? final learner fraction vs reversal interval
    sweep = [200, 400, 900, 2000, STABLE]
    fracs = []
    for flip in sweep:
        if flip in finals:
            fracs.append(finals[flip]); continue
        _, frac, _ = run(rule, patch, flip, seed=0)
        fracs.append(float(frac[-100:].mean()))
    xs = [min(f, 4000) for f in sweep]          # cap the 'stable' point for the log axis
    a1.plot(xs, fracs, "o-", color="#2a9d2a", lw=2, ms=7)
    a1.axhline(0.5, color="#bbb", ls=":", lw=1)
    a1.set_xscale("log"); a1.set_ylim(0, 1.02)
    a1.set_xlabel("reversal interval (steps) — slower / more stable world ->")
    a1.set_ylabel("final fraction that LEARNS")
    a1.set_title("When does learning pay? (vs world change-rate)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return finals


def make_gif(rule, patch, out: Path, seed=0):
    _, _, frames = run(rule, patch, CHANGING, seed=seed, record=True)
    fig, ax = plt.subplots(figsize=(4.7, 4.7))
    ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(frames[0][0])
    ttl = ax.set_title("", fontsize=9)

    def upd(k):
        rgb, lf, t = frames[k]
        im.set_data(rgb)
        ttl.set_text(f"changing world, t={t}   learners (blue): {lf*100:.0f}%")
        return im, ttl

    FuncAnimation(fig, upd, frames=len(frames), blit=False).save(
        out, writer=PillowWriter(fps=16))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 13: learning under selection")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rule, patch = load_glider()
    print("== learning under selection: learners vs fixed-reflex creatures compete ==")
    finals = figure(rule, patch, outdir / "round13_selection.png")
    print(f"  final learner fraction: stable world {finals[STABLE]:.2f}, "
          f"changing world {finals[CHANGING]:.2f}")
    print(f"wrote {outdir/'round13_selection.png'}")
    if args.gif:
        make_gif(rule, patch, outdir / "round13_selection.gif")
        print(f"wrote {outdir/'round13_selection.gif'}")

    (outdir / "round13_selection.json").write_text(json.dumps(
        {"T": T, "stable_learner_frac": finals[STABLE],
         "changing_learner_frac": finals[CHANGING]}, indent=2))
    print(f"wrote {outdir/'round13_selection.json'}")


if __name__ == "__main__":
    main()
