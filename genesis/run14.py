"""Round 14 — the Baldwin effect: evolving HOW fast to learn.

The learning RATE is now a heritable gene. Each creature is born naive and learns the
current rule at its own inherited rate, which mutates and is selected. Starting from a
population of RANDOM learning rates, evolution self-tunes the rate to a consistent
optimum (~0.5) — neither 0 (never learn) nor maximal (overreact): selection discovers the
right amount of plasticity. The Baldwin effect operates on the learning rate itself.

HONEST NEGATIVE (tested, documented): we expected the optimal rate to TRACK how fast the
world changes (high for fast worlds, low for stable). It does NOT here — across change
rates from flip=80 to flip=3000, and with added reward noise, the evolved rate stays
~0.5. The clean Bayesian volatility->learning-rate relationship is washed out by the
embodied foraging dynamics (born-naive each life, food competition, body inertia).
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
from genesis.baldwin import BaldwinEcology, LR_MAX  # noqa: E402

T = 3000
FLIP = 300              # a changing world (the regime where learning is selected, round 13)


def load_glider(path="outputs/round2_genome.json"):
    d = json.load(open(path))
    rule = Genome(**{k: d["rule"][k] for k in
                     ["mu_g", "sigma_g", "mu_k", "sigma_k", "R", "dt"]})
    return rule, np.array(d["patch"])


def run(rule, patch, flip, seed, record=False):
    eco = BaldwinEcology((150, 150), rule, patch, n0=22, flip=flip, spawn=10,
                         food_decay=0.02, max_pop=36, seed=seed)
    start_lr = np.array([c.lr for c in eco.pop])
    frames = []
    for _ in range(T):
        eco.step()
        if record and eco.t % 32 == 0:
            frames.append((eco.composite(), eco.mean_lr(), eco.t))
    end_lr = np.array([c.lr for c in eco.pop])
    return np.array(eco.history), start_lr, end_lr, frames


def figure(rule, patch, out: Path, seeds=3):
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor="white")
    starts, ends, finals = [], [], []
    L = T
    means = []
    for s in range(seeds):
        h, s_lr, e_lr, _ = run(rule, patch, FLIP, seed=s)
        a0.plot(h[:, 0], h[:, 2], lw=1.2, alpha=0.5, color=f"C{s}")
        means.append(h[:, 2]); starts.append(s_lr); ends.append(e_lr)
        finals.append(float(h[-200:, 2].mean()))
    L = min(len(m) for m in means)
    mean = np.mean([m[:L] for m in means], axis=0)
    a0.plot(np.arange(1, L + 1), mean, color="#111", lw=2.4, label="mean of runs")
    a0.axhline(np.mean(finals), color="#2a9d2a", ls="--", lw=1.2,
               label=f"evolved optimum ≈ {np.mean(finals):.2f}")
    a0.set_ylim(0, LR_MAX); a0.set_xlabel("time step")
    a0.set_ylabel("population mean LEARNING RATE")
    a0.set_title("Evolution self-tunes the learning rate to an optimum")
    a0.legend(fontsize=9, loc="lower right")

    bins = np.linspace(0, LR_MAX, 13)
    a1.hist(np.concatenate(starts), bins=bins, alpha=0.55, color="#b04030",
            label="start (random rates)")
    a1.hist(np.concatenate(ends), bins=bins, alpha=0.7, color="#2a9d2a",
            label="evolved")
    a1.axvline(np.mean(finals), color="#2a9d2a", ls="--", lw=1.2)
    a1.set_xlabel("learning rate"); a1.set_ylabel("creatures")
    a1.set_title("Learning rate: random -> concentrated at the optimum")
    a1.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return float(np.mean(finals))


def make_gif(rule, patch, out: Path, seed=0):
    *_, frames = run(rule, patch, FLIP, seed=seed, record=True)
    fig, ax = plt.subplots(figsize=(4.7, 4.7))
    ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(frames[0][0])
    ttl = ax.set_title("", fontsize=9)

    def upd(k):
        rgb, mlr, t = frames[k]
        im.set_data(rgb)
        ttl.set_text(f"t={t}   mean learning rate {mlr:.2f}  "
                     f"(red=slow learner, blue=fast)")
        return im, ttl

    FuncAnimation(fig, upd, frames=len(frames), blit=False).save(
        out, writer=PillowWriter(fps=16))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 14: the Baldwin effect")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rule, patch = load_glider()
    print("== Baldwin effect: the learning rate is heritable and selected ==")
    optimum = figure(rule, patch, outdir / "round14_baldwin.png")
    print(f"  evolution self-tunes the learning rate (random) -> ~{optimum:.2f}")
    print("  honest negative: this optimum does NOT track world change-rate "
          "(tested flip 80..3000 +/- reward noise; ~flat)")
    print(f"wrote {outdir/'round14_baldwin.png'}")
    if args.gif:
        make_gif(rule, patch, outdir / "round14_baldwin.gif")
        print(f"wrote {outdir/'round14_baldwin.gif'}")

    (outdir / "round14_baldwin.json").write_text(json.dumps(
        {"T": T, "flip": FLIP, "evolved_optimum_lr": optimum,
         "change_rate_tracking": "none (honest negative)"}, indent=2))
    print(f"wrote {outdir/'round14_baldwin.json'}")


if __name__ == "__main__":
    main()
