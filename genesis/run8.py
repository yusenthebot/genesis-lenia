"""Round 8 — predator & prey: a world that pushes back.

A second species hunts the foragers. We run the two-species world and find robust,
honest results:

  TOP-DOWN REGULATION — predators hold the prey population far below its food-limited
  carrying capacity (prey ~14 with predators vs ~55 without): the prey's very numbers
  are now set by something hunting it.
  COEXISTENCE — both species persist with ongoing population fluctuations.
  CO-EVOLUTION (the honest surprise) — selection acts, but prey evolve LOWER evasion,
  not higher: in this world fleeing costs more foraging than it saves from predation,
  so "forage hard, accept the risk" out-reproduces "flee." Not a storybook arms race;
  the evolved answer to predation is to out-breed it.

Visualised: prey/predator population dynamics (with a no-predator control), the strategy
genes over time, and a GIF of predators (red) chasing prey (blue) through the food (green).
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
from genesis.predprey import PredPrey  # noqa: E402

SIZE = 150
T = 1900
# the coexistence + regulation regime found by probing
REGIME = dict(n_prey=18, n_pred=6, pred_eat=0.24, pred_feed=2.3, pred_decay=0.03,
              pred_e0=1.3, pred_repro=2.4, pred_cost=1.5, food_spawn=15,
              max_pred=22, max_prey=55)


def load_glider(path="outputs/round2_genome.json"):
    d = json.load(open(path))
    rule = Genome(**{k: d["rule"][k] for k in
                     ["mu_g", "sigma_g", "mu_k", "sigma_k", "R", "dt"]})
    return rule, np.array(d["patch"])


def run(rule, patch, seed=0, predators=True, record=False):
    spec = dict(REGIME)
    if not predators:
        spec["n_pred"] = 0
    pp = PredPrey((SIZE, SIZE), rule, patch, seed=seed, **spec)
    frames = []
    for t in range(T):
        pp.step()
        if record and t % 22 == 0:
            frames.append((pp.composite(), pp.history[-1][1], pp.history[-1][2], t))
    return np.array(pp.history), frames


def figure(hist, hist_noPred, out: Path):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor="white")
    t = hist[:, 0]
    ax0.plot(t, hist[:, 1], color="#2060d0", lw=1.8, label="prey (with predators)")
    ax0.plot(t, hist[:, 2], color="#c0303a", lw=1.8, label="predators")
    ax0.plot(hist_noPred[:, 0], hist_noPred[:, 1], color="#888", lw=1.5, ls="--",
             label="prey (NO predators)")
    reg = hist[-500:, 1].mean()
    free = hist_noPred[-500:, 1].mean()
    ax0.annotate(f"predators regulate prey:\n{free:.0f} -> {reg:.0f} "
                 f"({free/max(reg,1):.1f}x suppression)",
                 (0.42, 0.62), xycoords="axes fraction", fontsize=9, color="#2060d0")
    ax0.set_xlabel("time step"); ax0.set_ylabel("population")
    ax0.set_title("Top-down regulation + coexistence")
    ax0.legend(fontsize=8, loc="upper left")

    ax1.plot(t, hist[:, 3], color="#2060d0", lw=1.8, label="prey evasion gene")
    ax1.plot(t, hist[:, 4], color="#c0303a", lw=1.8, label="predator pursuit gene")
    ax1.set_xlabel("time step"); ax1.set_ylabel("mean heritable gene")
    ax1.set_title("Co-evolution: prey evolve to forage-hard, not flee")
    ax1.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def make_gif(frames, out: Path):
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(frames[0][0])
    ttl = ax.set_title("", fontsize=10)

    def upd(k):
        rgb, npy, npd, t = frames[k]
        im.set_data(rgb)
        ttl.set_text(f"t={t}   prey={npy}  predators={npd}")
        return im, ttl

    FuncAnimation(fig, upd, frames=len(frames), blit=False).save(
        out, writer=PillowWriter(fps=18))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 8: predator-prey")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rule, patch = load_glider()
    print("== predator-prey: two coupled species ==")
    hist, _ = run(rule, patch, seed=0, predators=True)
    hist_np, _ = run(rule, patch, seed=0, predators=False)
    reg = hist[-500:, 1].mean(); free = hist_np[-500:, 1].mean()
    print(f"  prey with predators ~{reg:.0f} vs no predators ~{free:.0f} "
          f"({free/max(reg,1):.1f}x suppression); predators end {int(hist[-1,2])}")
    print(f"  genes: prey evasion {hist[20,3]:.1f} -> {hist[-1,3]:.1f}, "
          f"predator pursuit {hist[20,4]:.1f} -> {hist[-1,4]:.1f}")

    figure(hist, hist_np, outdir / "round8_predprey.png")
    print(f"wrote {outdir/'round8_predprey.png'}")
    if args.gif:
        _, frames = run(rule, patch, seed=0, predators=True, record=True)
        make_gif(frames, outdir / "round8_predprey.gif")
        print(f"wrote {outdir/'round8_predprey.gif'}")

    (outdir / "round8_predprey.json").write_text(json.dumps(
        {"regime": REGIME, "size": SIZE, "T": T,
         "prey_with_pred": float(reg), "prey_no_pred": float(free),
         "evade_start": float(hist[20, 3]), "evade_end": float(hist[-1, 3]),
         "pursue_start": float(hist[20, 4]), "pursue_end": float(hist[-1, 4])},
        indent=2))
    print(f"wrote {outdir/'round8_predprey.json'}")


if __name__ == "__main__":
    main()
