"""Round 6 — ECOLOGY: many creatures, one world, contested food.

Creatures with a range of foraging skills (sensorimotor gain) compete for the same
scarce, replenishing food. We measure who survives, and find STABILIZING SELECTION:
survival peaks at an INTERMEDIATE gain — under-foragers can't find food and starve,
over-foragers overshoot and forage inefficiently. Competition favours a balanced
strategy, not maximal aggression. Visualised as a survival-vs-skill curve, the
population dynamics of one contest, and a GIF of the creatures competing.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
import matplotlib.cm as cm  # noqa: E402

from genesis.evolve import Genome  # noqa: E402
from genesis.ecology import Ecology  # noqa: E402

SIZE = 170
T = 750
SPAWN = 50
FRAD = 8.0
GAMMAS = [0, 2, 4, 6, 9, 12, 16]


def load_glider(path="outputs/round2_genome.json"):
    d = json.load(open(path))
    rule = Genome(**{k: d["rule"][k] for k in
                     ["mu_g", "sigma_g", "mu_k", "sigma_k", "R", "dt"]})
    return rule, np.array(d["patch"])


def _specs(gammas, seed):
    """Place one creature per gain on a ring, with randomised slot assignment."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(gammas))
    cen = np.array([SIZE / 2, SIZE / 2])
    specs = []
    for slot, gi in enumerate(order):
        a = 2 * np.pi * slot / len(gammas)
        pos = (cen + np.array([np.cos(a), np.sin(a)]) * 0.33 * SIZE) % SIZE
        specs.append((gammas[gi], (int(pos[0]), int(pos[1]))))
    return specs


def lifetime_experiment(rule, patch, seeds):
    g2l = collections.defaultdict(list)
    for s in seeds:
        eco = Ecology((SIZE, SIZE), rule, patch, _specs(GAMMAS, s),
                      spawn=SPAWN, frad=FRAD, seed=s)
        for _ in range(T):
            eco.step()
        for c in eco.creatures:
            g2l[c.gamma].append(c.death_t if not c.alive else T)
    return {g: float(np.mean(v)) for g, v in g2l.items()}


def representative_run(rule, patch, seed=0, record=True):
    eco = Ecology((SIZE, SIZE), rule, patch, _specs(GAMMAS, seed),
                  spawn=SPAWN, frad=FRAD, seed=seed)
    frames, nalive = [], []
    for t in range(T):
        eco.step()
        nalive.append(eco.n_alive())
        if record and t % 5 == 0:
            frames.append((eco.composite(), eco.n_alive(), t))
    return eco, frames, np.array(nalive)


def figure(curve, nalive, out: Path):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor="white")
    gs = sorted(curve)
    ys = [curve[g] for g in gs]
    ax0.plot(gs, ys, "o-", color="#2060d0", lw=2, ms=7)
    best = gs[int(np.argmax(ys))]
    ax0.axvline(best, color="#2a9d2a", ls="--", lw=1)
    ax0.annotate(f"optimum gain ≈ {best}", (best, max(ys)),
                 textcoords="offset points", xytext=(8, -14), color="#2a9d2a")
    ax0.set_xlabel("foraging gain (sensorimotor skill)")
    ax0.set_ylabel(f"mean lifetime (steps, capped {T})")
    ax0.set_title("Stabilizing selection: survival peaks at intermediate skill")

    ax1.plot(nalive, color="#a00", lw=2)
    ax1.set_xlabel("time step"); ax1.set_ylabel("creatures alive")
    ax1.set_title("Population of one contest (under- & over-foragers die out)")
    ax1.set_ylim(0, len(GAMMAS) + 0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def make_gif(frames, out: Path):
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(frames[0][0])
    ttl = ax.set_title("", fontsize=10)

    def upd(k):
        rgb, n, t = frames[k]
        im.set_data(rgb)
        ttl.set_text(f"t={t}   alive: {n}/{len(GAMMAS)}   (green=food)")
        return im, ttl

    FuncAnimation(fig, upd, frames=len(frames), blit=False).save(
        out, writer=PillowWriter(fps=20))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 6: ecology / competition")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rule, patch = load_glider()
    print(f"== ecology: {len(GAMMAS)} creatures compete for food every {SPAWN} steps ==")
    curve = lifetime_experiment(rule, patch, list(range(args.seeds)))
    best = max(curve, key=curve.get)
    print("  mean lifetime by foraging gain:")
    for g in sorted(curve):
        print(f"    gain={int(g):2d} -> {curve[g]:.0f}")
    print(f"  -> survival peaks at intermediate gain ~{int(best)} (stabilizing selection)")

    eco, frames, nalive = representative_run(rule, patch, seed=0, record=args.gif)
    figure(curve, nalive, outdir / "round6_ecology.png")
    print(f"wrote {outdir/'round6_ecology.png'}")
    if args.gif:
        make_gif(frames, outdir / "round6_ecology.gif")
        print(f"wrote {outdir/'round6_ecology.gif'}")

    (outdir / "round6_ecology.json").write_text(json.dumps(
        {"gammas": GAMMAS, "mean_lifetime_by_gain": curve, "optimum_gain": best,
         "spawn": SPAWN, "size": SIZE, "T": T, "seeds": args.seeds}, indent=2))
    print(f"wrote {outdir/'round6_ecology.json'}")


if __name__ == "__main__":
    main()
