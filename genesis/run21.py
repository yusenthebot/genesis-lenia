"""Round 21 — unification: one creature, all four faculties, proven load-bearing.

A Lenia creature must stay on food that MOVES and FLASHES, under metabolism. One integrated
controller perceives -> remembers -> predicts -> plans/acts to track it and survive. We
ablate each faculty and watch survival fall:

  full        memory + prediction + planning   -> tracks the food, survives longest
  memory_only remembers last position, no predict -> lags the moving food, starves sooner
  no_memory   nothing held across the dark      -> stalls when the food is invisible, dies first

Each step down the ladder removes one faculty and costs survival -> every faculty earns its place.
The body (rounds 3-13), memory (15/18), prediction (16) and planning (19) are now ONE organism.
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
from genesis.unified import run, survival  # noqa: E402

MODES = [
    ("full", "full\n(memory+predict+plan)", "#2a9d2a"),
    ("memory_only", "memory only\n(no prediction)", "#d09030"),
    ("no_memory", "no memory\n(stalls in the dark)", "#b04030"),
]
SEEDS = range(0, 20)


def load_glider(path="outputs/round2_genome.json"):
    d = json.load(open(path))
    rule = Genome(**{k: d["rule"][k] for k in
                     ["mu_g", "sigma_g", "mu_k", "sigma_k", "R", "dt"]})
    return rule, np.array(d["patch"])


def _energy_trace(rule, patch, mode, seed):
    _, _, frames = run(rule, patch, mode, T=320, seed=seed, record=True)
    ts = [f[3] for f in frames]
    en = [f[2] for f in frames]
    return ts, en


def figure(rule, patch, out: Path):
    surv = {m: survival(rule, patch, m, SEEDS) for m, _, _ in MODES}

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor="white")
    labels = [lab for _, lab, _ in MODES]
    vals = [surv[m] for m, _, _ in MODES]
    colors = [c for _, _, c in MODES]
    a0.bar(labels, vals, color=colors)
    for i, v in enumerate(vals):
        a0.text(i, v + 3, f"{v:.0f}", ha="center", fontsize=12)
    a0.set_ylabel("steps survived (max 320, 20 seeds)")
    a0.set_title("Remove any faculty -> the creature starves sooner")

    seed = 0
    for m, lab, c in MODES:
        ts, en = _energy_trace(rule, patch, m, seed)
        a1.plot(ts, en, color=c, lw=2.0, label=lab.replace("\n", " "))
    a1.axhline(0, color="k", lw=0.8, ls=":")
    a1.set_xlabel("time step"); a1.set_ylabel("energy reserve")
    a1.set_title("Energy over one life (0 = death)")
    a1.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return surv


def make_gif(rule, patch, out: Path, seed=0):
    _, _, frames = run(rule, patch, "full", T=320, seed=seed, record=True)
    shape = frames[0][0].shape
    fig, ax = plt.subplots(figsize=(4.7, 4.7))
    ax.set_xticks([]); ax.set_yticks([])
    A0, pf0, e0, t0, vis0 = frames[0]
    fimg = np.zeros(shape)
    im = ax.imshow(np.dstack([A0, fimg, np.zeros(shape)]), vmin=0, vmax=1)
    (fdot,) = ax.plot([], [], "o", color="#20c020", ms=12, mew=2, mec="white")
    ttl = ax.set_title("", fontsize=9)

    def upd(k):
        A, pf, e, t, vis = frames[k]
        im.set_data(np.dstack([np.clip(A, 0, 1), np.zeros(shape), np.zeros(shape)]))
        fdot.set_data([pf[1]], [pf[0]])
        fdot.set_alpha(1.0 if vis else 0.25)
        sig = "food VISIBLE" if vis else "food dark — tracking on memory+prediction"
        ttl.set_text(f"t={t}  energy={e:.2f}   [{sig}]")
        return im, fdot, ttl

    FuncAnimation(fig, upd, frames=len(frames), blit=False).save(
        out, writer=PillowWriter(fps=20))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 21: unification")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rule, patch = load_glider()
    print("== unification: one creature, body+memory+prediction+planning, survival by ablation ==")
    surv = figure(rule, patch, outdir / "round21_unified.png")
    print(f"  survival: full={surv['full']:.0f}  memory_only={surv['memory_only']:.0f}  "
          f"no_memory={surv['no_memory']:.0f}  (max 320)")
    print(f"wrote {outdir/'round21_unified.png'}")
    if args.gif:
        make_gif(rule, patch, outdir / "round21_unified.gif")
        print(f"wrote {outdir/'round21_unified.gif'}")

    (outdir / "round21_unified.json").write_text(json.dumps(surv, indent=2))
    print(f"wrote {outdir/'round21_unified.json'}")


if __name__ == "__main__":
    main()
