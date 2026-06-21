"""Round 7 — evolution RUNNING inside the ecology.

Foraging gain is heritable; well-fed creatures reproduce (offspring inherit the gain
plus a mutation), starving ones die. Starting from a population of RANDOM gains, natural
selection drives the mean gain to the ecology's optimum on its own — the same intermediate
value round 6 found by sweeping. Selection is no longer measured; it is happening.

Visualised: the mean-gain trajectory converging across independent runs (with round 6's
swept optimum for reference), the gain distribution before vs after, and a GIF of the
evolving population (colour = foraging gain) shifting toward the optimal strategy.
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
from genesis.evo_ecology import EvoEcology, GMAX  # noqa: E402

SIZE = 160
T = 2200
SPAWN = 75          # scarce food -> food-limited population -> real selection
FRAD = 9.0
ROUND6_OPTIMUM = 4.0


def load_glider(path="outputs/round2_genome.json"):
    d = json.load(open(path))
    rule = Genome(**{k: d["rule"][k] for k in
                     ["mu_g", "sigma_g", "mu_k", "sigma_k", "R", "dt"]})
    return rule, np.array(d["patch"])


def _eco(rule, patch, seed):
    return EvoEcology((SIZE, SIZE), rule, patch, n0=16, spawn=SPAWN, frad=FRAD,
                      repro_energy=2.6, repro_cost=1.6, max_pop=30, seed=seed)


def evolve_run(rule, patch, seed, record=False):
    eco = _eco(rule, patch, seed)
    start = np.array([c.gamma for c in eco.pop])
    frames = []
    for t in range(T):
        eco.step()
        if record and t % 12 == 0:
            frames.append((eco.composite(), eco.mean_gamma(), len(eco.pop), t))
    end = np.array([c.gamma for c in eco.pop])
    return np.array(eco.history), start, end, frames


def figure(hists, start, end, out: Path):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor="white")
    for i, h in enumerate(hists):
        ax0.plot(h[:, 0], h[:, 2], lw=1.6, alpha=0.9,
                 label=f"run {i+1}" if i == 0 else None, color=f"C{i}")
    ax0.axhline(ROUND6_OPTIMUM, color="#2a9d2a", ls="--", lw=1.5,
                label=f"round-6 swept optimum ≈ {ROUND6_OPTIMUM:.0f}")
    ax0.set_xlabel("time step"); ax0.set_ylabel("population mean foraging gain")
    ax0.set_ylim(0, GMAX)
    ax0.set_title("Selection running: mean gain self-tunes to the optimum")
    ax0.legend(fontsize=9, loc="upper right")

    bins = np.linspace(0, GMAX, 13)
    ax1.hist(start, bins=bins, alpha=0.55, color="#b04030", label="start (random gains)")
    ax1.hist(end, bins=bins, alpha=0.7, color="#2a9d2a", label="evolved")
    ax1.axvline(ROUND6_OPTIMUM, color="#2a9d2a", ls="--", lw=1.2)
    ax1.set_xlabel("foraging gain"); ax1.set_ylabel("creatures")
    ax1.set_title("Gain distribution: random -> concentrated at the optimum")
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
        rgb, mg, n, t = frames[k]
        im.set_data(rgb)
        ttl.set_text(f"t={t}   pop={n}   mean gain={mg:.1f}")
        return im, ttl

    FuncAnimation(fig, upd, frames=len(frames), blit=False).save(
        out, writer=PillowWriter(fps=20))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 7: evolution inside the ecology")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rule, patch = load_glider()
    print(f"== evolving ecology: {args.runs} runs, scarce food (spawn {SPAWN}) ==")
    hists, starts, ends = [], [], []
    for s in range(args.runs):
        h, st, en, _ = evolve_run(rule, patch, s)
        hists.append(h); starts.append(st); ends.append(en)
        print(f"  run {s}: mean gain {st.mean():.1f} (random) -> {en.mean():.1f} "
              f"(evolved), final pop {int(h[-1,1])}")
    start_all = np.concatenate(starts)
    end_all = np.concatenate(ends)
    print(f"  pooled: {start_all.mean():.1f} -> {end_all.mean():.1f} "
          f"(round-6 swept optimum ~{ROUND6_OPTIMUM:.0f})")
    figure(hists, start_all, end_all, outdir / "round7_evolution.png")
    print(f"wrote {outdir/'round7_evolution.png'}")
    if args.gif:
        _, _, _, frames = evolve_run(rule, patch, 0, record=True)
        make_gif(frames, outdir / "round7_evolution.gif")
        print(f"wrote {outdir/'round7_evolution.gif'}")

    (outdir / "round7_evolution.json").write_text(json.dumps(
        {"spawn": SPAWN, "size": SIZE, "T": T, "runs": args.runs,
         "start_mean_gain": float(start_all.mean()),
         "evolved_mean_gain": float(end_all.mean()),
         "round6_optimum": ROUND6_OPTIMUM}, indent=2))
    print(f"wrote {outdir/'round7_evolution.json'}")


if __name__ == "__main__":
    main()
