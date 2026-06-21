"""Round 12 — measuring the mind.

We stop reporting task scores and give the mind a NUMBER: the mutual information, in
bits, between the creature's internal state (which food its brain prefers) and the
world's hidden variable (which food is nutritious). A learner's brain encodes the world
(~0.7 bits); the same body with plasticity off encodes 0. Sweeping how fast the world
reverses traces the mind's OPERATING ENVELOPE — the rate of change it can still track.
A GIF shows a live "knowledge meter" rising and falling as the creature learns and the
world flips.
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
from genesis.embodied_learning import run_life  # noqa: E402
from genesis.measure import mutual_information_bits, brain_world_series, windowed_mi  # noqa: E402

T = 2400
FLIP = 400
GAIN = 0.6
FOOD_DECAY = 0.015
KNOW_WIN = 600          # window for the live "recent knowledge" meter (> one flip)


def load_glider(path="outputs/round2_genome.json"):
    d = json.load(open(path))
    rule = Genome(**{k: d["rule"][k] for k in
                     ["mu_g", "sigma_g", "mu_k", "sigma_k", "R", "dt"]})
    return rule, np.array(d["patch"])


def _mi(rule, patch, lr, w0, flip, seed):
    el, _ = run_life(rule, patch, T=T, lr=lr, w0=w0, seed=seed, flip=flip,
                     gain=GAIN, food_decay=FOOD_DECAY)
    b, w = brain_world_series(el.hist)
    return mutual_information_bits(b, w)


def figure(rule, patch, out: Path, seeds=4):
    ml = float(np.mean([_mi(rule, patch, 0.3, (0, 0), FLIP, s) for s in range(seeds)]))
    ma = float(np.mean([_mi(rule, patch, 0.0, (1, 1), FLIP, s) for s in range(seeds)]))
    flips = [100, 200, 400, 800, 1600]
    env = [float(np.mean([_mi(rule, patch, 0.3, (0, 0), f, s) for s in range(3)]))
           for f in flips]

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor="white",
                                 gridspec_kw={"width_ratios": [1, 1.5]})
    a0.bar(["learner\n(plastic brain)", "ablated\n(no learning)"], [ml, ma],
           color=["#2a9d2a", "#b04030"])
    for i, v in enumerate([ml, ma]):
        a0.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=12)
    a0.axhline(1.0, color="#bbb", ls=":", lw=1)
    a0.text(1.0, 1.0, " 1 bit ceiling", color="#888", fontsize=8, va="bottom")
    a0.set_ylim(0, 1.1); a0.set_ylabel("I(brain ; world)  [bits]")
    a0.set_title("How much the mind KNOWS about the world")

    a1.plot(flips, env, "o-", color="#2060d0", lw=2, ms=7)
    a1.axhline(0, color="#bbb", lw=0.8)
    a1.set_xscale("log")
    a1.set_xlabel("reversal interval (steps) — slower world ->")
    a1.set_ylabel("I(brain ; world)  [bits]")
    a1.set_title("The mind's operating envelope: faster change -> less knowable")
    a1.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return ml, ma, dict(zip(flips, env))


def make_gif(rule, patch, out: Path, seed=0):
    el, frames = run_life(rule, patch, T=T, lr=0.3, w0=(0, 0), seed=seed, flip=FLIP,
                          gain=GAIN, food_decay=FOOD_DECAY, record=True)
    b, w = brain_world_series(el.hist)
    ts, mi = windowed_mi(b, w, win=KNOW_WIN, stride=4)
    fig, (axc, axm) = plt.subplots(1, 2, figsize=(8.4, 4.3))
    axc.set_xticks([]); axc.set_yticks([]); axc.set_title("the creature", fontsize=9)
    im = axc.imshow(frames[0][0])
    axm.set_xlim(0, T); axm.set_ylim(0, 1.0)
    axm.set_xlabel("time"); axm.set_ylabel("I(brain ; world) [bits], recent")
    axm.set_title("the mind's knowledge of the world", fontsize=9)
    (line,) = axm.plot([], [], color="#2a9d2a", lw=2)

    def upd(k):
        rgb, active, wv, t = frames[k]
        im.set_data(rgb)
        m = ts <= t
        line.set_data(ts[m], mi[m])
        cur = mi[m][-1] if m.any() else 0.0
        axm.set_title(f"the mind's knowledge: {cur:.2f} bits", fontsize=9)
        return im, line

    FuncAnimation(fig, upd, frames=len(frames), blit=False).save(
        out, writer=PillowWriter(fps=16))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 12: measure the mind")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rule, patch = load_glider()
    print("== measuring the mind: I(brain ; world) in bits ==")
    ml, ma, env = figure(rule, patch, outdir / "round12_measure.png")
    print(f"  I(brain;world): learner {ml:.2f} bits vs ablated {ma:.2f} bits")
    print(f"  operating envelope (flip -> bits): "
          + ", ".join(f"{k}:{v:.2f}" for k, v in env.items()))
    print(f"wrote {outdir/'round12_measure.png'}")
    if args.gif:
        make_gif(rule, patch, outdir / "round12_measure.gif")
        print(f"wrote {outdir/'round12_measure.gif'}")

    (outdir / "round12_measure.json").write_text(json.dumps(
        {"learner_bits": ml, "ablated_bits": ma, "envelope": env,
         "T": T, "flip": FLIP}, indent=2))
    print(f"wrote {outdir/'round12_measure.json'}")


if __name__ == "__main__":
    main()
