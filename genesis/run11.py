"""Round 11 — embodied learning: a plastic brain inside a Lenia creature.

The learner of round 9 was a detached point; here the same kind of plasticity lives
inside a real Lenia body. The creature forages two food types; its drift is a plastic
policy w over them; only one type is nutritious and the rule reverses mid-life. We show:

  THE BRAIN LEARNS (and re-learns): the drift weights track which food is nutritious and
  FLIP after every reversal — argmax(w)==active ~0.9 of the time.
  BEHAVIOURAL PAYOFF: with food that evaporates if uneaten (so the creature must actively
  chase fresh food), the embodied learner eats ~0.89 nutritious vs ~0.52 for the same body
  with plasticity off. The learned policy genuinely steers the body to the right food.

This unites body (round 2) + agency (round 3) + within-life learning (round 9) in ONE creature.
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
from genesis.embodied_learning import run_life, nutritious_fraction  # noqa: E402

T = 2400
FLIP = 400
GAIN = 0.6
FOOD_DECAY = 0.015      # food evaporates if uneaten -> creature must actively forage fresh food


def load_glider(path="outputs/round2_genome.json"):
    d = json.load(open(path))
    rule = Genome(**{k: d["rule"][k] for k in
                     ["mu_g", "sigma_g", "mu_k", "sigma_k", "R", "dt"]})
    return rule, np.array(d["patch"])


def _scores(eco):
    h = np.array(eco.hist)
    active = h[:, 1].astype(int)
    nut = np.where(active == 0, h[:, 2], h[:, 3]); tot = h[:, 2] + h[:, 3]
    nutfrac = float(nut.sum() / (tot.sum() + 1e-9))
    prefers_A = h[:, 4] > h[:, 5]
    brain_correct = float((prefers_A == (active == 0)).mean())
    return nutfrac, brain_correct


def figure(rule, patch, out: Path, seeds=5):
    lf, af, bc = [], [], []
    for s in range(seeds):
        el, _ = run_life(rule, patch, T=T, lr=0.3, w0=(0, 0), seed=s, flip=FLIP, gain=GAIN, food_decay=FOOD_DECAY)
        ea, _ = run_life(rule, patch, T=T, lr=0.0, w0=(1, 1), seed=s, flip=FLIP, gain=GAIN, food_decay=FOOD_DECAY)
        n_l, c_l = _scores(el); n_a, _ = _scores(ea)
        lf.append(n_l); af.append(n_a); bc.append(c_l)
    # representative run for the trajectory panels
    el, _ = run_life(rule, patch, T=T, lr=0.3, w0=(0, 0), seed=0, flip=FLIP, gain=GAIN, food_decay=FOOD_DECAY)
    ea, _ = run_life(rule, patch, T=T, lr=0.0, w0=(1, 1), seed=0, flip=FLIP, gain=GAIN, food_decay=FOOD_DECAY)
    h = np.array(el.hist)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor="white")
    # weights tracking the active type, flipping at reversals
    for f in range(FLIP, T, FLIP):
        a0.axvline(f, color="#ddd", lw=1)
    for k in range(0, T, FLIP):                              # shade which food is nutritious
        act = (k // FLIP) % 2
        a0.axvspan(k, min(k + FLIP, T), color="#dfeffb" if act == 0 else "#fdebd9",
                   alpha=0.6, lw=0)
    a0.plot(h[:, 0], h[:, 4], color="#2060d0", lw=1.6, label="w_A (drift toward food A)")
    a0.plot(h[:, 0], h[:, 5], color="#d06010", lw=1.6, label="w_B (drift toward food B)")
    a0.axhline(0, color="#bbb", lw=0.8)
    a0.set_xlabel("time step (shade = which food is nutritious; lines = reversals)")
    a0.set_ylabel("plastic drift weight")
    a0.set_title(f"The embodied brain learns & flips (argmax(w)=active {np.mean(bc):.0%})")
    a0.legend(fontsize=8, loc="lower left")

    lx, la = nutritious_fraction(el.hist); _, aa = nutritious_fraction(ea.hist)
    a1.plot(lx, la, color="#2a9d2a", lw=1.6, label="learner (plastic)")
    a1.plot(lx[:len(aa)], aa, color="#b04030", lw=1.4, label="ablated (fixed w)")
    a1.axhline(0.5, color="#bbb", ls=":", lw=1)
    a1.set_ylim(0, 1.0); a1.set_xlabel("time step")
    a1.set_ylabel("rolling fraction of food eaten that is nutritious")
    a1.set_title(f"Behavioural payoff: {np.mean(lf):.2f} vs {np.mean(af):.2f} ablated")
    a1.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return float(np.mean(lf)), float(np.mean(af)), float(np.mean(bc))


def make_gif(rule, patch, out: Path, seed=0):
    _, frames = run_life(rule, patch, T=T, lr=0.3, w0=(0, 0), seed=seed, flip=FLIP,
                         gain=GAIN, food_decay=FOOD_DECAY, record=True)
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(frames[0][0])
    ttl = ax.set_title("", fontsize=9)

    def upd(k):
        rgb, active, w, t = frames[k]
        im.set_data(rgb)
        pref = "A" if w[0] > w[1] else "B"
        nutr = "A" if active == 0 else "B"
        ttl.set_text(f"t={t}   nutritious: food {nutr}   brain chases: {pref}")
        return im, ttl

    FuncAnimation(fig, upd, frames=len(frames), blit=False).save(
        out, writer=PillowWriter(fps=20))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 11: embodied learning")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rule, patch = load_glider()
    print("== embodied learning: plastic brain in a Lenia creature, reversal task ==")
    lf, af, bc = figure(rule, patch, outdir / "round11_embodied.png")
    print(f"  brain preference correct {bc:.0%} (tracks + flips at reversals)")
    print(f"  nutritious eaten: learner {lf:.2f} vs ablated {af:.2f} (chance 0.50)")
    print(f"wrote {outdir/'round11_embodied.png'}")
    if args.gif:
        make_gif(rule, patch, outdir / "round11_embodied.gif")
        print(f"wrote {outdir/'round11_embodied.gif'}")

    (outdir / "round11_embodied.json").write_text(json.dumps(
        {"T": T, "flip": FLIP, "gain": GAIN, "brain_correct": bc,
         "learner_nutfrac": lf, "ablated_nutfrac": af}, indent=2))
    print(f"wrote {outdir/'round11_embodied.json'}")


if __name__ == "__main__":
    main()
