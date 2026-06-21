"""Round 4 — SURVIVAL: now the creature must forage to stay alive.

We turn on metabolism: the creature carries an energy reserve that drains every step and
is refilled only by eating. When it empties, the body dissipates and the creature dies.
Food appears at random places over time, so survival demands *continual* foraging. We
evolve the forager to maximise lifetime and show the stakes with the decisive control:

  evolved (sensing on)  vs  ablated (sensing off)  vs  random drift

A creature that cannot sense food starves in ~100 steps; the evolved one lives
indefinitely by finding and eating food. Intelligence now has teeth: it is the
difference between living and dying.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from genesis.evolve import Genome, place_patch  # noqa: E402
from genesis.foraging import ForagingWorld  # noqa: E402

# the metabolic regime (world constants) found by probing: food is small and sparse,
# the energy buffer is shallow, so survival needs SKILLED foraging — a non-forager
# starves, a creature that aims at food lives indefinitely.
DECAY = 0.012       # energy drained per step
FEED = 0.06         # eaten food -> energy
ENERGY0 = 1.3       # starting reserve (~108 steps without food)
ETA = 0.3           # consumption rate
SPAWN = 85          # a new food cluster every SPAWN steps
FRAD = 7.0          # food cluster radius (small -> must be aimed at)
SIZE = 140
TMAX = 900


def load_glider(path="outputs/round2_genome.json"):
    d = json.load(open(path))
    rule = Genome(**{k: d["rule"][k] for k in
                     ["mu_g", "sigma_g", "mu_k", "sigma_k", "R", "dt"]})
    return rule, np.array(d["patch"])


def survival_episode(rule, patch, gamma, sense, seed, *, mode="gradient",
                     record=False, tmax=TMAX):
    """Run until the creature dies or tmax. Return lifetime (+ traces if record)."""
    w = ForagingWorld((SIZE, SIZE), rule.to_params(), sense_sigma=sense,
                      gamma=gamma, eta=ETA, decay=DECAY, feed=FEED, energy0=ENERGY0)
    w.A = place_patch((SIZE, SIZE), patch)
    for _ in range(30):
        w.step()
    m0 = w.A.sum()
    death = 0.25 * m0
    rng = np.random.default_rng(seed)
    fixed = None
    if mode == "random":
        a = rng.uniform(0, 2 * np.pi)
        fixed = np.array([np.cos(a), np.sin(a)])
    energy_tr, mass_tr, frames = [], [], []
    lifetime = tmax
    for t in range(tmax):
        if t % SPAWN == 0:
            ang = rng.uniform(0, 2 * np.pi)
            pos = (np.array([SIZE / 2, SIZE / 2])
                   + np.array([np.cos(ang), np.sin(ang)]) * 0.32 * SIZE
                   * rng.uniform(0.7, 1.2)) % SIZE
            w.add_food_blob(tuple(pos), radius=FRAD, amp=1.0)
        if mode == "random":
            w._drift += w.gamma * 0.06 * fixed
            for ax in range(2):
                s = int(round(w._drift[ax]))
                if s:
                    w.A = np.roll(w.A, s, axis=ax)
                    w._drift[ax] -= s
            g = w.gamma; w.gamma = 0.0; w.step(); w.gamma = g
        else:
            w.step()
        if record:
            energy_tr.append(w.energy)
            mass_tr.append(float(w.A.sum()))
            if t % 4 == 0:
                frames.append((np.clip(w.A, 0, 1).copy(),
                               np.clip(w.F, 0, 1).copy(), w.energy))
        if w.A.sum() < death or not w.alive:
            lifetime = t
            break
    if record:
        return lifetime, np.array(energy_tr), np.array(mass_tr), frames
    return lifetime


def mean_life(rule, patch, gamma, sense, seeds, **kw):
    return float(np.mean([survival_episode(rule, patch, gamma, sense, s, **kw)
                          for s in seeds]))


def evolve_survivor(rule, patch, seeds):
    best = None
    for gamma, sense in itertools.product([6.0, 9.0, 12.0], [14.0, 18.0]):
        score = mean_life(rule, patch, gamma, sense, seeds)
        if best is None or score > best[0]:
            best = (score, gamma, sense)
    return {"life": best[0], "gamma": best[1], "sense": best[2]}


def survival_figure(rule, patch, gamma, sense, lifes, out: Path):
    """Energy curves (forager refills vs ablated starves) + lifetime bars."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor="white",
                                   gridspec_kw={"width_ratios": [1.7, 1]})
    # energy traces for one episode each
    _, e_ev, _, _ = survival_episode(rule, patch, gamma, sense, 0, record=True)
    _, e_ab, _, _ = survival_episode(rule, patch, 0.0, sense, 0, record=True)
    ax0.plot(e_ev, color="#2a9d2a", lw=2, label="evolved (sensing on) — refills & lives")
    ax0.plot(e_ab, color="#b04030", lw=2, label="ablated (sensing off) — drains & dies")
    if len(e_ab) < TMAX:
        ax0.plot(len(e_ab) - 1, 0, "x", color="#b04030", ms=12, mew=3)
        ax0.annotate("death", (len(e_ab) - 1, 0), color="#b04030",
                     fontsize=10, ha="left", va="bottom")
    ax0.set_xlabel("time step"); ax0.set_ylabel("energy reserve")
    ax0.set_title("Survival depends on foraging")
    ax0.legend(fontsize=9, loc="upper right")

    labels = list(lifes.keys())
    vals = [lifes[k] for k in labels]
    ax1.bar(labels, vals, color=["#2a9d2a", "#b04030", "#808080"][:len(labels)])
    for i, v in enumerate(vals):
        ax1.text(i, v + TMAX * 0.01, f"{v:.0f}", ha="center", fontsize=11)
    ax1.set_ylabel("mean lifetime (steps, capped at %d)" % TMAX)
    ax1.set_title("Lifetime: sensing is survival")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def make_gif(rule, patch, gamma, sense, out: Path, seed=3):
    _, _, _, frames = survival_episode(rule, patch, gamma, sense, seed, record=True)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.set_xticks([]); ax.set_yticks([])
    A0, F0, e0 = frames[0]
    im = ax.imshow(np.dstack([A0, F0 * 0.9, np.zeros_like(A0)]))
    ttl = ax.set_title(f"energy {e0:.2f}", fontsize=10)

    def upd(k):
        A, F, e = frames[k]
        im.set_data(np.dstack([A, F * 0.9, np.zeros_like(A)]))
        ttl.set_text(f"alive — energy {e:.2f}   (red=creature, green=food)")
        return im, ttl

    FuncAnimation(fig, upd, frames=len(frames), blit=False).save(
        out, writer=PillowWriter(fps=20))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 4: forage to survive")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rule, patch = load_glider()
    train_seeds = list(range(4))
    test_seeds = list(range(100, 112))

    print(f"== metabolic regime: decay={DECAY} feed={FEED} energy0={ENERGY0} "
          f"food every {SPAWN} steps ==")
    print("== evolving the survivor (gain + sensing radius) ==")
    best = evolve_survivor(rule, patch, train_seeds)
    print(f"best gain={best['gamma']} sense={best['sense']} "
          f"(train lifetime {best['life']:.0f}/{TMAX})")

    print("== survival benchmark on UNSEEN food schedules ==")
    ev = mean_life(rule, patch, best["gamma"], best["sense"], test_seeds)
    ab = mean_life(rule, patch, 0.0, best["sense"], test_seeds)
    rd = mean_life(rule, patch, best["gamma"], best["sense"], test_seeds, mode="random")
    print(f"  evolved (sensing on): {ev:.0f} steps")
    print(f"  ablated (sensing off): {ab:.0f} steps")
    print(f"  random drift         : {rd:.0f} steps")

    lifes = {"evolved\n(sensing on)": ev, "ablated\n(sensing off)": ab,
             "random\ndrift": rd}
    survival_figure(rule, patch, best["gamma"], best["sense"], lifes,
                    outdir / "round4_survival.png")
    print(f"wrote {outdir/'round4_survival.png'}")
    if args.gif:
        make_gif(rule, patch, best["gamma"], best["sense"],
                 outdir / "round4_survival.gif")
        print(f"wrote {outdir/'round4_survival.gif'}")

    out = {"rule": rule.__dict__, "survivor": best,
           "regime": {"decay": DECAY, "feed": FEED, "energy0": ENERGY0,
                      "spawn": SPAWN, "tmax": TMAX},
           "lifetimes": {"evolved": ev, "ablated": ab, "random": rd},
           "test_seeds": test_seeds}
    (outdir / "round4_survivor.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {outdir/'round4_survivor.json'}")


if __name__ == "__main__":
    main()
