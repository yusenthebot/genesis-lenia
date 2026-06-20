"""Round 3 — AGENCY: a creature that senses food and steers to it to survive.

We take the round-2 glider (an emergent Lenia body) and give it a sensorimotor reflex:
it senses the food gradient over its body and drifts up it (mass-exact, no blow-up).
We then EVOLVE the sensorimotor gain + sensing radius to forage well across *random*
food layouts, and prove the behaviour is real agency with the decisive control:

  evolved (sensing on)  vs  ablated (sensing off, same body)  vs  random drift

Agency = the ablated/ random creatures cannot reach food placed in arbitrary
directions; the evolved one turns toward food wherever it is. Verified by the
foraging numbers AND by trajectories that visibly bend toward food.
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
from genesis.metrics import circular_centroid  # noqa: E402


def load_glider(path="outputs/round2_genome.json"):
    d = json.load(open(path))
    rule = Genome(**{k: d["rule"][k] for k in
                     ["mu_g", "sigma_g", "mu_k", "sigma_k", "R", "dt"]})
    return rule, np.array(d["patch"])


def _food_positions(size, n, seed):
    rng = np.random.default_rng(seed)
    c = np.array([size / 2, size / 2])
    dist = 0.32 * size
    out = []
    for _ in range(n):
        ang = rng.uniform(0, 2 * np.pi)
        out.append(tuple((c + np.array([np.cos(ang), np.sin(ang)]) * dist
                          * rng.uniform(0.85, 1.15)) % size))
    return out


def forage_episode(rule, patch, gamma, sense, seed, *, size=140, steps=360,
                   settle=30, n_food=1, mode="gradient", record=False):
    """Run one foraging episode; return fraction of food eaten (+ path/food)."""
    w = ForagingWorld((size, size), rule.to_params(), sense_sigma=sense,
                      gamma=gamma, eta=0.25)
    w.A = place_patch((size, size), patch)
    for _ in range(settle):
        w.step()
    foods = _food_positions(size, n_food, seed)
    for f in foods:
        w.add_food_blob(f, radius=9.0, amp=1.0)
    f0 = w.F.sum() + 1e-9
    # a "random drift" control: steer toward a fixed random direction, not food
    if mode == "random":
        rng = np.random.default_rng(seed + 9991)
        ang = rng.uniform(0, 2 * np.pi)
        w._fixed_dir = np.array([np.cos(ang), np.sin(ang)])
    path = []
    for _ in range(steps):
        if mode == "random":
            # bypass sensing: rigidly drift toward the fixed direction
            w._drift += w.gamma * 0.06 * w._fixed_dir
            for ax in range(2):
                s = int(round(w._drift[ax]))
                if s:
                    w.A = np.roll(w.A, s, axis=ax)
                    w._drift[ax] -= s
            g = w.gamma
            w.gamma = 0.0
            w.step()
            w.gamma = g
        else:
            w.step()
        if record:
            path.append(circular_centroid(w.A))
    if record:
        return w.eaten / f0, np.array(path), foods, w
    return w.eaten / f0


def mean_forage(rule, patch, gamma, sense, seeds, **kw):
    return float(np.mean([forage_episode(rule, patch, gamma, sense, s, **kw)
                          for s in seeds]))


def evolve_forager(rule, patch, seeds):
    """Coarse search over the sensorimotor genes (gain, sensing radius)."""
    best = None
    for gamma, sense in itertools.product([3.0, 5.0, 7.0, 10.0], [12.0, 16.0, 22.0]):
        score = mean_forage(rule, patch, gamma, sense, seeds)
        if best is None or score > best[0]:
            best = (score, gamma, sense)
    return {"score": best[0], "gamma": best[1], "sense": best[2]}


def wrapped_segments(path, size):
    """Path in wrapped coords, with NaN breaks at torus crossings (clean lines)."""
    p = path.astype(float)
    jumps = np.where((np.abs(np.diff(p, axis=0)) > size / 2).any(axis=1))[0]
    segs, last = [], 0
    for b in jumps:
        segs.append(p[last:b + 1])
        segs.append(np.full((1, 2), np.nan))
        last = b + 1
    segs.append(p[last:])
    return np.vstack(segs)


def agency_figure(rule, patch, gamma, sense, out: Path, size=140):
    """Trajectories from several food directions — each bends toward the food."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), facecolor="white")
    for ax, seed in zip(axes.ravel(), range(6)):
        frac, path, foods, w = forage_episode(rule, patch, gamma, sense, seed,
                                               size=size, record=True)
        # initial food (faint, where it started) + remaining/eaten halo
        f_disp = w.F / (w.F.max() + 1e-9)
        ax.imshow(np.dstack([np.zeros_like(f_disp), f_disp, np.zeros_like(f_disp)]))
        for fy, fx in foods:
            ax.plot(fx, fy, "*", color="lime", ms=16, mec="k", mew=0.5)
        seg = wrapped_segments(np.array(path), size)
        ax.plot(seg[:, 1], seg[:, 0], "-", color="cyan", lw=1.0)
        ax.plot(path[0][1], path[0][0], "wo", ms=7, mec="k")  # start
        ax.set_title(f"food@seed{seed}: ate {frac*100:.0f}%", fontsize=9)
        ax.set_xlim(0, size); ax.set_ylim(size, 0)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Round 3 — evolved chemotaxis (gain={gamma}, sense={sense}): "
                 f"the creature turns toward food in every direction", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=110)
    plt.close(fig)


def benchmark_figure(results, out: Path):
    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor="white")
    labels = list(results.keys())
    vals = [results[k] * 100 for k in labels]
    colors = ["#2a9d2a", "#b04030", "#808080"]
    ax.bar(labels, vals, color=colors[:len(labels)])
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=11)
    ax.set_ylabel("mean food eaten (% over random layouts)")
    ax.set_ylim(0, 105)
    ax.set_title("Agency: sensing is what reaches the food")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def make_gif(rule, patch, gamma, sense, out: Path, size=140, steps=360, seed=2):
    w = ForagingWorld((size, size), rule.to_params(), sense_sigma=sense,
                      gamma=gamma, eta=0.25)
    w.A = place_patch((size, size), patch)
    for _ in range(30):
        w.step()
    for f in _food_positions(size, 3, seed):
        w.add_food_blob(f, radius=9.0, amp=1.0)
    frames = []
    for i in range(steps):
        w.step()
        if i % 3 == 0:
            rgb = np.dstack([np.clip(w.A, 0, 1),
                             np.clip(w.F / 1.0, 0, 1) * 0.9,
                             np.zeros_like(w.A)])
            frames.append(rgb)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(frames[0])
    ttl = ax.set_title("creature (red) forages food (green)", fontsize=9)

    def upd(k):
        im.set_data(frames[k])
        return im, ttl

    FuncAnimation(fig, upd, frames=len(frames), blit=False).save(
        out, writer=PillowWriter(fps=20))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 3: evolve a foraging agent")
    ap.add_argument("--size", type=int, default=140)
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rule, patch = load_glider()
    train_seeds = list(range(8))
    test_seeds = list(range(100, 116))

    print("== evolving the forager (sensorimotor gain + sensing radius) ==")
    best = evolve_forager(rule, patch, train_seeds)
    print(f"best gain={best['gamma']} sense={best['sense']} "
          f"(train forage {best['score']*100:.0f}%)")

    print("== agency benchmark on UNSEEN layouts ==")
    evolved = mean_forage(rule, patch, best["gamma"], best["sense"], test_seeds)
    ablated = mean_forage(rule, patch, 0.0, best["sense"], test_seeds)
    randomd = mean_forage(rule, patch, best["gamma"], best["sense"], test_seeds,
                          mode="random")
    print(f"  evolved (sensing on): {evolved*100:.0f}%")
    print(f"  ablated (sensing off): {ablated*100:.0f}%")
    print(f"  random drift         : {randomd*100:.0f}%")

    results = {"evolved\n(sensing on)": evolved, "ablated\n(sensing off)": ablated,
               "random\ndrift": randomd}
    benchmark_figure(results, outdir / "round3_benchmark.png")
    agency_figure(rule, patch, best["gamma"], best["sense"],
                  outdir / "round3_agency.png", size=args.size)
    print(f"wrote {outdir/'round3_agency.png'} and round3_benchmark.png")
    if args.gif:
        make_gif(rule, patch, best["gamma"], best["sense"],
                 outdir / "round3_forage.gif", size=args.size)
        print(f"wrote {outdir/'round3_forage.gif'}")

    out = {"rule": rule.__dict__, "forager": best,
           "agency": {"evolved": evolved, "ablated": ablated, "random": randomd},
           "test_seeds": test_seeds}
    (outdir / "round3_forager.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {outdir/'round3_forager.json'}")


if __name__ == "__main__":
    main()
