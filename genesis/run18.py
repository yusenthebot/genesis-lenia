"""Round 18 — embodied memory: a recurrent brain in a Lenia creature, where memory PAYS.

The Lenia forager's drift is produced by a small recurrent net. The world FLASHES: the
food gradient is visible only briefly, then dark while the food persists. We evolve the
net (ES) and compare how much food the embodied creature collects:

  recurrent, flashing world  -> holds the direction in memory, forages through the dark
  feedforward, flashing world -> stalls when the signal cuts out
  feedforward, STEADY world   -> forages fine (the control: it is not broken, it just
                                  cannot cope with an intermittent signal)

So memory is not a free upgrade — it pays exactly when the world hides information that
must be carried across time. The two tracks (embodied body, recurrent mind) are reunited.
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
from genesis.embodied_memory import evolve_es, fitness, episode  # noqa: E402

GENS = 30
FLASH_OFF = 26        # long dark windows -> memory genuinely required to reach food


def load_glider(path="outputs/round2_genome.json"):
    d = json.load(open(path))
    rule = Genome(**{k: d["rule"][k] for k in
                     ["mu_g", "sigma_g", "mu_k", "sigma_k", "R", "dt"]})
    return rule, np.array(d["patch"])


def figure(rule, patch, out: Path):
    # train three conditions; average the flashing pair over 2 ES seeds (high variance)
    test = tuple(range(20, 30))            # many unseen seeds -> stable estimate
    rec_runs = [evolve_es(rule, patch, recurrent=True, gens=GENS, flash_off=FLASH_OFF, seed=s)
                for s in (0, 1)]
    ff_runs = [evolve_es(rule, patch, recurrent=False, gens=GENS, flash_off=FLASH_OFF, seed=s)
               for s in (0, 1)]
    th_fs, _ = evolve_es(rule, patch, recurrent=False, gens=GENS, flash_off=0, seed=0)
    c_rf = np.mean([c for _, c in rec_runs], axis=0)
    c_ff = np.mean([c for _, c in ff_runs], axis=0)
    rf = float(np.mean([fitness(rule, patch, t, True, seeds=test, flash_off=FLASH_OFF)
                        for t, _ in rec_runs]))
    ff = float(np.mean([fitness(rule, patch, t, False, seeds=test, flash_off=FLASH_OFF)
                        for t, _ in ff_runs]))
    fs = fitness(rule, patch, th_fs, False, seeds=test, flash_off=0)
    th_rf = rec_runs[0][0]

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor="white")
    g = np.arange(1, GENS + 1)
    a0.plot(g, c_rf, color="#2a9d2a", lw=2.4, label="recurrent (memory), flashing world")
    a0.plot(g, c_ff, color="#b04030", lw=2.4, label="feedforward (no memory), flashing world")
    a0.set_xlabel("evolution generation"); a0.set_ylabel("food collected / episode")
    a0.set_title("Evolving a forager when the signal flashes")
    a0.legend(fontsize=9, loc="upper left")

    labels = ["recurrent\nflashing", "feedforward\nflashing", "feedforward\nSTEADY (control)"]
    vals = [rf, ff, fs]
    a1.bar(labels, vals, color=["#2a9d2a", "#b04030", "#888"])
    for i, v in enumerate(vals):
        a1.text(i, v + 0.05, f"{v:.1f}", ha="center", fontsize=11)
    a1.set_ylabel("food collected / episode (unseen seeds)")
    a1.set_title("Memory pays only when the world hides information")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return th_rf, dict(recurrent_flashing=rf, feedforward_flashing=ff,
                       feedforward_steady=fs)


def make_gif(rule, patch, theta, out: Path, seed=2):
    _, _, _, frames = episode(rule, patch, theta, True, T=300, seed=seed,
                              flash_off=FLASH_OFF, record=True)
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.set_xticks([]); ax.set_yticks([])
    A0, F0, *_ = frames[0]
    im = ax.imshow(np.dstack([A0, F0 * 0.9, np.zeros_like(A0)]))
    ttl = ax.set_title("", fontsize=10)

    def upd(k):
        A, F, coll, t, dark = frames[k]
        im.set_data(np.dstack([A, F * 0.9, np.zeros_like(A)]))
        sig = "DARK (no signal) — coasting on memory" if dark else "signal ON"
        ttl.set_text(f"t={t}  food collected={coll}   [{sig}]")
        return im, ttl

    FuncAnimation(fig, upd, frames=len(frames), blit=False).save(
        out, writer=PillowWriter(fps=18))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 18: embodied memory")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rule, patch = load_glider()
    print("== embodied memory: a recurrent brain in a Lenia creature, flashing food ==")
    theta_rec, res = figure(rule, patch, outdir / "round18_embodied_memory.png")
    print(f"  food collected: recurrent(flashing)={res['recurrent_flashing']:.1f}  "
          f"feedforward(flashing)={res['feedforward_flashing']:.1f}  "
          f"feedforward(STEADY ctrl)={res['feedforward_steady']:.1f}")
    print(f"wrote {outdir/'round18_embodied_memory.png'}")
    if args.gif:
        make_gif(rule, patch, theta_rec, outdir / "round18_embodied_memory.gif")
        print(f"wrote {outdir/'round18_embodied_memory.gif'}")

    (outdir / "round18_embodied_memory.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {outdir/'round18_embodied_memory.json'}")


if __name__ == "__main__":
    main()
