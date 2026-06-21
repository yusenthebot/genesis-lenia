"""Round 33 — grounded communication: a signal that drives foraging.

A SCOUT sees where food is and signals; a BLIND FORAGER sees only the signal and must navigate
to the food. Evolved jointly. With the channel the blind forager reaches the food; ablate it
(random signal) and it is lost. The signal carries ACTIONABLE SPATIAL information a body uses to
forage — fusing the embodied track (navigation) with the social track (signalling).
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

import genesis.communicate_grounded as G  # noqa: E402


def _sighted_rate(seed=777, n=60):
    """Upper bound: a forager that SEES the food heads straight at it."""
    rng = np.random.default_rng(seed)
    ok = 0
    for _ in range(n):
        th = rng.uniform(0, 2 * np.pi)
        food = G.D * np.array([np.cos(th), np.sin(th)])
        pos = np.zeros(2); mind = G.D
        head = np.array([np.cos(th), np.sin(th)])
        for _ in range(G.T):
            pos = pos + G.V * head; mind = min(mind, np.linalg.norm(pos - food))
        ok += mind < G.CATCH
    return ok / n


def figure(theta, curve, out: Path):
    cr = G.catch_rate(theta, ablate=False, seed=777)
    cra = G.catch_rate(theta, ablate=True, seed=777)
    sighted = _sighted_rate()

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(13.5, 4.4), facecolor="white")
    a0.bar(["sighted\n(sees food)", "comm pair\n(blind+scout)", "ablated\n(random signal)"],
           [sighted, cr, cra], color=["#888", "#2a9d2a", "#b04030"])
    for i, v in enumerate([sighted, cr, cra]):
        a0.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=11)
    a0.set_ylim(0, 1.05); a0.set_ylabel("catch rate (reaches food)")
    a0.set_title("the signal carries actionable info\n(blind forager forages only WITH comm)")

    # trajectories: with comm (reach food) vs ablated (lost)
    rng = np.random.default_rng(3)
    for k in range(7):
        th = rng.uniform(0, 2 * np.pi)
        _, _, tj, food = G.episode(theta, th, ablate=False, rng=rng, return_traj=True)
        a1.plot(tj[:, 0], tj[:, 1], color="#2a9d2a", lw=1.3, alpha=0.8)
        a1.plot(*food, "*", color="#d0a020", ms=13, mec="k")
    a1.plot(0, 0, "ko", ms=6); a1.set_aspect("equal"); a1.set_xticks([]); a1.set_yticks([])
    a1.set_title("WITH comm: forager steers to the food")

    rng = np.random.default_rng(3)
    for k in range(7):
        th = rng.uniform(0, 2 * np.pi)
        _, _, tj, food = G.episode(theta, th, ablate=True, rng=rng, return_traj=True)
        a2.plot(tj[:, 0], tj[:, 1], color="#b04030", lw=1.3, alpha=0.8)
        a2.plot(*food, "*", color="#d0a020", ms=13, mec="k")
    a2.plot(0, 0, "ko", ms=6); a2.set_aspect("equal"); a2.set_xticks([]); a2.set_yticks([])
    a2.set_title("ABLATED: lost (signal carries no info)")

    fig.suptitle("Round 33 — grounded communication: a scout's signal guides a blind forager to food",
                 fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=118)
    plt.close(fig)
    return dict(sighted=sighted, comm=cr, ablated=cra)


def make_gif(theta, out: Path):
    rng = np.random.default_rng(7)
    eps = []
    for _ in range(6):
        th = rng.uniform(0, 2 * np.pi)
        _, _, tj_c, food = G.episode(theta, th, ablate=False, rng=rng, return_traj=True)
        _, _, tj_a, _ = G.episode(theta, th, ablate=True, rng=np.random.default_rng(1), return_traj=True)
        eps.append((tj_c, tj_a, food))
    fig, (axc, axa) = plt.subplots(1, 2, figsize=(8.4, 4.4))
    for ax, t in zip((axc, axa), ("WITH comm", "ablated")):
        ax.set_xlim(-G.D - 6, G.D + 6); ax.set_ylim(-G.D - 6, G.D + 6)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=10)
    food_c = [ax.plot([], [], "*", color="#d0a020", ms=14, mec="k")[0] for ax in (axc, axa)]
    (lc,) = axc.plot([], [], color="#2a9d2a", lw=2)
    (la,) = axa.plot([], [], color="#b04030", lw=2)
    n = len(eps); per = G.T + 1

    def upd(k):
        e = k // per; s = k % per
        tj_c, tj_a, food = eps[e % n]
        food_c[0].set_data([food[0]], [food[1]]); food_c[1].set_data([food[0]], [food[1]])
        lc.set_data(tj_c[:s + 1, 0], tj_c[:s + 1, 1])
        la.set_data(tj_a[:s + 1, 0], tj_a[:s + 1, 1])
        return [lc, la] + food_c

    FuncAnimation(fig, upd, frames=n * per, blit=False).save(
        out, writer=PillowWriter(fps=10))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 33: grounded communication")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)

    print("== grounded communication: a scout guides a blind forager to food ==")
    theta, curve = G.evolve_es(gens=140, seed=0)
    res = figure(theta, curve, outdir / "round33_grounded.png")
    print(f"  catch rate: sighted {res['sighted']:.2f} | comm pair {res['comm']:.2f} | "
          f"ablated {res['ablated']:.2f}")
    print(f"wrote {outdir/'round33_grounded.png'}")
    if args.gif:
        make_gif(theta, outdir / "round33_grounded.gif")
        print(f"wrote {outdir/'round33_grounded.gif'}")
    (outdir / "round33_grounded.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
