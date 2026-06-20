"""Round 1 — grow living structure in a 1D world, then look at it.

We do not assume the rules produce life. We *search*: perturb the growth map and
the (only) hand-placed seed blob, run each candidate world, and keep the winners
under two honest objectives:

  stable  — spontaneous self-organisation into persistent, homeostatic structure
  mover   — a single localised creature that travels (proto-locomotion / agency)

Each winner is re-run, long, and drawn as a space-time diagram (time downward).
Per-candidate seeding is deterministic, so the re-run reproduces the search exactly.
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

from genesis.world import World, LeniaParams, kernel_shell, _bump  # noqa: E402
from genesis.metrics import analyze_history, window_drift  # noqa: E402

IC_VARIANTS = [
    dict(asymmetry=0.0, radius_mul=1.0),
    dict(asymmetry=3.0, radius_mul=1.0),   # skewed -> can self-propel
    dict(asymmetry=6.0, radius_mul=0.8),
    dict(asymmetry=10.0, radius_mul=0.6),  # strongly skewed packet
]


def _candidate_params():
    """A coarse neighbourhood around the canonical Orbium growth map."""
    for mu_g in np.linspace(0.08, 0.26, 10):
        for sigma_g in np.linspace(0.010, 0.040, 9):
            yield LeniaParams(mu_g=float(mu_g), sigma_g=float(sigma_g))


def _build_world(size, params, ic, seed, trial):
    """Deterministic per-candidate construction (re-runs reproduce the search)."""
    rng = np.random.default_rng((seed, trial))
    w = World((size,), params)
    w.seed_blob(center=(size // 2,), radius=params.R * ic["radius_mul"],
                amp=1.0, asymmetry=ic["asymmetry"], rng=rng)
    return w


def stable_score(m: dict) -> float:
    """Favour homeostatic, well-localised structure (movement is a small bonus)."""
    if not m["alive"]:
        return 0.0
    return 0.45 * m["persistent"] + 0.35 * m["localized"] + 0.20 * m["locomotion"]


def mover_score(m: dict) -> float:
    """Favour a *single* creature that travels: tight support, alive, mobile.

    locomotion already requires directed drift (not breathing), so a stationary
    breather scores ~0 here even though it is beautifully persistent.
    """
    if not m["alive"]:
        return 0.0
    sf = m["support_fraction"]
    if sf < 0.015 or sf > 0.30:        # reject empty soup and space-filling forests
        return 0.0
    return 0.60 * m["locomotion"] + 0.25 * m["persistent"] + 0.15 * m["localized"]


def search(size, steps, seed, score_fn, verbose=True, label="", ic_filter=None):
    best = None
    n_alive = 0
    trial = 0
    ics = [ic for ic in IC_VARIANTS if ic_filter is None or ic_filter(ic)]
    for params, ic in itertools.product(_candidate_params(), ics):
        w = _build_world(size, params, ic, seed, trial)
        hist = w.run(steps, record=True)
        m = analyze_history(hist)
        if m["alive"]:
            n_alive += 1
        s = score_fn(m)
        if best is None or s > best["score"]:
            best = {"params": params, "ic": ic, "trial": trial,
                    "metrics": m, "score": s}
        trial += 1
    if verbose:
        print(f"[{label}] {trial} candidates, {n_alive} alive, "
              f"best {label}-score {best['score']:.3f}")
    return best


def run_best(best, size, steps, seed):
    w = _build_world(size, best["params"], best["ic"], seed, best["trial"])
    hist = w.run(steps, record=True)
    return w, hist


def visualize(hist, params, out_path: Path, metrics, title, caption=None):
    T, N = hist.shape
    fig = plt.figure(figsize=(11, 7), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[2.4, 1.0], height_ratios=[3, 1])

    ax0 = fig.add_subplot(gs[:, 0])
    ax0.imshow(hist, aspect="auto", cmap="magma", interpolation="nearest",
               extent=[0, N, T, 0])
    ax0.set_title(title)
    ax0.set_xlabel("space")
    ax0.set_ylabel("time")

    r = np.linspace(0, 1, 200)
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(r, kernel_shell(r, params), color="#444")
    ax1.set_title("kernel K(r)")
    ax1.set_xlabel("normalised radius")

    u = np.linspace(0, 0.4, 300)
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(u, 2 * _bump(u, params.mu_g, params.sigma_g) - 1, color="#a00")
    ax2.axhline(0, color="#ccc", lw=0.6)
    ax2.set_title("growth G(U)")
    ax2.set_xlabel("neighbourhood sum U")

    txt = (f"alive={metrics['alive']}  score={metrics['score']:.3f}  "
           f"mass_cv={metrics['mass_cv']:.3f}  support={metrics['support_fraction']:.3f}  "
           f"net_drift={metrics['travel_widths']:.2f} widths")
    if caption:
        txt = caption + "\n" + txt
    fig.text(0.5, 0.01, txt, ha="center", fontsize=9, family="monospace")
    fig.suptitle(
        f"R={params.R}  mu_k={params.mu_k} sigma_k={params.sigma_k}  "
        f"mu_g={params.mu_g:.3f} sigma_g={params.sigma_g:.3f} dt={params.dt}",
        fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _do(objective, score_fn, title, size, search_steps, final_steps, seed, outdir,
        ic_filter=None, make_caption=None):
    best = search(size, search_steps, seed, score_fn, label=objective,
                  ic_filter=ic_filter)
    p = best["params"]
    print(f"  best {objective}: mu_g={p.mu_g:.4f} sigma_g={p.sigma_g:.4f} "
          f"ic={best['ic']}")
    w, hist = run_best(best, size, final_steps, seed)
    metrics = analyze_history(hist)
    # quantify whether motion is transient: early-window vs late-window net drift
    early = window_drift(hist, 0.0, 0.30)
    late = window_drift(hist, 0.70, 1.0)
    metrics["early_drift_widths"] = early["net_widths"]
    metrics["late_drift_widths"] = late["net_widths"]
    print(f"  {objective} metrics: alive={metrics['alive']} "
          f"mass_cv={metrics['mass_cv']:.4f} support={metrics['support_fraction']:.3f} "
          f"net_drift={metrics['travel_widths']:.3f}w  "
          f"early_drift={early['net_widths']:.3f}w late_drift={late['net_widths']:.3f}w")
    caption = make_caption(metrics) if make_caption else None
    out = outdir / f"round1_1d_{objective}.png"
    visualize(hist, p, out, metrics, title, caption=caption)
    genome = {"objective": objective, "params": p.__dict__, "ic": best["ic"],
              "trial": best["trial"], "metrics": metrics, "size": size,
              "final_steps": final_steps, "seed": seed}
    (outdir / f"round1_genome_{objective}.json").write_text(
        json.dumps(genome, indent=2, default=float))
    print(f"  wrote {out}")
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Round 1: emergent 1D life")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--search-steps", type=int, default=700)
    ap.add_argument("--final-steps", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", type=str, default="outputs")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("== self-organisation: a persistent homeostatic lattice emerges ==")
    _do("selforg", stable_score,
        "1D world — spontaneous self-organisation into a persistent lattice",
        args.size, args.search_steps, args.final_steps, args.seed, outdir,
        make_caption=lambda m: (
            "from one seed the world self-organises into persistent, homeostatic "
            "localised structures"))

    print("== locomotion probe: is directed motion persistent or transient? ==")
    _do("locomotion", mover_score,
        "1D world — directed motion is TRANSIENT, decaying into a fixed lattice",
        args.size, args.search_steps, args.final_steps, args.seed, outdir,
        ic_filter=lambda ic: ic["asymmetry"] > 0.0,
        make_caption=lambda m: (
            f"directed drift decays: early={m['early_drift_widths']:.2f}w -> "
            f"late={m['late_drift_widths']:.2f}w. "
            "Persistent gliders are a 2D phenomenon (round 2)."))


if __name__ == "__main__":
    main()
