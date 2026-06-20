"""Round 2 — evolve a travelling creature in a 2D world, then watch it move.

Pipeline: evolve genomes toward locomotion -> re-run the winner on a roomier field
-> render a space-time strip, an unwrapped trajectory, and an animated GIF. The
creature is discovered by selection, not placed by hand.
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

from genesis.world import World  # noqa: E402
from genesis.evolve import evolve, Individual, place_patch  # noqa: E402
from genesis.metrics import analyze_history, circular_centroid  # noqa: E402


def render_run(ind: Individual, size: int, steps: int):
    """Re-run the winning individual on a (possibly larger) field, full history."""
    w = World((size, size), ind.rule.to_params())
    w.A = place_patch((size, size), ind.patch)
    hist = w.run(steps, record=True)
    return hist


def unwrapped_path(centroids: np.ndarray, shape) -> np.ndarray:
    """Accumulate wrap-corrected steps -> true trajectory across periodic edges."""
    shape = np.asarray(shape, dtype=np.float64)
    out = [centroids[0].astype(np.float64).copy()]
    for i in range(1, len(centroids)):
        d = centroids[i] - centroids[i - 1]
        d = (d + shape / 2.0) % shape - shape / 2.0
        out.append(out[-1] + d)
    return np.array(out)


def path_stats(path: np.ndarray, shape) -> dict:
    """Honest locomotion stats from the unwrapped trajectory (wrap-immune)."""
    steps = np.sqrt((np.diff(path, axis=0) ** 2).sum(axis=1))
    path_len = float(steps.sum())
    net = float(np.sqrt(((path[-1] - path[0]) ** 2).sum()))
    return {"net_widths": net / max(shape),
            "path_widths": path_len / max(shape),
            "straightness": net / (path_len + 1e-9)}


def snapshot_strip(hist, centroids, path, metrics, genome, out: Path, stats):
    T = hist.shape[0]
    shape = hist.shape[1:]
    fracs = [0.0, 0.2, 0.4, 0.6, 0.8, 0.99]
    idx = [min(T - 1, int(f * T)) for f in fracs]

    fig = plt.figure(figsize=(13, 6.0), facecolor="white")
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.85])
    vmax = max(1e-6, float(hist.max()))
    for j, i in enumerate(idx):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(hist[i], cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        c = centroids[i]
        ax.plot(c[1], c[0], "+", color="cyan", ms=10, mew=1.5)
        ax.set_title(f"t={i}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    # unwrapped trajectory (shows straight-line travel)
    axp = fig.add_subplot(gs[1, 0:3])
    axp.plot(path[:, 1], path[:, 0], color="#2060d0", lw=1.5)
    axp.plot(path[0, 1], path[0, 0], "go", ms=6, label="start")
    axp.plot(path[-1, 1], path[-1, 0], "rs", ms=6, label="end")
    axp.set_title("centroid trajectory (unwrapped)")
    axp.set_aspect("equal", adjustable="datalim")
    axp.legend(fontsize=8, loc="best")
    axp.invert_yaxis()

    # mass over time (homeostasis)
    axm = fig.add_subplot(gs[1, 3:6])
    mass = hist.reshape(T, -1).sum(axis=1)
    axm.plot(mass, color="#a00")
    axm.set_title("mass(t) — homeostasis")
    axm.set_xlabel("time")
    axm.set_ylim(0, mass.max() * 1.1)

    fig.suptitle(
        f"2D evolved creature — net travel {stats['net_widths']:.2f} widths, "
        f"straightness {stats['straightness']:.2f}, concentration "
        f"{metrics['concentration']:.2f}, mass_cv {metrics['mass_cv']:.4f}\n"
        f"rule: mu_g={genome.mu_g:.3f} sigma_g={genome.sigma_g:.4f} "
        f"mu_k={genome.mu_k:.3f} sigma_k={genome.sigma_k:.3f} R={genome.R:.0f}",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=110)
    plt.close(fig)


def make_gif(hist, out: Path, max_frames=120):
    T = hist.shape[0]
    stride = max(1, T // max_frames)
    frames = hist[::stride]
    vmax = max(1e-6, float(hist.max()))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(frames[0], cmap="magma", vmin=0, vmax=vmax,
                   interpolation="nearest")
    ttl = ax.set_title("t=0", fontsize=9)

    def upd(k):
        im.set_data(frames[k])
        ttl.set_text(f"t={k * stride}")
        return im, ttl

    anim = FuncAnimation(fig, upd, frames=len(frames), blit=False)
    anim.save(out, writer=PillowWriter(fps=20))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 2: evolve a 2D travelling creature")
    ap.add_argument("--search-size", type=int, default=84)
    ap.add_argument("--search-steps", type=int, default=320)
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--gens", type=int, default=18)
    ap.add_argument("--render-size", type=int, default=140)
    ap.add_argument("--render-steps", type=int, default=900)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true", help="also write an animated GIF")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("== evolving toward locomotion (rule + seed morphology) ==")
    res = evolve(size=args.search_size, steps=args.search_steps, seed=args.seed,
                 pop=args.pop, gens=args.gens)
    ind = res["individual"]
    rule = ind.rule
    print(f"best fitness {res['fitness']:.3f}  "
          f"(search travel {res['metrics']['travel_widths']:.2f}w)")

    print("== rendering winner on a roomier field ==")
    hist = render_run(ind, args.render_size, args.render_steps)
    metrics = analyze_history(hist, window=2.2 * rule.R)
    shape = hist.shape[1:]
    centroids = np.stack([circular_centroid(hist[i]) for i in range(len(hist))])
    path = unwrapped_path(centroids, shape)
    stats = path_stats(path, shape)
    print(f"render metrics: alive={metrics['alive']} mass_cv={metrics['mass_cv']:.4f} "
          f"support={metrics['support_fraction']:.3f} concentration={metrics['concentration']:.2f} "
          f"net_travel={stats['net_widths']:.2f}w straightness(unwrapped)={stats['straightness']:.2f}")

    strip = outdir / "round2_2d_creature.png"
    snapshot_strip(hist, centroids, path, metrics, rule, strip, stats)
    print(f"wrote {strip}")
    if args.gif:
        gif = outdir / "round2_2d_creature.gif"
        make_gif(hist, gif)
        print(f"wrote {gif}")

    out = {"rule": rule.__dict__, "patch": ind.patch.tolist(),
           "search_fitness": res["fitness"], "render_metrics": metrics,
           "render_path_stats": stats, "trail": res["trail"],
           "render_size": args.render_size, "render_steps": args.render_steps,
           "seed": args.seed}
    (outdir / "round2_genome.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {outdir / 'round2_genome.json'}")


if __name__ == "__main__":
    main()
