"""Compose a single progress montage from the per-round figures.

One glance shows the arc: emergence (1D) -> locomotion (2D) -> agency (2D).
Reads the already-rendered round figures in outputs/ and tiles them with captions.
Run after the round drivers: python -m genesis.overview
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402

PANELS = [
    ("outputs/round1_1d_selforg.png",
     "Round 1 · 1D · EMERGENCE — a single seed self-organises into a persistent "
     "homeostatic lattice"),
    ("outputs/round2_2d_creature.png",
     "Round 2 · 2D · LOCOMOTION — co-evolving rule + seed morphology discovers a "
     "travelling creature (a glider) de novo"),
    ("outputs/round3_agency.png",
     "Round 3 · 2D · AGENCY — an evolved forager senses food and steers to it "
     "(85% eaten vs 18% sensing-ablated, on unseen layouts)"),
]


def main(out="outputs/progress_overview.png"):
    panels = [(p, c) for p, c in PANELS if Path(p).exists()]
    n = len(panels)
    fig = plt.figure(figsize=(12, 4.6 * n), facecolor="white")
    fig.suptitle("genesis — intelligence grown, not coded:  "
                 "emergence  →  locomotion  →  agency   (1D → 2D → 3D)",
                 fontsize=15, fontweight="bold", y=0.995)
    for i, (path, caption) in enumerate(panels):
        ax = fig.add_subplot(n, 1, i + 1)
        ax.imshow(mpimg.imread(path))
        ax.set_title(caption, fontsize=11, loc="left")
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
