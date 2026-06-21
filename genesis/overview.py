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
    ("outputs/round4_survival.png",
     "Round 4 · 2D · SURVIVAL — with metabolism, foraging is life or death: the "
     "forager lives ~832 steps vs 158 ablated / 299 random (unseen schedules)"),
    ("outputs/round5_3d.png",
     "Round 5 · 3D · DIMENSIONAL GENERALITY — the SAME engine self-organises a seed "
     "into a robust, homeostatic 3D structure (mass_cv 0.0008); 1D -> 2D -> 3D, one codebase"),
    ("outputs/round6_ecology.png",
     "Round 6 · 2D · ECOLOGY — many creatures contest shared food: survival peaks at an "
     "INTERMEDIATE foraging skill (stabilizing selection) — under- & over-foragers die out"),
    ("outputs/round7_evolution.png",
     "Round 7 · 2D · EVOLUTION RUNNING — foraging skill is heritable; from random gains the "
     "population self-tunes to the optimum (~3.5), matching round 6's swept optimum"),
    ("outputs/round8_predprey.png",
     "Round 8 · 2D · PREDATOR-PREY — a second species hunts the foragers: predators regulate "
     "prey 4x below its food ceiling; prey evolve to forage-hard, not flee (honest co-evolution)"),
    ("outputs/round9_learning.png",
     "Round 9 · LEARNING — a plastic brain learns which source is rewarding and RE-LEARNS after "
     "the rule reverses (0.87 vs 0.49 ablated): adaptation within one life, not across generations"),
    ("outputs/round11_embodied.png",
     "Round 11 · EMBODIED LEARNING — the plastic brain lives INSIDE a Lenia creature: its drift "
     "weights flip with the rule (93% correct) and it eats 0.89 nutritious vs 0.52 ablated (body+mind united)"),
    ("outputs/round12_measure.png",
     "Round 12 · MEASURING THE MIND — I(brain;world)=0.69 bits for the learner vs 0.00 ablated; the "
     "operating envelope shows knowledge scaling with how slowly the world changes. Mind, measured."),
    ("outputs/round13_selection.png",
     "Round 13 · LEARNING UNDER SELECTION — learners vs fixed-reflex creatures compete: learning is "
     "selected (fraction -> 1) in a CHANGING world but lost in a STABLE one. Knowing wins, conditionally."),
    ("outputs/round14_baldwin.png",
     "Round 14 · THE BALDWIN EFFECT — the LEARNING RATE is heritable: from random rates evolution "
     "self-tunes the population to a consistent optimum (~0.5). (Honest negative: it doesn't track change-rate.)"),
    ("outputs/round15_memory.png",
     "Round 15 · A BRAIN WITH MEMORY — a recurrent controller holds a cue across a delay (1.00 bit of "
     "memory, 100% recall) where a memoryless reflex is stuck at chance. Memory unlocks a new class of mind."),
    ("outputs/round16_predict.png",
     "Round 16 · A BRAIN THAT PREDICTS — a recurrent model foresees an ambiguous world, anticipating "
     "every flip (1.00 bit of predictive info) where a reactive brain is at chance. A mind that sees ahead."),
    ("outputs/round18_embodied_memory.png",
     "Round 18 · EMBODIED MEMORY — a recurrent brain drives a Lenia forager in a FLASHING world: memory "
     "lets it coast to food through the dark, collecting ~1.9 vs ~1.2 for a memoryless reflex (wins 3/4 ES "
     "seeds). Memory pays — moderately, and only when the world hides information."),
    ("outputs/round19_planning.png",
     "Round 19 · PLANNING — acting on foresight: a planner intercepts a circling target (cuts across, 2x "
     "faster) where reaction tail-chases; an evolved recurrent controller LEARNS to anticipate. Mind loop complete."),
    ("outputs/round21_unified.png",
     "Round 21 · UNIFICATION — ONE creature tracks moving, flashing food to survive, integrating body + memory + "
     "prediction + planning; ablate any faculty (263 -> 188 -> 140 steps) and it starves sooner. Each one earns its place."),
    ("outputs/round22_openended.png",
     "Round 22 · OPEN-ENDEDNESS — MAP-Elites ILLUMINATES a behaviour space (move x size), filling ~84% of it with a "
     "ZOO of distinct creatures while fitness search collapses to one; diversity ACCUMULATES instead of converging."),
    ("outputs/round24_openmind.png",
     "Round 24 · OPEN-ENDED MINDS — same body, but the MIND varies: MAP-Elites illuminates a zoo of distinct foraging "
     "STRATEGIES (roam x spread) with visibly different trajectories, where fitness converges to one. Creativity of minds."),
    ("outputs/round25_creature3d.png",
     "Round 25 · THE 3D CREATURE — a serious push (multi-ring kernels + shaped search) finds a STABLE COMPACT 3D creature "
     "(an upgrade on R5's lattice), but the compact-OR-moving trade-off leaves the MOBILE 3D glider an honest open negative."),
]


def main(out="outputs/progress_overview.png"):
    panels = [(p, c) for p, c in PANELS if Path(p).exists()]
    n = len(panels)
    fig = plt.figure(figsize=(12, 4.6 * n), facecolor="white")
    fig.suptitle("genesis — intelligence grown, not coded   (1D → 2D → 3D engine)\n"
                 "emergence → locomotion → agency → survival → ecology → evolution → "
                 "learning → memory → prediction → planning",
                 fontsize=13, fontweight="bold", y=0.997)
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
