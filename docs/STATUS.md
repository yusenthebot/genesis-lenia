# STATUS — genesis

GOAL: in a sandbox world, grow real intelligence; derive the world's evolution;
scale the same engine 1D -> 2D -> 3D. (open /loop direction, evolving mode)

MODE: evolving / frontier. A cleared bar is a floor, not the finish.

ROUND: 7 (complete, committed)

CURRENT STATE:
- Dimension-agnostic Lenia engine (genesis/world.py): N-D field, FFT radial-kernel
  convolution + growth; same code 1D/2D/3D. 29 tests green.
- Emergence metrics (genesis/metrics.py): alive/localized/persistent/locomotion +
  scale-aware mass-concentration & gyration (creature vs soup), wrap-aware centroid.
- Evolution (genesis/evolve.py): co-evolves RULE + evolvable SEED MORPHOLOGY (patch).
- Foraging+metabolism (genesis/foraging.py): ForagingWorld = Lenia body + food field +
  sensorimotor reflex (sense food gradient -> rigid mass-exact np.roll drift) + eating +
  a scalar ENERGY reserve (metabolism drains it, eating refills it, 0 -> body dissipates).
- ROUND 1 (1D): self-organisation into a persistent homeostatic lattice; motion transient.
- ROUND 2 (2D): EVOLVED a genuine glider DE NOVO (travel 3.78w, straightness 0.99).
- ROUND 3 (2D): AGENCY. Evolved forager eats 85% vs 18% ablated vs 18% random on UNSEEN
  layouts -> the win is food-directed sensing, not motion.
- ROUND 4 (2D): SURVIVAL. Metabolism on; food spawns; the creature DIES without eating.
  On UNSEEN schedules the evolved forager lives ~832 vs 158 ablated vs 299 random (cap 900).
- ROUND 5 (3D, VERIFIED): DIMENSIONAL GENERALITY. The IDENTICAL engine runs a 3D world and
  self-organises a seed into a ROBUST, persistent 3D structure (tail mass_cv=0.0008, final
  mass ~22150 reproducible across 3 seeds) — the 3D analogue of round 1's lattice. Completes
  the 1D->2D->3D engine arc. HONEST NEGATIVE: a compact MOBILE 3D creature (round-2-glider
  analogue) is NOT delivered — across 5+ search strategies (single/multi-ring kernels, growth
  sweeps, evolved 3D morphology) the 3D dynamics are knife-edge (die/foam/proliferate, and
  flip on grid/seed); stable localised 3D creatures need the heavy specialised search 3D-Lenia
  research uses. Evidence: outputs/round5_3d.png + .gif; genesis/evolve3d.py (3D morphology GA).
- ROUND 6 (2D, VERIFIED): ECOLOGY. Many creatures (each a Lenia body in its own channel)
  compete for ONE shared, finite, replenishing food field; each has energy/metabolism +
  the foraging reflex (genesis/ecology.py). Result: STABILIZING SELECTION — survival peaks
  at INTERMEDIATE foraging gain (~4), with under-foragers (gain 0) starving first and
  over-foragers (gain 16) overshooting; robust across food-scarcity regimes (spawn 38/50/65,
  optimum=4 in all). Evidence: outputs/round6_ecology.png + .gif. run6.py.
- ROUND 7 (2D, VERIFIED): EVOLUTION RUNNING. genesis/evo_ecology.py — foraging gain is a
  HERITABLE gene; well-fed creatures REPRODUCE (offspring inherit gain + mutation), starving
  ones die. Under scarce food (food-limited population), from RANDOM gains the population
  self-tunes: mean gain 9.2 -> 3.5 across 3 runs, MATCHING round 6's independently-swept
  optimum (~4). Selection is happening, not measured. Evidence: outputs/round7_evolution.png
  (convergence curves + start/evolved gain histograms) + .gif (population colour = gain,
  shifting to the optimum). run7.py.

NEXT ROUND SEED (round 8): deepen co-evolution / intelligence. Ranked:
  (a) PREDATOR/PREY: a second species that eats creatures (not food) -> co-evolutionary arms
      race (prey evolve evasion, predators evolve pursuit). Two heritable strategies coupled.
  (b) EVOLVE MORPHOLOGY too in the ecology (not just gain) -> open-ended body+brain evolution.
  (c) WITHIN-LIFETIME LEARNING: a neural controller modulating drift (torch+MPS).
  (d) Quantify INTELLIGENCE (predictive/info-theoretic measures), or revisit stable 3D creature.

HOW TO RUN:
  cd ~/evolab/genesis
  .venv/bin/python -m genesis.run7 --gif      # round-7 evolution inside the ecology + gif
  .venv/bin/python -m genesis.run6 --gif      # round-6 ecology / competition + gif
  .venv/bin/python -m genesis.run5_3d --gif   # round-5 3D self-organisation + rotating gif
  .venv/bin/python -m genesis.run4 --gif      # round-4 survival (metabolism) + benchmark
  .venv/bin/python -m genesis.run3 --gif      # round-3 foraging agency + benchmark
  .venv/bin/python -m genesis.run2d --gif     # evolve a 2D glider, render strip+gif
  .venv/bin/python -m genesis.run1d           # round-1 1D self-organisation
  .venv/bin/python -m genesis.overview        # rebuild the progress montage
  .venv/bin/python -m pytest -q

VERIFICATION BAR: never trust tests/metrics alone — run the sim, LOOK at the field /
trajectory, confirm the metric matches the picture. (Round 2's soup proved why.)
