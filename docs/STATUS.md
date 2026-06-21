# STATUS — genesis

GOAL: in a sandbox world, grow real intelligence; derive the world's evolution;
scale the same engine 1D -> 2D -> 3D. (open /loop direction, evolving mode)

MODE: evolving / frontier. A cleared bar is a floor, not the finish.

ROUND: 4 (complete, committed)

CURRENT STATE:
- Dimension-agnostic Lenia engine (genesis/world.py): N-D field, FFT radial-kernel
  convolution + growth; same code 1D/2D/3D. 21 tests green.
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
- ROUND 4 (2D, VERIFIED): SURVIVAL. Metabolism on; food spawns over time; the creature
  DIES without eating. On UNSEEN food schedules the evolved forager lives ~832 steps
  (banks an energy surplus) vs 158 ablated vs 299 random (cap 900). Intelligence now =
  the difference between living and dying. Evidence: outputs/round4_survival.png + .gif.

NEXT ROUND SEED (round 5): pick the next frontier, ranked:
  (a) ECOLOGY: many creatures in one world + shared/finite food -> competition; then
      predator/prey. Needs multi-creature bookkeeping (per-creature energy + identity).
  (b) OPEN-ENDED EVOLUTION: a ZOO of distinct survivors + a behavioural-diversity metric;
      watch novelty/complexity grow over generations.
  (c) WITHIN-LIFETIME LEARNING: a small neural controller modulating the drift; adaptation
      not just evolution (torch + MPS).
  (d) 3D: same engine, one more axis (1D->2D->3D goal); 3D creature + volume render.

HOW TO RUN:
  cd ~/evolab/genesis
  .venv/bin/python -m genesis.run4 --gif      # round-4 survival (metabolism) + benchmark
  .venv/bin/python -m genesis.run3 --gif      # round-3 foraging agency + benchmark
  .venv/bin/python -m genesis.run2d --gif     # evolve a 2D glider, render strip+gif
  .venv/bin/python -m genesis.run1d           # round-1 1D self-organisation
  .venv/bin/python -m genesis.overview        # rebuild the progress montage
  .venv/bin/python -m pytest -q

VERIFICATION BAR: never trust tests/metrics alone — run the sim, LOOK at the field /
trajectory, confirm the metric matches the picture. (Round 2's soup proved why.)
