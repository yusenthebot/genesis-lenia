# STATUS — genesis

GOAL: in a sandbox world, grow real intelligence; derive the world's evolution;
scale the same engine 1D -> 2D -> 3D. (open /loop direction, evolving mode)

MODE: evolving / frontier. A cleared bar is a floor, not the finish.

ROUND: 2 (complete, committed)

CURRENT STATE:
- Dimension-agnostic Lenia engine (genesis/world.py): N-D field, FFT radial-kernel
  convolution + growth; same code 1D/2D/3D. 15 tests green.
- Emergence metrics (genesis/metrics.py): alive/localized/persistent/locomotion +
  scale-aware mass-concentration & gyration (separates a creature from soup),
  wrap-aware centroid, net-drift vs oscillation.
- Evolution (genesis/evolve.py): co-evolves RULE (LeniaParams) + an evolvable SEED
  MORPHOLOGY (smooth patch) under a locomotion fitness. (mu+lambda) GA.
- ROUND 1 (1D): spontaneous self-organisation into a persistent homeostatic lattice;
  directed motion transient (0.40w -> 0.00w).
- ROUND 2 (2D, VERIFIED by real run + eye): EVOLVED a genuine travelling creature
  (a glider) DE NOVO — net travel 3.78 widths, straightness 0.99, concentration 1.0,
  mass_cv 0.0021. Discovered, not hand-placed. Evidence: outputs/round2_2d_creature.png
  + .gif. Caught/fixed two Goodhart failures (soup gamed support; blobs can't reach
  the glider basin -> evolve the seed morphology).

NEXT ROUND SEED (round 3): turn locomotion into AGENCY. Add a resource/food field to
the world; select creatures that move TOWARD and consume it (sensing -> action ->
self-maintenance). This is where "intelligence" starts being real, not asserted.
Also: open-endedness — evolve a ZOO of distinct creatures + a diversity metric.

HOW TO RUN:
  cd ~/evolab/genesis
  .venv/bin/python -m genesis.run2d --gif     # evolve a 2D glider, render strip+gif
  .venv/bin/python -m genesis.run1d           # round-1 1D self-organisation
  .venv/bin/python -m pytest -q

VERIFICATION BAR: never trust tests/metrics alone — run the sim, LOOK at the field /
trajectory, confirm the metric matches the picture. (Round 2's soup proved why.)
