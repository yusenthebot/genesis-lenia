# STATUS — genesis

GOAL: in a sandbox world, grow real intelligence; derive the world's evolution;
scale the same engine 1D -> 2D -> 3D. (open /loop direction, evolving mode)

MODE: evolving / frontier. A cleared bar is a floor, not the finish.

ROUND: 1 (complete, committed)

CURRENT STATE:
- Dimension-agnostic Lenia engine (genesis/world.py): N-D field, FFT radial-kernel
  convolution + growth map; same code runs 1D/2D/3D. 9 tests green.
- Emergence metrics (genesis/metrics.py): alive/localized/persistent/locomotion,
  wrap-aware circular centroid, net-drift vs oscillation.
- 1D result (VERIFIED by real run): spontaneous self-organisation into a persistent
  homeostatic lattice (mass_cv~0, support~0.12). Directed motion is TRANSIENT
  (early drift ~0.40 widths -> late ~0.00). Honest negative on persistent 1D gliders.

NEXT ROUND SEED (round 2): go 2D. Reproduce a real travelling creature (Orbium-class
glider) in the SAME engine — unambiguous persistent locomotion. Then begin evolution
(perturb LeniaParams genome under a selection objective).

HOW TO RUN:
  cd ~/evolab/genesis
  .venv/bin/python -m genesis.run1d
  .venv/bin/python -m pytest -q

VERIFICATION BAR: never trust tests alone — run the sim, inspect the space-time /
field output, and confirm the metric matches what the picture shows.
