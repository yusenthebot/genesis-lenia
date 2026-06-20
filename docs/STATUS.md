# STATUS — genesis

GOAL: in a sandbox world, grow real intelligence; derive the world's evolution;
scale the same engine 1D -> 2D -> 3D. (open /loop direction, evolving mode)

MODE: evolving / frontier. A cleared bar is a floor, not the finish.

ROUND: 3 (complete, committed)

CURRENT STATE:
- Dimension-agnostic Lenia engine (genesis/world.py): N-D field, FFT radial-kernel
  convolution + growth; same code 1D/2D/3D. 19 tests green.
- Emergence metrics (genesis/metrics.py): alive/localized/persistent/locomotion +
  scale-aware mass-concentration & gyration (creature vs soup), wrap-aware centroid.
- Evolution (genesis/evolve.py): co-evolves RULE + evolvable SEED MORPHOLOGY (patch).
- Foraging (genesis/foraging.py): ForagingWorld = Lenia body + food field + sensorimotor
  reflex (sense food gradient over body -> rigid np.roll drift up-gradient, mass-exact)
  + eat. run3.py evolves the forager and runs the agency benchmark.
- ROUND 1 (1D): self-organisation into a persistent homeostatic lattice; motion transient.
- ROUND 2 (2D): EVOLVED a genuine glider DE NOVO (travel 3.78w, straightness 0.99).
- ROUND 3 (2D, VERIFIED by run + eye): AGENCY. Evolved a forager that senses food and
  steers to it. On UNSEEN random layouts: evolved (sensing on) eats 85% vs ablated
  (sensing off) 18% vs random-drift 18%. The random-drift control = same motion, no
  food-sensing, scores like ablated -> the win is specifically food-directed sensing,
  not movement. Evidence: outputs/round3_agency.png, round3_benchmark.png, round3_forage.gif.

NEXT ROUND SEED (round 4): deepen agency + open-endedness. Options, ranked:
  (a) METABOLIC SURVIVAL: turn on decay+feed so the creature DIES without food; fitness
      = lifetime. Foraging becomes survival, not reward. (foraging.py already has decay/feed.)
  (b) ECOLOGY: multiple creatures + food competition; or predator/prey. Flow-Lenia
      mass-conservation so species share a world.
  (c) LEARNING within lifetime (neural controller modulating drift; torch+MPS).
  (d) 3D world (same engine) once agency is rich.

HOW TO RUN:
  cd ~/evolab/genesis
  .venv/bin/python -m genesis.run3 --gif      # round-3 foraging agency + benchmark
  .venv/bin/python -m genesis.run2d --gif     # evolve a 2D glider, render strip+gif
  .venv/bin/python -m genesis.run1d           # round-1 1D self-organisation
  .venv/bin/python -m pytest -q

VERIFICATION BAR: never trust tests/metrics alone — run the sim, LOOK at the field /
trajectory, confirm the metric matches the picture. (Round 2's soup proved why.)
