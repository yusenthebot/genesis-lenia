# STATUS — genesis

GOAL: in a sandbox world, grow real intelligence; derive the world's evolution;
scale the same engine 1D -> 2D -> 3D. (open /loop direction, evolving mode)

MODE: evolving / frontier. A cleared bar is a floor, not the finish.

ROUND: 13 (complete, committed)

REVIEW (round 10): adversarially re-verified all 9 rounds -> ALL HOLD (incl fresh seeds). Negatives intact.

CURRENT STATE:
- Dimension-agnostic Lenia engine (genesis/world.py): N-D field, FFT radial-kernel
  convolution + growth; same code 1D/2D/3D. 51 tests green.
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
  (convergence curves + start/evolved gain histograms) + .gif. run7.py.
- ROUND 8 (2D, VERIFIED): PREDATOR-PREY. genesis/predprey.py — two species in one world:
  prey (forage food + flee predators) and predators (hunt+eat prey via a field interaction);
  both have energy/metabolism/reproduction + a heritable strategy gene. Results: (1) TOP-DOWN
  REGULATION — predators hold prey ~14 vs ~55 without predators (4x suppression), prey shows
  boom-bust fluctuations; (2) COEXISTENCE for the full run; (3) CO-EVOLUTION (honest surprise)
  — prey evolve LOWER evasion (7->3), not higher: fleeing costs more foraging than it saves, so
  "forage hard, out-breed predation" wins; predator pursuit stable (~10). HONEST NEGATIVE: no
  clean escalating arms race — across 5 regimes evasion was always selected down and the
  balance is knife-edge (strong predation -> prey extinction; moderate -> predator-cap-pinned
  coexistence). Evidence: outputs/round8_predprey.png + .gif.
- ROUND 9 (VERIFIED): WITHIN-LIFETIME LEARNING. genesis/learning.py — a creature with a PLASTIC
  value-learning brain on a REVERSAL-learning task (two food sources, only one nutritious, the
  rule flips every 320 steps). The learner tracks the rewarding source and RE-LEARNS after every
  reversal: accuracy 0.87 vs 0.49 for the SAME creature with plasticity off (lr=0, chance). This
  is adaptation within ONE life, not across generations — the missing ingredient. Pure numpy (no
  torch, avoided the dependency gate). Evidence: outputs/round9_learning.png (accuracy-over-life
  with reversal dips+recoveries + learner-vs-ablated bars) + .gif. run9.py.
- ROUND 11 (VERIFIED): EMBODIED LEARNING. genesis/embodied_learning.py — the plastic brain now lives
  INSIDE a Lenia creature: it forages two food types, its drift is a PLASTIC policy
  w_A*grad(F_A)+w_B*grad(F_B), only one type nutritious, rule flips mid-life, food evaporates if
  uneaten (so it must actively forage). The drift weights flip with the rule (argmax(w)==active 93%)
  and the embodied learner eats 0.89 nutritious vs 0.52 for the same body with plasticity off. Body
  (R2) + agency (R3) + learning (R9) united in ONE creature. Evidence: outputs/round11_embodied.png
  (weight-flip trajectory + nutritious-fraction curve) + .gif. run11.py.
- ROUND 12 (VERIFIED): MEASURING THE MIND. genesis/measure.py — quantify intelligence as the mutual
  information I(brain;world) in BITS between the creature's internal state (which food its brain
  prefers) and the world's hidden variable (which food is nutritious). Result: learner 0.69 bits vs
  ablated 0.00 bits. OPERATING ENVELOPE (MI vs reversal interval): 0.17 bits (fast world, flip 100)
  -> 0.88 bits (slow world, flip 1600) -> the rate of change the mind can track, approaching the
  1-bit ceiling. The mind MEASURED, not asserted. Evidence: outputs/round12_measure.png (MI bar +
  envelope) + .gif. run12.py.
- ROUND 13 (VERIFIED): LEARNING UNDER SELECTION. genesis/evo_learning.py — learners (plastic brains)
  and fixed-reflex creatures (non-plastic inherited strategy) COMPETE in one world for two food types
  with 'is_learner' heritable; the nutritious type reverses at a tunable rate. Result (evolution of
  learning / Baldwin): in a CHANGING world the learner fraction RISES 0.5 -> 1.00 (learning wins); in
  a STABLE world it FALLS 0.5 -> 0.39 (fixed reflexes win — learning's startup cost buys nothing). The
  change-rate sweep shows a sharp transition (~interval 1000-2000): learning pays ONLY in worlds that
  change within a lifetime. Knowing more (R12) makes a creature win — conditionally. Evidence:
  outputs/round13_selection.png (learner-fraction-over-time + when-does-learning-pay curve) + .gif
  (learners=blue take over fixed=red in a changing world). Slow (~6 min, ~9 sims).

NEXT ROUND SEED (round 14): ranked:
  (a) BALDWIN EFFECT: let the learning rule / brain priors themselves evolve -> does evolution tune
      HOW to learn? (b) Deeper controller: multi-cue / recurrent / short-term memory.
  (c) PREDICTIVE/PLANNING intelligence: a creature that predicts the next world-state, not just reacts.
  (d) MILESTONE REVIEW (13 rounds in). (e) Stable mobile 3D creature (hard/open).

HOW TO RUN (drivers verified in round-10 review):
  cd ~/evolab/genesis
  .venv/bin/python -m genesis.run13 --gif     # round-13 learning under selection + gif (~6 min)
  .venv/bin/python -m genesis.run12 --gif     # round-12 measuring the mind (bits) + gif
  .venv/bin/python -m genesis.run11 --gif     # round-11 embodied learning + gif
  .venv/bin/python -m genesis.run9 --gif      # round-9 within-lifetime learning + gif
  .venv/bin/python -m genesis.run8 --gif      # round-8 predator-prey + gif
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
