# STATUS — genesis

GOAL: in a sandbox world, grow real intelligence; derive the world's evolution;
scale the same engine 1D -> 2D -> 3D. (open /loop direction, evolving mode)

MODE: evolving / frontier. A cleared bar is a floor, not the finish.

ROUND: 25 (THE 3D CREATURE — compact found, mobile still negative; complete, committed)

REVIEW (round 23): re-verified R21-22 on FRESH unseen seeds -> BOTH HOLD (not cherry-picked). R21 ablation
ladder on seeds 30-49: full 288 > memory_only 192 > no_memory 157 (committed 263/188/140 on seeds 0-19 —
reproduces directionally). R22 on fresh seed 2: MAP-Elites fills 54/64 niches, fitness-GA diversity collapses
to 8 (reproduces exactly). 75 tests green; all images resolve; no orphan modules; honest negatives/caveats
(R18 seed-sensitivity, R21 hand-wired + bounce overshoot, R22 foam + bounded) all intact. NOTE: repo growing
(.git 62M, outputs 51M) from committed GIFs/PNGs — inherent to the visual README; acceptable, watch it.

REVIEW (round 20): re-verified R18-19. R19 reproduces EXACTLY (pursuit 54 vs planner 24; recurrent 0.45 vs
feedforward 0.05). R18 HOLDS ON AVERAGE but is the project's MOST SEED-SENSITIVE result: a quick under-
trained probe FLIPPED it (recurrent 0.2 < feedforward 1.0); careful re-check (4 ES seeds, committed settings)
gives recurrent mean 1.90 vs feedforward 1.21, recurrent wins 3/4 -> real moderate effect, NOT the clean 2x
the single figure showed. CORRECTED overclaims: montage caption ("collects nothing" -> "~1.2") + README
(added honest seed-sensitivity caveat) + whole-arc capstone intro with honest-scope box. 69 tests green; all
images resolve. Earlier reviews (R10: rounds 1-9; R17: rounds 11-16) still hold.

REVIEW (round 10): re-verified rounds 1-9 -> all hold. REVIEW (round 17): re-verified rounds 11-16 ->
ALL HOLD. R11 93%/0.89-vs-0.52; R12 0.69 bits + envelope; R13 changing->1.00 / stable->0.44; R14 lr
0.48->0.59; R15 1.00-vs-0.53 (1 bit); R16 1.00-vs-0.50 (1 bit). R15/R16 robust on FRESH unused seeds
(recurrent ~1.0, feedforward ~chance -> not cherry-picked). 62 tests green; no orphan modules; all README
images resolve; honest negatives intact. NOTE: repo growing (.git ~50M, outputs ~46M) from committed
GIFs/PNGs — inherent to the visual README; acceptable, watch it.

CURRENT STATE:
- Dimension-agnostic Lenia engine (genesis/world.py): N-D field, FFT radial-kernel
  convolution + growth; same code 1D/2D/3D. 80 tests green.
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
  the 1D->2D->3D engine arc. HONEST NEGATIVE (ADVANCED in R25): a compact MOBILE 3D creature is NOT
  delivered. (R25 found a stable COMPACT 3D creature — conc 1.00 — but it does not move; the mobile 3D
  glider remains the open negative.) Evidence: outputs/round5_3d.png + .gif; genesis/evolve3d.py.
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
  outputs/round13_selection.png + .gif. run13.py.
- ROUND 14 (VERIFIED, mixed): THE BALDWIN EFFECT. genesis/baldwin.py — the LEARNING RATE is a
  heritable gene; creatures are born naive and learn at their own inherited rate, mutated+selected.
  POSITIVE: from RANDOM rates the population self-tunes to ~0.57 (concentrated high) -> evolution
  discovers a good amount of plasticity (favouring fast learning in a changing world); the Baldwin
  effect operates on the rate. HONEST NEGATIVE: the evolved rate does NOT track the world's change-
  rate (tested flip 80..3000, +/- reward noise -> ~flat / no clean gradient). The clean Bayesian
  volatility->learning-rate law is washed out by the embodied foraging dynamics. Evidence:
  outputs/round14_baldwin.png + .gif. run14.py.
- ROUND 15 (VERIFIED): A BRAIN WITH MEMORY. genesis/memory_brain.py — a tiny RECURRENT controller
  (hidden state, evolved by an ES, pure numpy) on a cue-recall task: a cue appears, vanishes, and
  after a random delay the creature must act on it. Result: recurrent 1.00 accuracy + holds 1.00 BIT
  of memory across the delay; feedforward (memory ablated) 0.53 (chance) + 0.00 bits — it literally
  cannot, the cue is gone when it must act. Memory unlocks a task class impossible for a reflex, and
  the memory is both MEASURED (1 bit, via measure.py MI) and VISUALISED (the held hidden-state trace).
  Fast (~6s, no FFTs). Evidence: outputs/round15_memory.png (ES curves + memory trace) + .gif
  (watch the cue set the memory ... fire the correct action). run15.py.
- ROUND 16 (VERIFIED): A BRAIN THAT PREDICTS. genesis/predict_brain.py — a recurrent net (ES, numpy)
  must predict the next state of a periodic AMBIGUOUS world (pattern [0,0,1,1]: a 0 leads to 0 at one
  phase and 1 at another, so you must track phase + model the pattern). Result: recurrent 1.00 accuracy
  + 1.00 BIT of PREDICTIVE INFORMATION I(brain;next-state), anticipating EVERY flip; feedforward
  (reactive) 0.50 + 0.00 bits. The brain builds a model and FORESEES the world. run16.py.
- ROUND 18 (VERIFIED): EMBODIED MEMORY — the two tracks reunited + payoff proven. genesis/embodied_memory.py
  — a recurrent brain (ES, numpy) drives a LENIA forager's drift; the world FLASHES (food gradient visible
  briefly, then dark while the food persists). Result (food collected/episode, averaged over 2 ES seeds,
  unseen eval seeds): recurrent-flashing 1.9 vs feedforward-flashing 1.0; feedforward-STEADY control 3.4.
  So flashing CRUSHES the memoryless forager (3.4 -> 1.0, -70%) and memory roughly DOUBLES it (-> 1.9),
  recovering much of the loss -> memory PAYS exactly when the world hides information across time. The
  embodied (rounds 3-13) and deep-mind (rounds 15-16) tracks are now ONE creature. Evidence:
  outputs/round18_embodied_memory.png + .gif. run18.py.
- ROUND 19 (VERIFIED): PLANNING — acting on foresight (the mind's capstone). genesis/planning.py — a
  target ORBITS on a circle; an agent must intercept. REACTIVE pursuit heads at where the target IS and
  tail-chases around the circle; a model-based PLANNER heads where it WILL be and cuts across. Result:
  planner catches in 24 steps vs reactive pursuit 54 (~2.2x faster), with the classic pursuit-curve-vs-
  interception trajectories. And an EVOLVED RECURRENT controller (sees only relative position, must infer
  the motion) LEARNS to anticipate: catch rate 0.45 vs 0.05 for a feedforward (reaction-only) controller.
  Completes the mind's core loop perceive->model->predict->ACT-TO-ACHIEVE. (Honest: ES doesn't reach the
  hand-coded planner; interception is hard to learn from scratch, but recurrent >> feedforward 9x.) run19.py.
- ROUND 21 (VERIFIED): UNIFICATION — one creature, all four faculties, proven load-bearing by ablation.
  genesis/unified.py — a Lenia creature must stay on food that MOVES (constant-velocity, bouncing) and FLASHES
  (visible briefly then dark) under METABOLISM, so tracking it is survival. ONE integrated controller:
  perceive -> REMEMBER (hold the estimate across dark) -> PREDICT (dead-reckon by remembered velocity) ->
  PLAN/ACT (drive the Lenia drift to the lead/intercept). Ablation ladder (steps survived, max 320, 20 seeds):
  full 263 > memory_only 188 (no prediction -> lags) > no_memory 140 (stalls in the dark). Each faculty earns
  its place. Body (3-13)+memory (15/18)+prediction (16)+planning (19) are now ONE organism. HONEST: controller
  is hand-integrated (faculties were shown EVOLVABLE earlier; this proves they COMPOSE + each is necessary);
  the lead occasionally overshoots at a bounce so full isn't best on every seed (best on avg + 10/16 seeds).
  Fast (~4s, deterministic). Evidence: outputs/round21_unified.png (ablation bars + energy-over-life) + .gif. run21.py.
- ROUND 22 (VERIFIED): OPEN-ENDEDNESS — does the world keep GENERATING, or converge? genesis/openended.py —
  MAP-Elites ILLUMINATES a 2-D behaviour space (drift-speed x body-mass), keeping the most VIABLE creature in
  each niche; vs a fitness-GA baseline. Result: MAP-Elites fills 54/64 niches (84%) across the FULL move x size
  range, and diversity ACCUMULATES (coverage climbs 24->54 over evaluations); the fitness GA's population
  diversity COLLAPSES/stays ~8 as selection converges to one body type. So the substrate is GENERATIVE across
  this behaviour space, not single-attractor. HONEST: the "zoo" spans a clean small glider (mass ~140) through
  large multi-blob "foam" textures (high mass) — the behaviour space is filled with viable PATTERNS, not all
  discrete single organisms; and the map is bounded (no claim of UNBOUNDED open-endedness). Evidence:
  outputs/round22_openended.png + .gif. run22.py.
- ROUND 24 (VERIFIED): OPEN-ENDED MINDS — a zoo of foraging STRATEGIES, not body shapes. genesis/openmind.py —
  same body (the evolved glider), but the MIND is a 4-parameter foraging policy (sensing range, gain, exploration,
  momentum); the creature forages scattered food. MAP-Elites illuminates the foraging-BEHAVIOUR space (how much it
  ROAMS x how WIDELY its path SPREADS), keeping the best-eating forager per niche; vs a fitness GA. Result:
  MAP-Elites fills 28 niches (of the reachable ~30; low-roam can't be high-spread) with VISIBLY DISTINCT
  trajectories — compact efficient foragers, sprawling rovers, columnar pacers — while the fitness GA's population
  diversity collapses to ~5. So open-endedness holds for MINDS too, not just morphologies. HONEST: the reachable
  behaviour region is a triangle (~30/64 is near max, not 64); and a compact path can eat as much as a long rover
  (small world) -> the diversity is in STYLE, not all in success. Evidence: outputs/round24_openmind.png (strategy
  map + coverage curve + TRAJECTORY gallery) + .gif (4 minds foraging). run24.py.
- ROUND 25 (VERIFIED, mixed): THE 3D CREATURE — a serious push at the R5 negative. genesis/creature3d.py upgrades
  R5's search with (1) MULTI-RING kernels (kernel_peaks), (2) a SHAPED viability gradient (partial credit for healthy
  mass + compactness + persistence, so the GA can climb where the round-2 fitness hard-gates on `alive` and gives no
  gradient), (3) a motion reward. POSITIVE: a STABLE COMPACT 3D creature is found — a single localised body,
  concentration 1.00, mass ~3162 (an upgrade on R5's diffuse self-organisation). HONEST NEGATIVE (sharpened): a
  MOBILE 3D glider is still NOT found. The compactness x motion scatter shows exactly why — compact bodies are
  STATIONARY attractors, the structures that MOVE (travel ~0.5) are DIFFUSE (conc ~0.3); the glider is the unmet
  INTERSECTION (compact AND moving), and that corner is empty. Evidence: outputs/round25_creature3d.png (3 views of
  the compact creature + the compact-OR-moving trade-off scatter) + .gif (rotating 3D creature). run25.py.

NEXT ROUND SEED (round 26 — most frontiers explored; lean consolidation/honesty): ranked:
  (a) CAPSTONE REVIEW + "what is this" essay (leading): 25 rounds in, the arc is comprehensive and the last big
      negative (3D glider) is now sharply characterised. Re-verify a couple recent rounds, write the definitive
      whole-project telling, ensure the README/montage are the artifact. A natural place to consolidate.
  (b) One more 3D-glider attempt from the DIFFUSE-MOVING side (relax compactness, seek a propagating localised wave)
      — lower odds, but the scatter hints motion is reachable; could try seeding from 2D-glider-like asymmetry in 3D.
  (c) GATED: a richer substrate (Flow-Lenia / torch-GPU / differentiable Lenia) would unlock harder 3D + bigger
      worlds, but needs a NEW DEPENDENCY -> ask the user before adding.

HOW TO RUN (drivers verified in round-10 review):
  cd ~/evolab/genesis
  .venv/bin/python -m genesis.run25 --gif     # round-25 the 3D creature (compact found, mobile negative) + gif (~7s)
  .venv/bin/python -m genesis.run24 --gif     # round-24 open-ended minds (foraging-strategy zoo) + gif (~6 min)
  .venv/bin/python -m genesis.run22 --gif     # round-22 open-endedness (MAP-Elites zoo) + gif (~5 min)
  .venv/bin/python -m genesis.run21 --gif     # round-21 unification (ablation of faculties) + gif (~4s)
  .venv/bin/python -m genesis.run19 --gif     # round-19 planning / interception + gif (~90s)
  .venv/bin/python -m genesis.run18 --gif     # round-18 embodied memory (recurrent brain in Lenia) + gif (~4 min)
  .venv/bin/python -m genesis.run16 --gif     # round-16 a brain that predicts + gif (~5s)
  .venv/bin/python -m genesis.run15 --gif     # round-15 a brain with memory + gif (~6s)
  .venv/bin/python -m genesis.run14 --gif     # round-14 Baldwin effect + gif (~4 min)
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
