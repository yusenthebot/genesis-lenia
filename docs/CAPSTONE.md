# genesis — what this is

A capstone reflection after 35 rounds. For the per-round detail see the [README](../README.md);
for the working state see [STATUS.md](STATUS.md) and [progress.md](progress.md).

## The one-paragraph version

Starting from a single direction — *"grow real intelligence in a sandbox world; derive its
evolution; scale the same engine 1D → 2D → 3D"* — an autonomous loop built, round by round, a
world and a mind inside it on **one** substrate: a continuous cellular automaton (the Lenia
family). The engine is dimension-agnostic (`len(shape)` *is* the dimensionality), so nothing
above it is hand-placed — structure, locomotion, agency, evolution, and the mind all *emerge*
from local rules and selection. Thirty-five rounds later the sandbox contains a creature that
emerges, moves, forages, survives, competes, evolves, is hunted, learns within its life,
**remembers**, **predicts**, **plans**, is **integrated** into one organism whose every faculty
is load-bearing, a substrate shown to keep **generating** new bodies and strategies rather than
converging, and — once the world holds two agents — a whole **social** layer: a **shared language**
that *emerges*, becomes *compositional* (both under pressure and, principled, through cultural
transmission), *grounds* itself in foraging action, and a separate faculty of **theory of mind**
that reads another agent's hidden goal from its behaviour. Each result is a runnable driver, a
figure, and a metric — and every honest negative is kept, not hidden.

## The arc, as one line

emergence → locomotion → agency → survival → **3D** → ecology → evolution → predator–prey →
learning → embodied learning → measuring the mind (bits) → learning-under-selection → the
Baldwin effect → **memory** → **prediction** → embodied memory → **planning** → **unification**
→ **open-endedness (bodies)** → **open-ended minds (strategies)** → **the 3D creature** →
**Flow-Lenia** (mass-conserving substrate) → **emergent communication** → **compositional
communication** → **grounded communication** → **iterated learning** → **theory of mind**.

Three threads run through it: (1) a **mind's core loop**, grown from scratch — *perceive → model
→ predict → act-to-achieve* — then *integrated* (one creature that needs memory + prediction +
planning together to survive; ablate any one and it starves); (2) **open-endedness** — illuminate
a behaviour space and a zoo of distinct bodies *and* foraging strategies appears, where plain
optimisation collapses to one; (3) **social intelligence** — the deepest recent thread: two agents
evolve a shared code (2 bits, ablation-proven); it becomes *compositional* both under a structural
pressure (topo 0.25→0.79) and — the principled mechanism — through cultural transmission's
learnability bottleneck (Kirby); it *grounds* in action (a scout's signal lets a blind forager
forage, 0.58 vs 0.05 ablated); and a separate faculty, *theory of mind*, infers another agent's
hidden goal from its motion alone (belief sharpening as it watches, ablate→chance).

## What it is — and is not

**Is:** a from-scratch, `numpy`-only, fully reproducible demonstration that the *ingredients* of
intelligence — agency, memory, learning, prediction, planning, open-ended generation, and
**communication** — can each be **grown** from local rules + selection, measured (in food eaten,
steps survived, **bits** of mutual information, catch time, niches filled, topographic similarity),
and **composed**. It was built autonomously: each round wrote its own next step, verified it by
*actually running it and looking* (not just passing tests), and recorded an honest result.

**Is not:** one omni-creature that does everything at once (faculties are shown in the focused
settings that make them measurable, embodied where that is the point); large-scale or competitive
with engineered agents (grids are tens of cells, brains are a handful of `tanh` units); and it
does **not** contain a *mobile* 3D creature — that is the one standing, exhaustively-explained
negative.

## The honest negatives (kept on purpose)

- **The mobile 3D / mobile creature — the headline negative, now fully explained.** The engine
  scales to 3D and self-organises (round 5); round 25 finds a *stable compact* 3D creature but not
  a *moving* one. Rounds 27–29 rebuilt the substrate as **Flow-Lenia** (mass-conserving, which
  makes 3D creatures *robust*) and searched hard — single-channel, multi-channel, asymmetric and
  rotated flows, two advection schemes, proper evolutionary searches. All land on a *stationary*
  creature, and round 28 says **why**: the flow `F = ∇G` is a *gradient* flow (curl-free), so it
  relaxes to a stationary equilibrium. Translation needs a non-gradient flow; reaching it credibly
  needs the **gated** differentiable-Lenia apparatus. A thoroughly-characterised wall, not a mystery.
- **Unification is hand-wired.** The faculties were each shown *evolvable*, but one tiny ES net
  could not learn all four jointly, so the integrating controller is engineered; ablation proves
  the faculties *compose and each is necessary*.
- **Open-endedness is bounded; compositionality is partial.** Illumination fills a *finite*
  behaviour map (not unbounded). Compositional language emerges under pressure (topographic
  similarity 0.25 → 0.79) but zero-shot generalisation is only partial (the listener lags).
- Smaller, documented ones: the predator–prey arms race runs *down* not up; the Baldwin learning
  rate does not track the world's change-rate; embodied memory is real but seed-sensitive (an early
  over-claim corrected in a later review).

Seven review rounds (10, 17, 20, 23, 26, 32, 36) re-verified earlier results on **fresh, unseen
seeds** — round 32 the substrate + first communication rounds, round 36 the later social rounds
(grounded comm 0.71 vs 0.03 ablated, iterated learning bottleneck 0.28 vs full 0.10, theory of mind
0.78 vs 0.25 chance) — all hold; two over-claims (embodied memory; a ToM "beats naive" docstring)
were found and corrected. The discipline is the point: a cleared bar is a floor, and a result only
counts if it survives someone trying to break it.

## Where it goes next

The **core goal — real intelligence — still has a rich, ungated vein**: **social / multi-agent**
intelligence, now five rounds deep (emergent → compositional → grounded → iterated communication,
plus theory of mind). Natural ungated continuations in pure `numpy`: **multi-agent coordination**
(a task solvable only together; division of labour), comm-enabled coordination, harder theory of
mind (behaviour that *misleads* — detours/obstacles, where modelling beats a position oracle), and
cultural/iterated dynamics over the grounded language.

The one place that is genuinely **gated** (a dependency decision for the human) is the **mobile
creature**: it is a proven wall in numpy, and the credible path is **differentiable Lenia**
(gradient-descend through the dynamics to *find* a glider, the method the Flow-Lenia paper used) —
which needs torch / JAX. `torch / GPU` would also unlock larger 3D worlds and learned controllers.

So: keep pushing the ungated social/intelligence vein; the gated motion path waits for a deliberate
call. The present project stands complete in kind, broad and deep, and truthful about its scale.
