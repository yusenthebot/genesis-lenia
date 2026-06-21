# genesis — what this is

A capstone reflection after 25 rounds. For the per-round detail see the [README](../README.md);
for the working state see [STATUS.md](STATUS.md) and [progress.md](progress.md).

## The one-paragraph version

Starting from a single direction — *"grow real intelligence in a sandbox world; derive its
evolution; scale the same engine 1D → 2D → 3D"* — an autonomous loop built, round by round, a
world and a mind inside it on **one** substrate: a continuous cellular automaton (the Lenia
family). The engine is dimension-agnostic (`len(shape)` *is* the dimensionality), so nothing
above it is hand-placed — structure, locomotion, agency, evolution, and the mind all *emerge*
from local rules and selection. Twenty-five rounds later the sandbox contains a creature that
emerges, moves, forages, survives, competes, evolves, is hunted, learns within its life,
**remembers**, **predicts**, **plans**, is **integrated** into one organism whose every faculty
is load-bearing, and a substrate shown to keep **generating** new creatures and new strategies
rather than converging. Each result is a runnable driver, a figure, and a metric — and every
honest negative is kept, not hidden.

## The arc, as one line

emergence → locomotion → agency → survival → **3D** → ecology → evolution → predator–prey →
learning → embodied learning → measuring the mind (bits) → learning-under-selection → the
Baldwin effect → **memory** → **prediction** → embodied memory → **planning** → **unification**
→ **open-endedness (bodies)** → **open-ended minds (strategies)** → **the 3D creature**.

The spine of it is a mind's core loop, grown from scratch: **perceive → model → predict →
act-to-achieve**, then *integrated* (one creature that needs all of memory + prediction +
planning to survive — ablate any one and it starves), then shown to be *generative* (illuminate
a behaviour space and a zoo of distinct bodies *and* distinct foraging strategies appears, where
plain optimisation collapses to one).

## What it is — and is not

**Is:** a from-scratch, `numpy`-only, fully reproducible demonstration that the *ingredients* of
intelligence — agency, memory, learning, prediction, planning, and open-ended generation — can
each be **grown** from a single local rule plus selection, measured (in food eaten, in steps
survived, in **bits** of mutual information, in catch time, in niches filled), and **composed**
into one organism. It was built autonomously: each round wrote its own next step, verified it by
*actually running it and looking* (not just passing tests), and recorded an honest result.

**Is not:** one omni-creature that does everything at once (the deep-mind faculties are shown in
the focused settings that make them measurable, embodied where that is the point); large-scale or
competitive with engineered agents (grids are tens of cells, brains are a handful of `tanh`
units); and it does **not** yet contain a *mobile* 3D creature.

## The honest negatives (kept on purpose)

- **The mobile 3D creature (the one headline negative).** The engine scales to 3D and
  self-organises; round 25 finds a *stable compact* 3D creature — but not a *moving* one. The
  search shows exactly why: compact 3D bodies are stationary attractors, and the structures that
  move are diffuse — the 3D glider is the empty *intersection*.
- **Unification is hand-wired.** The faculties were each shown *evolvable*, but a single tiny
  evolution-strategies net could not learn all four jointly, so the integrating controller is
  engineered; ablation proves the faculties *compose and each is necessary*.
- **Open-endedness is bounded.** Illumination fills a *finite* behaviour map; this is not a claim
  of *unbounded* open-endedness.
- Smaller, documented ones: the predator–prey arms race runs *down* not up; the Baldwin learning
  rate does not track the world's change-rate; embodied memory is real but seed-sensitive
  (corrected from an early over-claim in a later review).

Five review rounds (10, 17, 20, 23, and this one) re-verified earlier results on **fresh, unseen
seeds**; one over-claim was found and corrected. The discipline is the point: a cleared bar is a
floor, and a result only counts if it survives someone trying to break it.

## Where it goes next (a decision, not a default)

The `numpy`-CA substrate is near its ceiling. Every remaining *big* leap needs a new
dependency — a **gate** the loop will not cross on its own:

- **Flow-Lenia** (mass-conserving; multiple species share a world) → *unbounded* open-endedness
  and a real shot at mobile 3D creatures.
- **torch / GPU** → larger 3D worlds and bigger, learned controllers.
- **Differentiable Lenia** → gradient-*find* the 3D glider the genetic search can't reach.

Until one of those is chosen, the project is at a natural, honest plateau: complete in kind,
modest in scale, and truthful about both.
