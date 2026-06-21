"""Iterated learning — compositionality from cultural transmission (round 34).

Round 31 made language compositional with a HAND-ADDED structural reward. The principled
mechanism (Kirby et al.) is cultural transmission through a LEARNABILITY BOTTLENECK: each
"generation" learns the language from only a SUBSET of (meaning, signal) examples, then must
PRODUCE the whole language — generalising to the meanings it never saw. Holistic languages
can't be reconstructed from few examples, so they degrade; compositional (systematic) ones
survive and accumulate. Over generations topographic similarity RISES under the bottleneck,
but stays flat when the whole language is transmitted each time. No hand-added structure term.

The learner is a tanh-MLP trained by hand-coded gradient descent (pure numpy). Its smooth
interpolation of held-out meanings is the inductive bias that compositional structure exploits.
"""

from __future__ import annotations

import numpy as np

A = 3          # attribute 1 values
B = 3          # attribute 2 values
M = 3          # signal dimensionality
ALL = [(a, b) for a in range(A) for b in range(B)]


def _features(meanings):
    X = np.zeros((len(meanings), A + B))
    for i, (a, b) in enumerate(meanings):
        X[i, a] = 1.0; X[i, A + b] = 1.0
    return X


def _train_mlp(X, Y, H=16, epochs=500, lr=0.08, seed=0):
    """A tiny 1-hidden-layer MLP regressor (meaning-features -> signal), batch GD, numpy backprop."""
    rng = np.random.default_rng(seed)
    din, dout, n = X.shape[1], Y.shape[1], X.shape[0]
    W1 = rng.normal(0, 0.4, (H, din)); b1 = np.zeros(H)
    W2 = rng.normal(0, 0.4, (dout, H)); b2 = np.zeros(dout)
    for _ in range(epochs):
        A1 = np.tanh(X @ W1.T + b1)              # (n,H)
        Yh = A1 @ W2.T + b2                       # (n,dout)
        dYh = 2.0 * (Yh - Y) / n
        dW2 = dYh.T @ A1; db2 = dYh.sum(0)
        dZ1 = (dYh @ W2) * (1 - A1 ** 2)
        dW1 = dZ1.T @ X; db1 = dZ1.sum(0)
        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2

    def predict(meanings):
        Xa = _features(meanings)
        return np.tanh(Xa @ W1.T + b1) @ W2.T + b2
    return predict


def topo_similarity(lang):
    """lang: (len(ALL), M) signals. Spearman corr of meaning-distance vs signal-distance."""
    md, sd = [], []
    for i in range(len(ALL)):
        for j in range(i + 1, len(ALL)):
            md.append((ALL[i][0] != ALL[j][0]) + (ALL[i][1] != ALL[j][1]))
            sd.append(np.linalg.norm(lang[i] - lang[j]))
    md = np.array(md, float); sd = np.array(sd)
    rm = md.argsort().argsort() - md.size / 2.0
    rs = sd.argsort().argsort() - sd.size / 2.0
    return float((rm @ rs) / (np.sqrt((rm @ rm) * (rs @ rs)) + 1e-9))


def iterate(bottleneck, gens=18, seed=0, expressive=True, return_hist=False):
    """Run iterated learning. bottleneck = #meanings each generation learns from (< len(ALL)
    is the real transmission bottleneck; == len(ALL) is full transmission, the control)."""
    rng = np.random.default_rng(seed)
    lang = rng.normal(0, 1.0, (len(ALL), M))     # generation-0: a random (holistic) language
    if expressive:
        lang = _expressive(lang)
    curve = [topo_similarity(lang)]
    hist = [lang.copy()]
    for g in range(gens):
        idx = rng.choice(len(ALL), size=bottleneck, replace=False)
        subset = [ALL[i] for i in idx]
        pred = _train_mlp(_features(subset), lang[idx], seed=seed * 100 + g)
        lang = pred(ALL)
        if expressive:
            lang = _expressive(lang)             # keep meanings distinguishable (anti-collapse)
        curve.append(topo_similarity(lang))
        hist.append(lang.copy())
    if return_hist:
        return lang, np.array(curve), np.array(hist)
    return lang, np.array(curve)


def _expressive(lang, target=2.0):
    """Light expressivity pressure: rescale so signals keep a minimum spread (prevents the
    trivial all-meanings-one-signal collapse that pure transmission would drift to)."""
    c = lang - lang.mean(0)
    spread = np.sqrt((c ** 2).sum(1)).mean() + 1e-9
    return c * (target / spread)
