"""genesis — a sandbox world where intelligence is grown, not coded.

The substrate is a continuous cellular automaton (Lenia family) written so the
*same* engine runs in 1D, 2D and 3D: the world is an N-dimensional field, the
update is a radial-kernel convolution plus a growth map. Dimensionality is just
``len(shape)``. Everything above the engine — emergent creatures, evolution,
sensorimotor agency — is meant to emerge from local rules, never hand-placed.
"""

from genesis.world import World, LeniaParams
from genesis.metrics import analyze_history

__all__ = ["World", "LeniaParams", "analyze_history"]
