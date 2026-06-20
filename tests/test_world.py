"""Engine invariants — the properties that must hold in every dimension."""

import numpy as np
import pytest

from genesis.world import World, LeniaParams, kernel_shell
from genesis.metrics import analyze_history, circular_centroid


@pytest.mark.parametrize("shape", [(64,), (24, 24), (12, 12, 12)])
def test_engine_runs_in_any_dimension(shape):
    """The identical engine must run in 1D, 2D and 3D."""
    w = World(shape, LeniaParams(R=5.0))
    assert w.ndim == len(shape)
    w.seed_blob()
    w.step()
    assert w.A.shape == shape
    assert np.isfinite(w.A).all()


def test_state_stays_bounded():
    w = World((128,), LeniaParams())
    w.seed_blob(asymmetry=3.0)
    w.run(200)
    assert w.A.min() >= 0.0
    assert w.A.max() <= 1.0
    assert np.isfinite(w.A).all()


def test_kernel_is_normalised_and_radial():
    w = World((48, 48), LeniaParams(R=8.0))
    assert w.K.sum() == pytest.approx(1.0, abs=1e-9)
    # kernel is zero outside radius R from the centre
    cx, cy = 24, 24
    assert w.K[cx, cy + 20] == 0.0  # 20 > R


def test_kernel_shell_single_vs_multiring():
    r = np.linspace(0, 1, 50, endpoint=False)
    single = kernel_shell(r, LeniaParams(kernel_peaks=(1.0,)))
    assert single.max() > 0
    multi = kernel_shell(r, LeniaParams(kernel_peaks=(1.0, 0.5)))
    assert multi.shape == r.shape
    # outside [0,1) the shell is zero
    assert kernel_shell(np.array([1.5]), LeniaParams())[0] == 0.0


def test_empty_world_is_a_fixed_point():
    """An empty world must stay empty: growth(0) must be non-positive there."""
    w = World((64,), LeniaParams())
    w.run(10)
    assert w.mass == 0.0


def test_circular_centroid_handles_wrap():
    """Mass split across the periodic edge should centre near the edge, not mid."""
    f = np.zeros(100)
    f[:3] = 1.0
    f[-3:] = 1.0
    c = circular_centroid(f)[0]
    # true centre is the wrap seam (~0/100), not 50
    assert min(c, 100 - c) < 5


def test_analyze_reports_dead_for_empty_history():
    hist = np.zeros((50, 64))
    m = analyze_history(hist)
    assert m["alive"] is False
    assert m["dead"] is True
