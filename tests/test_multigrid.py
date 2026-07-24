"""Multigrid-specific properties.

These are the tests that would catch a regression in the preconditioner while
leaving the answer correct -- a solver that still converges but has quietly
stopped being O(N).
"""

from __future__ import annotations

import numpy as np
import pytest

import darcy
from conftest import two_phase


def test_hierarchy_depth_is_logarithmic() -> None:
    """Coarsening reaches a 1x1 grid, so depth is log2(n) + 1."""
    for n, expected in [(1, 1), (2, 2), (4, 3), (8, 4), (16, 5), (64, 7), (128, 8)]:
        result = darcy.solve(np.full((n, n), 1.0), tol=1e-10)
        assert result.levels == expected, f"n={n}: {result.levels} levels, expected {expected}"


def test_odd_grids_still_coarsen_fully() -> None:
    """An odd n must not stall the hierarchy at a large coarse grid."""
    for n in (17, 33, 47, 63):
        result = darcy.solve(np.full((n, n), 1.0), tol=1e-10)
        assert result.levels >= 5, f"n={n} only produced {result.levels} levels"


@pytest.mark.parametrize("cycle_index", [1, 2])
def test_iteration_count_beats_jacobi_by_an_order_of_magnitude(cycle_index: int) -> None:
    a = two_phase(128, seed=0)
    mg = darcy.solve(a, tol=1e-10, cycle_index=cycle_index)
    jacobi = darcy.solve(a, tol=1e-10, preconditioner="jacobi", max_iter=100_000)
    assert mg.converged and jacobi.converged
    assert mg.iterations * 10 < jacobi.iterations, (
        f"multigrid took {mg.iterations} iterations vs Jacobi's {jacobi.iterations}"
    )


def test_w_cycle_iteration_count_is_near_grid_independent() -> None:
    """The core claim of the solver: refining the grid must not cost iterations.

    Jacobi-CG roughly doubles its iteration count for every doubling of n. The
    W-cycle should stay within a small constant factor across a 16x refinement;
    anything worse means the coarse-grid correction has broken.
    """
    counts = {}
    for n in (32, 64, 128, 256, 512):
        result = darcy.solve(two_phase(n, seed=0), tol=1e-10, cycle_index=2)
        assert result.converged
        counts[n] = result.iterations

    growth = counts[512] / counts[32]
    assert growth < 2.0, f"iteration counts grew {growth:.2f}x across 16x refinement: {counts}"
    assert counts[512] < 50, f"512x512 needed {counts[512]} iterations: {counts}"


def test_more_smoothing_never_increases_iterations() -> None:
    a = two_phase(128, seed=1)
    light = darcy.solve(a, tol=1e-10, pre_sweeps=1, post_sweeps=1)
    heavy = darcy.solve(a, tol=1e-10, pre_sweeps=3, post_sweeps=3)
    assert heavy.iterations <= light.iterations


def test_w_cycle_needs_fewer_iterations_than_v_cycle() -> None:
    a = two_phase(256, seed=2)
    v = darcy.solve(a, tol=1e-10, cycle_index=1)
    w = darcy.solve(a, tol=1e-10, cycle_index=2)
    assert w.iterations < v.iterations


def test_residual_history_converges_steadily() -> None:
    """The residual must fall steadily, not oscillate.

    CG minimises the error in the A-norm, not the 2-norm, so the *first* step
    can and does raise the 2-norm residual as the preconditioner rescales it
    (measured around 5x here). After that, a working symmetric cycle gives a
    steady contraction of roughly 0.3 per iteration; a broken or unsymmetric
    cycle oscillates instead, which is what this pins down.
    """
    result = darcy.solve(two_phase(128, seed=3), tol=1e-10, record_history=True)
    history = result.history
    assert history[0] == 1.0
    assert len(history) == result.iterations + 1
    assert history[-1] < 1e-10

    ratios = history[1:] / history[:-1]
    assert ratios[1:].max() < 1.0, f"residual grew mid-solve by {ratios[1:].max():.2f}x"
    assert np.median(ratios) < 0.6, f"median contraction only {np.median(ratios):.2f}"


@pytest.mark.parametrize("cycle_index", [1, 2])
@pytest.mark.parametrize("sweeps", [1, 2, 3])
def test_preconditioner_is_symmetric(cycle_index: int, sweeps: int) -> None:
    """<M r, s> == <r, M s>. CG's convergence theory requires it.

    This is the test that justifies the mirrored smoothing order in the cycle.
    Swap the post-smoothing colours to match the pre-smoothing ones and this
    fails immediately, while the solver still appears to work.
    """
    rng = np.random.default_rng(4)
    n = 32
    a = two_phase(n, seed=8)

    for _ in range(5):
        r = rng.standard_normal((n, n))
        s = rng.standard_normal((n, n))
        mr = darcy.apply_preconditioner(
            a, r, pre_sweeps=sweeps, post_sweeps=sweeps, cycle_index=cycle_index
        )
        ms = darcy.apply_preconditioner(
            a, s, pre_sweeps=sweeps, post_sweeps=sweeps, cycle_index=cycle_index
        )
        left = float(np.sum(mr * s))
        right = float(np.sum(ms * r))
        scale = max(abs(left), abs(right), 1e-300)
        assert abs(left - right) / scale < 1e-10, f"asymmetry {abs(left - right) / scale:.2e}"


@pytest.mark.parametrize("cycle_index", [1, 2])
def test_preconditioner_is_positive_definite(cycle_index: int) -> None:
    """<M r, r> > 0 for r != 0, the other half of the CG requirement."""
    rng = np.random.default_rng(5)
    n = 32
    a = two_phase(n, seed=9)
    for _ in range(20):
        r = rng.standard_normal((n, n))
        mr = darcy.apply_preconditioner(a, r, cycle_index=cycle_index)
        assert float(np.sum(mr * r)) > 0.0


def test_preconditioner_approximates_the_inverse() -> None:
    """M should be a good approximation to A^-1: ||I - M A|| must be well below 1.

    This is the quantitative statement of "the preconditioner works", measured
    as the worst error-reduction factor over random vectors.
    """
    rng = np.random.default_rng(6)
    n = 64
    a = two_phase(n, seed=10)
    worst = 0.0
    for _ in range(10):
        x = rng.standard_normal((n, n))
        # e = x - M A x is the error left after one cycle applied to A x.
        residual = x - darcy.apply_preconditioner(a, darcy.apply_operator(a, x))
        worst = max(worst, float(np.linalg.norm(residual) / np.linalg.norm(x)))
    assert worst < 0.5, f"one cycle only reduced the error by {worst:.3f}"


@pytest.mark.slow
def test_scales_to_a_million_cells() -> None:
    """A 1024x1024 solve (1.05M unknowns) converges in bounded iterations."""
    result = darcy.solve(two_phase(1024, seed=0), tol=1e-10)
    assert result.converged
    assert result.iterations < 60, f"took {result.iterations} iterations"
    assert result.levels == 11
