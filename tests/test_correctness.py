"""Correctness of the discretisation and the solve.

The load-bearing test here is agreement with an independent SciPy direct solve.
Everything else checks a property the exact solution must satisfy, so a
regression shows up even where no reference is available.
"""

from __future__ import annotations

import numpy as np
import pytest

import darcy
from conftest import two_phase

scipy = pytest.importorskip("scipy", reason="the reference solver needs SciPy")
from darcy.reference import assemble_matrix, solve_reference  # noqa: E402


@pytest.mark.parametrize("n", [16, 31, 32, 48, 64])
def test_matches_sparse_direct_solve(n: int, preconditioner: str) -> None:
    """Iterative solve agrees with a sparse LU factorisation of the same operator.

    Includes odd n: the 2x2 aggregation coarsening has to handle a trailing
    aggregate of width one, and an off-by-one there would show up here.
    """
    a = two_phase(n, seed=n)
    result = darcy.solve(a, tol=1e-12, max_iter=200_000, preconditioner=preconditioner)
    assert result.converged, f"did not converge: {result}"

    expected = solve_reference(a)
    rel_err = np.linalg.norm(result.u - expected) / np.linalg.norm(expected)
    assert rel_err < 1e-8, f"rel err {rel_err:.2e} after {result.iterations} iterations"


def test_preconditioners_agree_on_the_same_solution() -> None:
    """All three preconditioners must reach the same fixed point."""
    a = two_phase(40, seed=7)
    solutions = [
        darcy.solve(a, tol=1e-13, max_iter=200_000, preconditioner=p).u
        for p in ("multigrid", "jacobi", "none")
    ]
    for other in solutions[1:]:
        assert np.allclose(solutions[0], other, rtol=1e-7, atol=1e-14)


def test_variable_forcing_matches_reference() -> None:
    """A non-constant right-hand side is handled, not silently ignored."""
    n = 32
    a = two_phase(n, seed=3)
    xs = np.linspace(0, 1, n)
    f = np.outer(np.sin(np.pi * xs), 1.0 + xs**2)

    result = darcy.solve(a, f, tol=1e-12)
    expected = solve_reference(a, f)
    rel_err = np.linalg.norm(result.u - expected) / np.linalg.norm(expected)
    assert rel_err < 1e-8


def test_operator_matches_assembled_matrix() -> None:
    """The matrix-free stencil equals the assembled sparse operator.

    This isolates the discretisation from the solver: if this passes and a solve
    disagrees, the bug is in the Krylov/multigrid code, not the stencil.
    """
    rng = np.random.default_rng(0)
    n = 17
    a = two_phase(n, seed=11)
    matrix = assemble_matrix(a)
    for _ in range(5):
        x = rng.standard_normal((n, n))
        assert np.allclose(darcy.apply_operator(a, x), (matrix @ x.ravel()).reshape(n, n), atol=1e-12)


def test_operator_is_symmetric() -> None:
    """<Ax, y> == <x, Ay> to round-off. CG's convergence theory depends on it."""
    rng = np.random.default_rng(1)
    a = two_phase(24, seed=5)
    for _ in range(10):
        x = rng.standard_normal((24, 24))
        y = rng.standard_normal((24, 24))
        left = float(np.sum(darcy.apply_operator(a, x) * y))
        right = float(np.sum(x * darcy.apply_operator(a, y)))
        assert abs(left - right) <= 1e-10 * max(abs(left), 1.0)


def test_operator_is_positive_definite() -> None:
    """<Ax, x> > 0 for x != 0, which is what makes CG applicable at all."""
    rng = np.random.default_rng(2)
    a = two_phase(24, seed=6)
    for _ in range(20):
        x = rng.standard_normal((24, 24))
        assert float(np.sum(x * darcy.apply_operator(a, x))) > 0.0


def test_solution_is_positive_and_bounded() -> None:
    """With f > 0 and a > 0 the maximum principle gives u > 0 in the interior."""
    result = darcy.solve(two_phase(48, seed=9), f=1.0, tol=1e-12)
    assert (result.u > 0).all()
    # Comparison against the constant-coefficient solution with the smallest
    # permeability, which bounds the solution from above.
    slowest = darcy.solve(np.full((48, 48), 3.0), f=1.0, tol=1e-12)
    assert result.u.max() <= slowest.u.max() * (1 + 1e-9)


def test_symmetric_problem_gives_symmetric_solution() -> None:
    """A symmetric permeability field must produce a symmetric solution."""
    result = darcy.solve(np.full((40, 40), 5.0), tol=1e-13)
    assert np.allclose(result.u, result.u.T, atol=1e-11)
    assert np.allclose(result.u, result.u[::-1, :], atol=1e-11)


def test_scales_inversely_with_permeability() -> None:
    """u depends on a only through 1/a, so doubling a halves u."""
    u1 = darcy.solve(np.full((40, 40), 1.0), tol=1e-13).u
    u2 = darcy.solve(np.full((40, 40), 2.0), tol=1e-13).u
    assert np.allclose(u1, 2.0 * u2, rtol=1e-8)


def test_linear_in_the_forcing_term() -> None:
    """The problem is linear in f: solve(2f) == 2 solve(f)."""
    a = two_phase(32, seed=4)
    u1 = darcy.solve(a, f=1.0, tol=1e-13).u
    u3 = darcy.solve(a, f=3.0, tol=1e-13).u
    assert np.allclose(3.0 * u1, u3, rtol=1e-8)


def test_zero_forcing_gives_zero_solution() -> None:
    """f == 0 short-circuits to the exact answer without iterating."""
    result = darcy.solve(two_phase(16, seed=1), f=0.0)
    assert result.converged
    assert result.iterations == 0
    assert np.all(result.u == 0.0)


def test_residual_is_actually_achieved() -> None:
    """The reported residual matches one recomputed from scratch.

    CG updates the residual recursively, which can drift from the true residual
    over many iterations; this checks the reported number is not a fiction.
    """
    n = 64
    a = two_phase(n, seed=2)
    tol = 1e-11
    result = darcy.solve(a, tol=tol)
    h2 = (1.0 / (n + 1)) ** 2
    b = np.full((n, n), h2)
    true_residual = np.linalg.norm(b - darcy.apply_operator(a, result.u)) / np.linalg.norm(b)
    assert true_residual < 10 * tol
    assert result.residual == pytest.approx(true_residual, rel=0.05, abs=1e-13)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_tiny_grids(n: int) -> None:
    """Grids too small to coarsen must still solve exactly."""
    result = darcy.solve(np.full((n, n), 2.0), tol=1e-13)
    assert result.converged
    expected = solve_reference(np.full((n, n), 2.0))
    assert np.allclose(result.u, expected, atol=1e-12)


def test_extreme_permeability_contrast() -> None:
    """A 10^6 jump ratio still converges: the regime that breaks naive multigrid."""
    n = 64
    a = darcy.two_phase_permeability(darcy.gaussian_random_field(n, seed=13), hi=1e6, lo=1.0)
    result = darcy.solve(a, tol=1e-10, max_iter=2000)
    assert result.converged, f"failed on a 1e6 contrast: {result}"
    expected = solve_reference(a)
    rel_err = np.linalg.norm(result.u - expected) / np.linalg.norm(expected)
    assert rel_err < 1e-6, f"rel err {rel_err:.2e}"
