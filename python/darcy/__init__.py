"""Matrix-free multigrid solver for 2-D Darcy flow.

Solves

    -div( a(x) grad u(x) ) = f(x)    on (0, 1)^2,    u = 0 on the boundary

with a cell-centred finite-volume discretisation using harmonic averaging of
the permeability at cell faces -- the correct treatment for the discontinuous,
two-phase media this solver targets.

The linear system is symmetric positive definite and is solved with Conjugate
Gradient preconditioned by a geometric-algebraic multigrid V-cycle. The
operator is never assembled: the 5-point stencil is applied on the fly, so
storage is O(N) and every kernel is a unit-stride OpenMP loop.

Quick start
-----------
>>> import numpy as np, darcy
>>> a = darcy.two_phase_permeability(darcy.gaussian_random_field(128, seed=0))
>>> result = darcy.solve(a)
>>> result.converged, result.iterations < 20
(True, True)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from . import _core
from .fields import gaussian_random_field, two_phase_permeability

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "BatchResult",
    "SolveResult",
    "__version__",
    "apply_operator",
    "apply_preconditioner",
    "build_info",
    "gaussian_random_field",
    "has_openmp",
    "num_threads",
    "set_num_threads",
    "solve",
    "solve_batch",
    "two_phase_permeability",
]

__version__: str = _core.__version__

Preconditioner = Literal["multigrid", "jacobi", "none"]

# Solving to tighter than this is pointless: the discretisation error of a
# second-order scheme dwarfs it, and CG stagnates in round-off noise.
_DEFAULT_TOL = 1e-10


@dataclass(frozen=True)
class SolveResult:
    """Outcome of a single solve.

    Attributes
    ----------
    u:
        Solution field with shape ``(n, n)``.
    iterations:
        Conjugate Gradient iterations performed.
    residual:
        Final relative residual ``||b - A u|| / ||b||``.
    converged:
        Whether ``tol`` was reached within ``max_iter``.
    levels:
        Number of multigrid levels in the hierarchy (1 when unpreconditioned).
    history:
        Relative residual after each iteration; empty unless
        ``record_history=True`` was passed.
    """

    u: NDArray[np.float64]
    iterations: int
    residual: float
    converged: bool
    levels: int
    history: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))

    def __repr__(self) -> str:
        return (
            f"SolveResult(shape={self.u.shape}, iterations={self.iterations}, "
            f"residual={self.residual:.3e}, converged={self.converged}, levels={self.levels})"
        )


@dataclass(frozen=True)
class BatchResult:
    """Outcome of a batched solve over independent permeability samples."""

    u: NDArray[np.float64]
    iterations: NDArray[np.int64]
    residuals: NDArray[np.float64]
    converged: NDArray[np.bool_]

    @property
    def all_converged(self) -> bool:
        return bool(self.converged.all())

    def __repr__(self) -> str:
        return (
            f"BatchResult(batch={self.u.shape[0]}, shape={self.u.shape[1:]}, "
            f"mean_iterations={self.iterations.mean():.1f}, "
            f"all_converged={self.all_converged})"
        )


def solve(
    a: ArrayLike,
    f: float | ArrayLike = 1.0,
    *,
    tol: float = _DEFAULT_TOL,
    max_iter: int = 10_000,
    preconditioner: Preconditioner = "multigrid",
    pre_sweeps: int = 1,
    post_sweeps: int = 1,
    cycle_index: int = 2,
    record_history: bool = False,
) -> SolveResult:
    """Solve ``-div(a grad u) = f`` with homogeneous Dirichlet conditions.

    Parameters
    ----------
    a:
        Permeability field, shape ``(n, n)``. Must be finite and strictly
        positive. Non-float64, Fortran-ordered, and non-contiguous inputs are
        converted automatically.
    f:
        Forcing term: a scalar, or an array with the same shape as ``a``.
    tol:
        Convergence threshold on the relative residual.
    max_iter:
        Iteration cap. A result with ``converged=False`` is returned rather
        than an exception if it is hit.
    preconditioner:
        ``"multigrid"`` (default) gives a near grid-independent iteration
        count. ``"jacobi"`` and ``"none"`` exist as baselines; their iteration
        counts grow like O(n).
    pre_sweeps, post_sweeps:
        Red-black Gauss-Seidel sweeps around each coarse-grid correction.
    cycle_index:
        1 for a V-cycle, 2 for a W-cycle (default). The W-cycle is the
        measured default because unsmoothed aggregation needs the stronger
        cycle to stay grid-independent; see the README.
    record_history:
        Record the relative residual after every iteration.

    Raises
    ------
    ValueError
        If ``a`` is not square 2-D, or contains a non-positive or non-finite
        entry, or if a solver parameter is out of range.
    """
    raw = _core.solve(
        a,
        f,
        tol=tol,
        max_iter=max_iter,
        preconditioner=preconditioner,
        pre_sweeps=pre_sweeps,
        post_sweeps=post_sweeps,
        cycle_index=cycle_index,
        record_history=record_history,
    )
    return SolveResult(
        u=raw.u,
        iterations=raw.iterations,
        residual=raw.residual,
        converged=raw.converged,
        levels=raw.levels,
        history=np.asarray(raw.history, dtype=np.float64),
    )


def solve_batch(
    a: ArrayLike,
    f: float | ArrayLike = 1.0,
    *,
    tol: float = _DEFAULT_TOL,
    max_iter: int = 10_000,
    preconditioner: Preconditioner = "multigrid",
    pre_sweeps: int = 1,
    post_sweeps: int = 1,
    cycle_index: int = 2,
    threads: int = 0,
) -> BatchResult:
    """Solve a batch of independent problems, parallelised across samples.

    ``a`` has shape ``(batch, n, n)``. Parallelising over samples rather than
    within each solve keeps every thread on its own working set, which scales
    far better than the intra-solve parallelism when the individual grids are
    small -- the usual case when generating neural-operator training data.

    The GIL is released for the whole batch.
    """
    u, iters, residuals, converged = _core.solve_batch(
        a,
        f,
        tol=tol,
        max_iter=max_iter,
        preconditioner=preconditioner,
        pre_sweeps=pre_sweeps,
        post_sweeps=post_sweeps,
        cycle_index=cycle_index,
        threads=threads,
    )
    return BatchResult(
        u=u,
        iterations=np.asarray(iters, dtype=np.int64),
        residuals=np.asarray(residuals, dtype=np.float64),
        converged=np.asarray(converged, dtype=bool),
    )


def apply_operator(a: ArrayLike, x: ArrayLike) -> NDArray[np.float64]:
    """Apply the finite-volume operator ``A`` to ``x``, matrix-free.

    Exposed mainly so tests can verify symmetry and positive-definiteness of
    the discretisation directly, without reconstructing the matrix.
    """
    return _core.apply_operator(a, x)


def apply_preconditioner(
    a: ArrayLike,
    r: ArrayLike,
    *,
    pre_sweeps: int = 1,
    post_sweeps: int = 1,
    cycle_index: int = 2,
) -> NDArray[np.float64]:
    """Apply one multigrid cycle to ``r``, starting from a zero initial guess.

    This is the preconditioner ``M`` that CG uses. It is exposed so that its
    symmetry and positive-definiteness -- the two properties CG's convergence
    theory rests on -- can be tested directly rather than inferred from solver
    behaviour.
    """
    return _core.apply_preconditioner(
        a, r, pre_sweeps=pre_sweeps, post_sweeps=post_sweeps, cycle_index=cycle_index
    )


def num_threads() -> int:
    """Maximum number of OpenMP threads available to the solver."""
    return _core.num_threads()


def set_num_threads(threads: int) -> None:
    """Set the OpenMP thread count used by subsequent solves."""
    _core.set_num_threads(threads)


def has_openmp() -> bool:
    """Whether the extension was compiled with OpenMP support."""
    return _core.has_openmp()


def build_info() -> dict[str, Any]:
    """Compiler, threading, and version details, for bug reports."""
    return {
        "version": __version__,
        "compiler": _core.__compiler__,
        "openmp": has_openmp(),
        "max_threads": num_threads(),
        "numpy": np.__version__,
    }
