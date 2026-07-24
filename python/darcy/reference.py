"""SciPy reference implementation, used to validate the C++ solver.

This assembles the finite-volume operator as an explicit sparse matrix and
factorises it directly. It is deliberately written independently of the C++
code -- a different algorithm (direct rather than iterative) built from a
different representation (assembled rather than matrix-free) -- so that
agreement between the two is real evidence of correctness rather than two
copies of the same mistake.

Requires SciPy, which is an optional dependency of the package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["assemble_matrix", "solve_reference"]


def _harmonic(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    return 2.0 * x * y / (x + y)


def assemble_matrix(a: NDArray[np.float64]) -> Any:
    """Assemble the sparse finite-volume operator for permeability ``a``.

    Returns a SciPy CSR matrix of shape ``(n*n, n*n)`` (typed loosely because
    SciPy ships no stubs and is only an optional dependency).
    """
    from scipy import sparse

    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("a must be a square 2-D array")
    n = a.shape[0]

    a_e = np.zeros((n, n))
    a_w = np.zeros((n, n))
    a_n = np.zeros((n, n))
    a_s = np.zeros((n, n))
    a_e[:, :-1] = _harmonic(a[:, :-1], a[:, 1:])
    a_e[:, -1] = a[:, -1]
    a_w[:, 1:] = _harmonic(a[:, 1:], a[:, :-1])
    a_w[:, 0] = a[:, 0]
    a_n[:-1, :] = _harmonic(a[:-1, :], a[1:, :])
    a_n[-1, :] = a[-1, :]
    a_s[1:, :] = _harmonic(a[1:, :], a[:-1, :])
    a_s[0, :] = a[0, :]

    idx = np.arange(n * n).reshape(n, n)
    rows = [idx.ravel()]
    cols = [idx.ravel()]
    vals = [(a_e + a_w + a_n + a_s).ravel()]

    def add(r: NDArray[np.int64], c: NDArray[np.int64], v: NDArray[np.float64]) -> None:
        rows.append(r.ravel())
        cols.append(c.ravel())
        vals.append(-v.ravel())

    add(idx[:, :-1], idx[:, 1:], a_e[:, :-1])
    add(idx[:, 1:], idx[:, :-1], a_w[:, 1:])
    add(idx[:-1, :], idx[1:, :], a_n[:-1, :])
    add(idx[1:, :], idx[:-1, :], a_s[1:, :])

    return sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n * n, n * n),
    )


def solve_reference(a: NDArray[np.float64], f: float | NDArray[np.float64] = 1.0) -> NDArray[np.float64]:
    """Solve the Darcy problem with a sparse direct factorisation."""
    from scipy.sparse.linalg import spsolve

    a = np.asarray(a, dtype=np.float64)
    n = a.shape[0]
    h = 1.0 / (n + 1)
    matrix = assemble_matrix(a)
    rhs = np.full(n * n, f, dtype=np.float64) if np.isscalar(f) else np.asarray(f, float).ravel()
    return np.asarray(spsolve(matrix, rhs * h * h)).reshape(n, n)
