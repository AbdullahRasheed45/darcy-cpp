"""Benchmark harness.

Reports the two things that actually matter for an iterative solver: how the
CG iteration count scales with the grid (the algorithmic question) and how the
wall time compares against a sparse direct factorisation (the practical one).

Timings take the minimum over repeats rather than the mean: the minimum is the
run least polluted by scheduler noise and cache eviction, which is the standard
convention for microbenchmarks.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

import darcy

__all__ = ["BenchmarkRow", "format_table", "run_benchmark", "thread_scaling", "to_json"]


@dataclass
class BenchmarkRow:
    grid: int
    preconditioner: str
    seconds: float
    iterations: int
    levels: int
    residual: float
    scipy_seconds: float | None = None
    relative_error: float | None = None

    @property
    def speedup(self) -> float | None:
        if self.scipy_seconds is None or self.seconds <= 0:
            return None
        return self.scipy_seconds / self.seconds


def _time(fn: Callable[..., Any], repeats: int) -> tuple[float, Any]:
    best = float("inf")
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return best, result


def run_benchmark(
    grids: Sequence[int] = (64, 128, 256, 512),
    preconditioners: Sequence[str] = ("multigrid", "jacobi"),
    repeats: int = 3,
    compare_scipy: bool = True,
    tol: float = 1e-10,
    seed: int = 0,
    verbose: bool = True,
) -> list[BenchmarkRow]:
    """Time the solver across grid sizes and preconditioners."""
    rows: list[BenchmarkRow] = []
    if verbose:
        info = darcy.build_info()
        print(f"darcy-cpp {info['version']} | {info['compiler']}")
        print(f"OpenMP: {info['openmp']} | threads: {info['max_threads']}")
        print()

    for n in grids:
        a = darcy.two_phase_permeability(darcy.gaussian_random_field(n, seed=seed))

        u_ref = None
        scipy_seconds = None
        if compare_scipy:
            try:
                from darcy.reference import solve_reference

                scipy_seconds, u_ref = _time(lambda a=a: solve_reference(a), repeats=1)
            except ImportError:  # pragma: no cover - SciPy is optional
                if verbose:
                    print("scipy not installed; skipping the direct-solver baseline")
                compare_scipy = False

        for name in preconditioners:
            # Jacobi CG on a large grid takes O(n) iterations; cap it rather
            # than let a baseline dominate the whole benchmark run.
            max_iter = 100_000 if name != "multigrid" else 1_000
            seconds, result = _time(
                lambda a=a, name=name, max_iter=max_iter: darcy.solve(
                    a, tol=tol, preconditioner=name, max_iter=max_iter
                ),
                repeats,
            )
            rel_err = None
            if u_ref is not None:
                rel_err = float(np.linalg.norm(result.u - u_ref) / np.linalg.norm(u_ref))
            row = BenchmarkRow(
                grid=n,
                preconditioner=name,
                seconds=seconds,
                iterations=result.iterations,
                levels=result.levels,
                residual=result.residual,
                scipy_seconds=scipy_seconds,
                relative_error=rel_err,
            )
            rows.append(row)
            if verbose:
                print(_format_row(row))
    if verbose:
        print()
        print(format_table(rows))
    return rows


def _format_row(row: BenchmarkRow) -> str:
    speedup = f"{row.speedup:.2f}x" if row.speedup is not None else "-"
    err = f"{row.relative_error:.1e}" if row.relative_error is not None else "-"
    return (
        f"  {row.grid:>5} {row.preconditioner:>10}  "
        f"{row.seconds * 1e3:>9.1f} ms  {row.iterations:>6} iters  "
        f"speedup {speedup:>7}  rel.err {err}"
    )


def format_table(rows: Sequence[BenchmarkRow]) -> str:
    """Render the results as a Markdown table, ready to paste into a README."""
    header = (
        "| grid | preconditioner | CG iters | levels | time (ms) | "
        "SciPy direct (ms) | speed-up | rel. err |"
    )
    sep = "|" + "|".join(["---"] * 8) + "|"
    lines = [header, sep]
    for r in rows:
        scipy_ms = f"{r.scipy_seconds * 1e3:.1f}" if r.scipy_seconds is not None else "-"
        speedup = f"{r.speedup:.2f}x" if r.speedup is not None else "-"
        err = f"{r.relative_error:.1e}" if r.relative_error is not None else "-"
        lines.append(
            f"| {r.grid} | {r.preconditioner} | {r.iterations} | {r.levels} | "
            f"{r.seconds * 1e3:.1f} | {scipy_ms} | {speedup} | {err} |"
        )
    return "\n".join(lines)


def thread_scaling(
    grid: int = 512, threads: Sequence[int] = (1, 2, 4, 8), repeats: int = 3, seed: int = 0
) -> list[dict[str, Any]]:
    """Measure strong scaling of a single solve against the thread count.

    Expect this to saturate well below linear: the stencil and BLAS-1 kernels
    are memory-bandwidth-bound, so once the memory controllers are busy, extra
    cores have nothing left to do.
    """
    a = darcy.two_phase_permeability(darcy.gaussian_random_field(grid, seed=seed))
    original = darcy.num_threads()
    out: list[dict[str, Any]] = []
    try:
        baseline = None
        for t in threads:
            darcy.set_num_threads(t)
            seconds, result = _time(lambda a=a: darcy.solve(a, tol=1e-10), repeats)
            baseline = baseline if baseline is not None else seconds
            out.append(
                {
                    "threads": t,
                    "seconds": seconds,
                    "speedup": baseline / seconds,
                    "iterations": result.iterations,
                }
            )
    finally:
        darcy.set_num_threads(original)
    return out


def to_json(rows: Sequence[BenchmarkRow]) -> list[dict[str, Any]]:
    return [asdict(r) | {"speedup": r.speedup} for r in rows]
