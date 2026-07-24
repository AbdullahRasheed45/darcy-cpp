#!/usr/bin/env python3
"""A five-minute tour of the solver.

Run with:  python examples/quickstart.py
"""

from __future__ import annotations

import numpy as np

import darcy


def main() -> None:
    print("build:", darcy.build_info(), "\n")

    # 1. A single solve on a random two-phase medium ----------------------
    a = darcy.two_phase_permeability(darcy.gaussian_random_field(256, seed=0))
    result = darcy.solve(a, f=1.0, tol=1e-10)
    print("single solve:", result)
    print(f"  peak head |u|_max = {np.abs(result.u).max():.4e}")
    print(f"  grid-independent: {result.iterations} iterations over {result.levels} levels\n")

    # 2. Compare preconditioners on the same problem ----------------------
    print("preconditioner comparison (256x256):")
    for name in ("multigrid", "jacobi", "none"):
        r = darcy.solve(a, preconditioner=name, max_iter=100_000)
        print(f"  {name:>10}: {r.iterations:>5} iterations")
    print()

    # 3. Batched dataset generation, parallel across samples --------------
    batch_a = np.stack(
        [darcy.two_phase_permeability(darcy.gaussian_random_field(64, seed=s)) for s in range(64)]
    )
    batch = darcy.solve_batch(batch_a, tol=1e-10)
    print("batch solve:", batch)
    print(f"  all converged: {batch.all_converged}, mean iters: {batch.iterations.mean():.1f}\n")

    # 4. Verify against SciPy if it is installed --------------------------
    try:
        from darcy.reference import solve_reference

        expected = solve_reference(a)
        rel_err = np.linalg.norm(result.u - expected) / np.linalg.norm(expected)
        print(f"agreement with SciPy sparse direct solve: {rel_err:.2e} relative error")
    except ImportError:
        print("(install scipy to cross-check against a direct solver)")


if __name__ == "__main__":
    main()
