#!/usr/bin/env python3
"""Render a permeability field, its solution, and the convergence history.

Needs matplotlib:  pip install matplotlib
Run with:          python examples/plot_solution.py --grid 256 --out darcy.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import darcy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("darcy.png"))
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("this example needs matplotlib: pip install matplotlib") from exc

    a = darcy.two_phase_permeability(darcy.gaussian_random_field(args.grid, seed=args.seed))
    result = darcy.solve(a, tol=1e-10, record_history=True)
    print(result)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    im0 = axes[0].imshow(a, cmap="viridis", origin="lower")
    axes[0].set_title("permeability a(x)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(result.u, cmap="magma", origin="lower")
    axes[1].set_title(f"pressure u(x)  ({result.iterations} CG iters)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    axes[2].semilogy(result.history, marker="o", markersize=3)
    axes[2].set_title("relative residual")
    axes[2].set_xlabel("iteration")
    axes[2].grid(True, which="both", alpha=0.3)

    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
