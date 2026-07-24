#!/usr/bin/env python3
"""Benchmark entry point.

Thin wrapper around :mod:`darcy.benchmark` so the suite can be run either as
``darcy bench`` or as a plain script. See that module for the methodology.

Examples
--------
    python benchmarks/bench.py
    python benchmarks/bench.py --grids 64 128 256 --json results.json
    OMP_NUM_THREADS=8 python benchmarks/bench.py --no-scipy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from darcy.benchmark import format_table, run_benchmark, thread_scaling, to_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grids", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument(
        "--preconditioners", nargs="+", default=["multigrid", "jacobi"],
        choices=["multigrid", "jacobi", "none"],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-scipy", action="store_true", help="skip the SciPy baseline")
    parser.add_argument("--thread-scaling", action="store_true", help="also measure strong scaling")
    parser.add_argument("--json", type=Path, help="write machine-readable results here")
    args = parser.parse_args()

    rows = run_benchmark(
        args.grids,
        preconditioners=args.preconditioners,
        repeats=args.repeats,
        compare_scipy=not args.no_scipy,
    )

    if args.thread_scaling:
        print("\nStrong scaling (single 512x512 solve):")
        for row in thread_scaling():
            print(f"  {row['threads']:>2} threads: {row['seconds'] * 1e3:8.1f} ms  "
                  f"({row['speedup']:.2f}x)")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(to_json(rows), indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
