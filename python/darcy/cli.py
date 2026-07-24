"""Command-line interface: ``darcy solve | dataset | bench | info``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import darcy


def _add_solver_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tol", type=float, default=1e-10, help="relative residual tolerance")
    p.add_argument("--max-iter", type=int, default=10_000, help="iteration cap")
    p.add_argument(
        "--preconditioner",
        choices=["multigrid", "jacobi", "none"],
        default="multigrid",
        help="preconditioner (default: multigrid)",
    )
    p.add_argument("--pre-sweeps", type=int, default=1, help="pre-smoothing sweeps per level")
    p.add_argument("--post-sweeps", type=int, default=1, help="post-smoothing sweeps per level")
    p.add_argument("--cycle-index", type=int, default=2, choices=[1, 2], help="1 = V-cycle, 2 = W-cycle")


def _permeability(grid: int, seed: int, alpha: float, tau: float) -> np.ndarray:
    return darcy.two_phase_permeability(
        darcy.gaussian_random_field(grid, alpha=alpha, tau=tau, seed=seed)
    )


def cmd_solve(args: argparse.Namespace) -> int:
    if args.input is not None:
        a = np.load(args.input)
        if isinstance(a, np.lib.npyio.NpzFile):
            a = a[args.key]
        a = np.asarray(a, dtype=np.float64)
        if a.ndim != 2:
            print(f"error: expected a 2-D field in {args.input}, got shape {a.shape}", file=sys.stderr)
            return 2
    else:
        a = _permeability(args.grid, args.seed, args.alpha, args.tau)

    start = time.perf_counter()
    result = darcy.solve(
        a,
        args.f,
        tol=args.tol,
        max_iter=args.max_iter,
        preconditioner=args.preconditioner,
        pre_sweeps=args.pre_sweeps,
        post_sweeps=args.post_sweeps,
        cycle_index=args.cycle_index,
    )
    elapsed = time.perf_counter() - start

    print(f"grid          {a.shape[0]} x {a.shape[0]}")
    print(f"preconditioner {args.preconditioner} ({result.levels} levels)")
    print(f"iterations    {result.iterations}")
    print(f"residual      {result.residual:.3e}")
    print(f"converged     {result.converged}")
    print(f"wall time     {elapsed:.4f} s")
    print(f"max |u|       {np.abs(result.u).max():.6e}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, a=a, u=result.u)
        print(f"wrote         {args.output}")
    return 0 if result.converged else 1


def cmd_dataset(args: argparse.Namespace) -> int:
    """Generate an (a, u) dataset for neural-operator training."""
    rng = np.random.default_rng(args.seed)
    a_all = np.empty((args.n_samples, args.grid, args.grid), dtype=np.float64)
    for i in range(args.n_samples):
        a_all[i] = darcy.two_phase_permeability(
            darcy.gaussian_random_field(args.grid, alpha=args.alpha, tau=args.tau, seed=rng)
        )

    start = time.perf_counter()
    batch = darcy.solve_batch(
        a_all,
        args.f,
        tol=args.tol,
        max_iter=args.max_iter,
        preconditioner=args.preconditioner,
        threads=args.threads,
    )
    elapsed = time.perf_counter() - start

    if not batch.all_converged:
        failed = int((~batch.converged).sum())
        print(f"error: {failed}/{args.n_samples} samples did not converge", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        a=a_all.astype(args.dtype),
        u=batch.u.astype(args.dtype),
    )
    per = elapsed / args.n_samples
    print(f"wrote {args.output}: {args.n_samples} samples at {args.grid}x{args.grid}")
    print(
        f"  {elapsed:.2f} s total, {per * 1e3:.1f} ms/sample, "
        f"{batch.iterations.mean():.1f} mean CG iterations"
    )
    print(f"  threads: {args.threads or darcy.num_threads()}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from darcy.benchmark import run_benchmark, to_json

    rows = run_benchmark(
        args.grids,
        preconditioners=args.preconditioners,
        repeats=args.repeats,
        compare_scipy=not args.no_scipy,
    )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(to_json(rows), indent=2))
        print(f"wrote {args.json}")
    return 0


def cmd_info(_: argparse.Namespace) -> int:
    print(json.dumps(darcy.build_info(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="darcy",
        description="Matrix-free multigrid solver for 2-D Darcy flow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"darcy-cpp {darcy.__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    fmt = parser.formatter_class

    solve = sub.add_parser("solve", help="solve one problem", formatter_class=fmt)
    solve.add_argument("--grid", type=int, default=128, help="grid size when generating a field")
    solve.add_argument("--seed", type=int, default=0, help="random seed for the generated field")
    solve.add_argument("--alpha", type=float, default=2.0, help="GRF smoothness")
    solve.add_argument("--tau", type=float, default=3.0, help="GRF inverse correlation length")
    solve.add_argument("--f", type=float, default=1.0, help="constant forcing term")
    solve.add_argument("--input", type=Path, help="load the permeability field from .npy/.npz")
    solve.add_argument("--key", default="a", help="array key when --input is an .npz")
    solve.add_argument("--output", type=Path, help="write (a, u) to this .npz")
    _add_solver_flags(solve)
    solve.set_defaults(func=cmd_solve)

    dataset = sub.add_parser("dataset", help="generate a training dataset", formatter_class=fmt)
    dataset.add_argument("--n-samples", type=int, default=1000)
    dataset.add_argument("--grid", type=int, default=64)
    dataset.add_argument("--seed", type=int, default=0)
    dataset.add_argument("--alpha", type=float, default=2.0)
    dataset.add_argument("--tau", type=float, default=3.0)
    dataset.add_argument("--f", type=float, default=1.0)
    dataset.add_argument("--threads", type=int, default=0, help="0 uses every available thread")
    dataset.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    dataset.add_argument("--output", type=Path, default=Path("data/darcy_train.npz"))
    _add_solver_flags(dataset)
    dataset.set_defaults(func=cmd_dataset)

    bench = sub.add_parser("bench", help="run the benchmark suite", formatter_class=fmt)
    bench.add_argument("--grids", type=int, nargs="+", default=[64, 128, 256, 512])
    bench.add_argument("--repeats", type=int, default=3)
    bench.add_argument("--no-scipy", action="store_true", help="skip the SciPy direct-solver baseline")
    bench.add_argument(
        "--preconditioners",
        nargs="+",
        default=["multigrid", "jacobi"],
        choices=["multigrid", "jacobi", "none"],
    )
    bench.add_argument("--json", type=Path, help="also write the results as JSON")
    bench.set_defaults(func=cmd_bench)

    info = sub.add_parser("info", help="print build and threading information")
    info.set_defaults(func=cmd_info)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ValueError as exc:  # solver input validation
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
