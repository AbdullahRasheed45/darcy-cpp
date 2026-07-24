# Design notes

This document explains *why* the solver is built the way it is. The code
comments cover the local "what"; this covers the decisions a reviewer or
interviewer would reasonably ask about.

## The problem and why it is hard

We solve the second-order elliptic PDE

$$-\nabla\cdot(a\,\nabla u) = f, \qquad u|_{\partial\Omega}=0$$

on the unit square, where `a` is a piecewise-constant (two-phase) permeability
field with jumps of up to six orders of magnitude. Discretised, this is a large
sparse symmetric positive-definite (SPD) linear system `A u = b`.

The difficulty is the **conditioning**. The eigenvalues of the discrete
Laplacian-like operator span a range that grows like `O(n²)` with the grid
size `n`. Conjugate Gradient converges in a number of iterations proportional to
`√κ(A) = O(n)`, so a naive solve costs `O(n) × O(N)` work per solve — the cost
per pixel grows with resolution. Diagonal (Jacobi) preconditioning rescales
individual equations but does nothing to the `O(n²)` spread, so it does not
change the asymptotics. This is visible directly in the benchmarks: Jacobi-CG
goes 282 → 4448 iterations from 64² to 1024².

Everything below is in service of removing that `√κ` factor.

## Why finite volume with harmonic averaging

A cell-centred finite-volume scheme integrates the PDE over each cell and
balances fluxes across faces. The flux across a face depends on the permeability
*at* the face, and for a coefficient that jumps between two cells the physically
correct face value is the **harmonic mean**, not the arithmetic mean:

- Harmonic averaging is the *exact* effective conductivity of two resistors in
  series, i.e. the exact 1-D flux for a piecewise-constant coefficient. It
  preserves flux continuity across a jump.
- Arithmetic averaging is inconsistent at a jump and loses an order of accuracy
  on exactly the discontinuous media this solver targets.

The resulting operator is SPD (verified numerically in both test suites), which
is what makes CG applicable.

## Why matrix-free

The operator is never assembled as a sparse matrix. Instead the 5-point stencil
is applied on the fly from six arrays of face coefficients. This is a deliberate
trade:

- **Storage is `O(N)`** with a tiny constant and no index arrays. A direct
  factorisation suffers fill-in — the factors are far denser than `A` — which is
  the real reason `spsolve` falls off a cliff at 1024² in the benchmarks and
  becomes untenable in 3-D.
- **Every kernel is a unit-stride streaming loop**, which is cache-friendly and
  trivially vectorises and parallelises.
- The cost is that we must *iterate*, which is precisely why the preconditioner
  matters so much.

The arithmetic intensity of the stencil is low (a handful of flops per double
loaded), so the solver is **memory-bandwidth-bound**. That single fact predicts
the parallel scaling (see below) and tells you a roofline analysis, not a
flop count, is the right performance model.

## Why Conjugate Gradient

For an SPD system CG is the natural Krylov method: it minimises the error in the
`A`-norm over the Krylov subspace using a three-term recurrence, so it needs
only a constant number of vectors regardless of iteration count (unlike GMRES,
which stores the whole basis). The residual is not monotone in the 2-norm — the
first preconditioned step can and does raise it — but it is monotone in the
`A`-norm, which is why the residual-history test allows the first step to jump
and forbids oscillation thereafter.

## The multigrid preconditioner — the core of the project

Multigrid attacks the conditioning problem at its source. The insight is a
spectral one:

- A cheap smoother (here **red-black Gauss-Seidel**) rapidly annihilates
  *high-frequency* error but barely touches *smooth* error.
- Smooth error on a fine grid looks high-frequency on a coarser grid, where it
  can be cheaply corrected.

Recursing this idea across a hierarchy of grids produces a preconditioner whose
quality does **not** degrade with `n`. The result is the flat iteration-count
column in the README: 21 → 31 across a 16× refinement.

### Coarsening: agglomeration that is exactly Galerkin

Coarse levels are built by 2×2 agglomeration with piecewise-constant transfer
operators — restriction `R = Pᵀ`, prolongation `P` injecting each coarse value
into its four children. The key structural fact, proved in the header of
`level.hpp` and checked entry-by-entry in the C++ tests, is that the Galerkin
coarse operator `PᵀAP` is **again a 5-point stencil of the same form**, and its
coefficients are obtained by *summing* the fine face coefficients that cross
each coarse face. So:

- coarsening costs `O(N)` and touches no sparse triple product;
- yet the coarse operator is algebraically identical to the Galerkin product,
  which is what guarantees it stays SPD — the precondition for the whole cycle
  being a valid CG preconditioner.

Handling **odd grid sizes** falls out of the same summation (the trailing
aggregate simply has width one), so the hierarchy does not require power-of-two
grids — tested explicitly at n = 17, 31, 33, 47, 63.

### Symmetry of the cycle

CG requires an SPD preconditioner. A multigrid V-cycle is SPD **only if it is
symmetric**: pre-smoothing and post-smoothing must be transposes of each other.
Here that means mirroring the red-black sweep order (red-then-black going down,
black-then-red coming up) and using a symmetric coarsest-grid solve. This is not
cosmetic — swap the post-smoothing colours and the preconditioner-symmetry test
fails immediately, while the solver still *appears* to work but with a degraded,
erratic residual history. The test suite pins this down directly by checking
`⟨Mr, s⟩ = ⟨r, Ms⟩` on random vectors.

### V-cycle vs W-cycle, and why W is the default

Piecewise-constant (unsmoothed) aggregation gives a weaker coarse-grid
correction than smoothed-aggregation or geometric interpolation would. A
V-cycle inherits that weakness: its CG count still creeps up with `n`
(25 → 102 over 32²–512²). The **W-cycle** visits the coarse levels twice per
level and compensates, staying genuinely flat (18 → 28). Because coarse levels
are geometrically small, the extra visits cost little — so the W-cycle wins on
*wall-clock* too, and is the default.

### The fork/join guard — a performance lesson

The W-cycle exposed a subtle performance trap. A multigrid hierarchy spends most
of its *visits* on tiny grids (a W-cycle reaches the coarsest level `2^depth`
times), and naively every kernel there opened an OpenMP parallel region. An
OpenMP fork/join costs microseconds; a 16×16 sweep costs far less. The overhead
dominated, and the W-cycle measured **~30× a V-cycle** instead of the ~2× the
flop count predicts.

The fix is one guard: kernels run serially below ~4096 cells (`if(parallel_)` on
each `#pragma omp parallel for`). That single change cut W-cycle wall time by
roughly 4× and is why the benchmark numbers are what they are. It is a good
concrete example of profiling beating intuition.

## Parallelism strategy

Two levels of parallelism are available, and the library uses the right one for
each situation:

1. **Within a solve** — every kernel is an OpenMP loop. But because the solver
   is memory-bandwidth-bound, a single large solve saturates the memory system
   at ~1.2× on 4 threads. This is expected and documented, not hidden.
2. **Across solves** — `solve_batch` parallelises over independent samples with
   a released GIL. Each thread runs a *serial* solve on its own working set, so
   there is no bandwidth contention between the inner kernels and the batch
   scales far better. This is the mode that matters for the actual use case
   (generating operator-learning datasets), reaching ~2,300 64² solves/s.

Determinism is preserved across thread counts: red-black Gauss-Seidel and
static-scheduled reductions give bit-reproducible results regardless of the
schedule, which the batch tests assert directly.

## Correctness strategy

Three independent layers, because "it converged" is not "it's right":

1. **Cross-validation against an independent method.** The SciPy reference
   assembles the operator as a sparse matrix and solves it with a *direct*
   factorisation — a different algorithm built from a different representation.
   Agreement to `10⁻¹²` is therefore real evidence, not two copies of one bug.
2. **Analytic invariants.** Properties the exact solution must satisfy —
   symmetry of the operator and solution, SPD-ness, linearity in `f`, inverse
   scaling in `a`, the maximum principle, an honest recomputed residual — hold
   even where no reference solution exists.
3. **Structural identities in C++.** The Galerkin coarsening identity and the
   `R = Pᵀ` adjointness are checked directly against dense reference matrices,
   isolating the multigrid machinery from the solver that uses it.

## What I would do next

- **Smoothed aggregation.** Replacing piecewise-constant `P` with one Jacobi
  smoothing step would strengthen the coarse correction enough that a *V-cycle*
  becomes grid-independent, roughly halving the work per solve.
- **3-D.** The matrix-free design carries over directly; a 7-point stencil and
  2×2×2 agglomeration are the only changes, and this is where avoiding fill-in
  stops being an optimisation and becomes the only viable option.
- **Mixed precision.** Run the preconditioner in single precision inside a
  double-precision CG — the cycle is only an approximate inverse, so it tolerates
  the lower precision while roughly doubling effective bandwidth.
- **A GPU backend.** Every kernel is already a data-parallel loop; the red-black
  smoother and matrix-free apply map cleanly onto a GPU.
