// SPDX-License-Identifier: MIT
//
// Preconditioned Conjugate Gradient driver and the library's public API.
//
// CG is the right Krylov method here because the finite-volume operator is
// symmetric positive definite: it minimises the A-norm of the error over the
// Krylov subspace using a three-term recurrence, so it needs only a handful of
// vectors of storage regardless of iteration count (unlike GMRES).
//
// The operator is never assembled. Each iteration applies the 5-point stencil
// on the fly, which means storage is O(N) with a small constant and every
// kernel is a unit-stride streaming loop -- the arithmetic intensity is low
// enough that the solver is memory-bandwidth-bound, which is the expected
// regime for a sparse iterative method and the reason OpenMP scaling tracks
// memory bandwidth rather than core count.
#pragma once

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "darcy/blas.hpp"
#include "darcy/config.hpp"
#include "darcy/level.hpp"
#include "darcy/multigrid.hpp"

namespace darcy {

enum class Preconditioner {
    None,       ///< Plain CG. Useful as a baseline.
    Jacobi,     ///< Diagonal scaling. Cheap, but iteration count still O(n).
    Multigrid,  ///< Grid-independent iteration count. The default.
};

Preconditioner preconditioner_from_string(const std::string& name);
const char* preconditioner_name(Preconditioner p);

struct SolveOptions {
    double tol = 1e-10;        ///< Relative residual ||b - Au|| / ||b||.
    int max_iter = 10000;      ///< Iteration cap before giving up.
    Preconditioner preconditioner = Preconditioner::Multigrid;
    MultigridOptions multigrid{};
    bool record_history = false;  ///< Keep the per-iteration residual norms.
};

struct SolveResult {
    std::vector<double> u;                 ///< Solution, row-major n x n.
    index_t n = 0;                         ///< Grid size.
    int iterations = 0;                    ///< CG iterations performed.
    double residual = 0.0;                 ///< Final relative residual.
    bool converged = false;                ///< Whether `tol` was reached.
    int levels = 1;                        ///< Multigrid levels used (1 if none).
    std::vector<double> history;           ///< Relative residual per iteration.
};

/// Thrown for invalid inputs (bad shape, non-positive or non-finite data).
class InvalidInput : public std::invalid_argument {
  public:
    using std::invalid_argument::invalid_argument;
};

/// Validate a permeability field. The discretisation divides by permeability
/// sums, so a non-positive or non-finite entry does not merely give a bad
/// answer -- it produces NaNs several thousand iterations later, which is a
/// miserable thing to debug. Fail loudly and immediately instead.
inline void validate_permeability(const double* a, index_t size) {
    for (index_t i = 0; i < size; ++i) {
        if (!(a[i] > 0.0) || !std::isfinite(a[i])) {
            throw InvalidInput(
                "permeability must be finite and strictly positive everywhere; "
                "offending value at flat index " +
                std::to_string(static_cast<long long>(i)));
        }
    }
}

/// Solve -div(a grad u) = f with homogeneous Dirichlet boundary conditions.
///
/// `a` and `rhs` are row-major n x n buffers; `rhs` holds the cell values of f
/// (the h^2 scaling is applied internally).
inline SolveResult solve(const double* DARCY_RESTRICT a, const double* DARCY_RESTRICT rhs, index_t n,
                         const SolveOptions& opts) {
    if (n < 1) throw InvalidInput("grid size must be at least 1");
    if (opts.tol <= 0.0) throw InvalidInput("tol must be positive");
    if (opts.max_iter < 1) throw InvalidInput("max_iter must be at least 1");

    const index_t size = n * n;
    validate_permeability(a, size);

    const double h = 1.0 / static_cast<double>(n + 1);
    const double h2 = h * h;

    Level fine(a, n);

    SolveResult out;
    out.n = n;
    out.u.assign(static_cast<std::size_t>(size), 0.0);

    std::vector<double> b(static_cast<std::size_t>(size));
    for (index_t i = 0; i < size; ++i) b[static_cast<std::size_t>(i)] = rhs[i] * h2;

    const double b_norm = blas::norm(b);
    if (b_norm == 0.0) {  // f == 0 gives u == 0 exactly; no iteration needed.
        out.converged = true;
        return out;
    }

    // The hierarchy takes ownership of the fine level rather than copying it,
    // so afterwards the operator must be reached through `A`, not `fine`.
    std::unique_ptr<Multigrid> mg;
    if (opts.preconditioner == Preconditioner::Multigrid) {
        mg = std::make_unique<Multigrid>(std::move(fine), opts.multigrid);
        out.levels = mg->num_levels();
    }
    const Level& A = mg ? mg->level(0) : fine;
    const std::vector<double>& inv_diag = A.inv_diag();

    std::vector<double> r = b, z(static_cast<std::size_t>(size)), p, ap(static_cast<std::size_t>(size));

    const auto precondition = [&](const std::vector<double>& in, std::vector<double>& to) {
        switch (opts.preconditioner) {
            case Preconditioner::None:
                to = in;
                break;
            case Preconditioner::Jacobi:
                for (index_t i = 0; i < size; ++i) {
                    const auto k = static_cast<std::size_t>(i);
                    to[k] = in[k] * inv_diag[k];
                }
                break;
            case Preconditioner::Multigrid:
                mg->apply(in.data(), to.data());
                break;
        }
    };

    precondition(r, z);
    p = z;
    double rz = blas::dot(r, z);

    out.residual = 1.0;
    if (opts.record_history) out.history.push_back(1.0);

    for (int it = 0; it < opts.max_iter; ++it) {
        A.apply(p.data(), ap.data());
        const double p_ap = blas::dot(p, ap);
        if (!(std::abs(p_ap) > 0.0) || !std::isfinite(p_ap)) {
            // A breakdown means the operator lost definiteness (only possible
            // from pathological input); report rather than divide by zero.
            out.iterations = it;
            out.converged = false;
            return out;
        }
        const double alpha = rz / p_ap;

        const double rr = blas::cg_update(alpha, p, ap, out.u, r);
        out.residual = std::sqrt(rr) / b_norm;
        out.iterations = it + 1;
        if (opts.record_history) out.history.push_back(out.residual);
        if (out.residual < opts.tol) {
            out.converged = true;
            break;
        }

        precondition(r, z);
        const double rz_new = blas::dot(r, z);
        const double beta = rz_new / rz;
        rz = rz_new;
        blas::xpby(z, beta, p);
    }
    return out;
}

/// Convenience overload for a constant forcing term.
inline SolveResult solve_constant_rhs(const double* a, index_t n, double f, const SolveOptions& opts) {
    std::vector<double> rhs(static_cast<std::size_t>(n * n), f);
    return solve(a, rhs.data(), n, opts);
}

inline Preconditioner preconditioner_from_string(const std::string& name) {
    if (name == "multigrid" || name == "mg") return Preconditioner::Multigrid;
    if (name == "jacobi") return Preconditioner::Jacobi;
    if (name == "none") return Preconditioner::None;
    throw InvalidInput("unknown preconditioner '" + name + "'; expected one of: multigrid, jacobi, none");
}

inline const char* preconditioner_name(Preconditioner p) {
    switch (p) {
        case Preconditioner::Multigrid:
            return "multigrid";
        case Preconditioner::Jacobi:
            return "jacobi";
        case Preconditioner::None:
            return "none";
    }
    return "unknown";
}

}  // namespace darcy
