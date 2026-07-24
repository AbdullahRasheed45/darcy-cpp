// SPDX-License-Identifier: MIT
//
// One level of the multigrid hierarchy: a symmetric positive-definite 5-point
// operator stored as its face transmissibilities.
//
// DISCRETISATION
// --------------
// The PDE is  -div(a grad u) = f  on (0,1)^2 with u = 0 on the boundary,
// discretised by a cell-centred finite-volume scheme on an n x n grid with
// spacing h = 1/(n+1). Integrating over a cell and applying the divergence
// theorem gives, for cell k = (i,j),
//
//     T_e (u_k - u_E) + T_w (u_k - u_W) + T_n (u_k - u_N) + T_s (u_k - u_S)
//         = f * h^2
//
// where the transmissibility of a face is the HARMONIC mean of the two
// adjacent permeabilities. Harmonic averaging is not a stylistic choice: it is
// the exact 1D flux for a piecewise-constant coefficient, so it preserves flux
// continuity across permeability jumps. Arithmetic averaging does not, and
// loses an order of accuracy on the two-phase media this solver targets.
//
// A face on the domain boundary sees a ghost value of zero at distance h, so
// its transmissibility is simply the cell's own permeability. The h^2 factors
// cancel between the operator and the right-hand side, so they never appear.
//
// COARSENING
// ----------
// Coarse levels are built by 2x2 agglomeration with a piecewise-constant
// prolongation P (each coarse cell injects into its 2x2 children) and the
// matching restriction R = P^T. The Galerkin coarse operator A_c = P^T A P is
// then *exactly* representable as another 5-point stencil of the same form:
//
//   - aggregates that touch only at a corner share no face, so they do not
//     couple: the coarse stencil stays 5-point;
//   - a coarse face coefficient is the sum of the fine face coefficients
//     crossing that coarse face;
//   - within an aggregate the internal couplings cancel against the diagonal
//     exactly, so the coarse diagonal is again the sum of its four coarse face
//     coefficients.
//
// That means the hierarchy is built by summing coefficients (O(N) work, no
// sparse triple product) while remaining algebraically identical to Galerkin
// coarsening -- which is what guarantees the coarse-grid correction is a
// well-posed SPD problem, even for the discontinuous coefficients this solver
// is built for.
#pragma once

#include <cmath>
#include <memory>
#include <vector>

#include "darcy/config.hpp"

namespace darcy {

/// Harmonic mean of two positive numbers.
inline double harmonic_mean(double x, double y) noexcept { return 2.0 * x * y / (x + y); }

/// A 5-point SPD operator on an n x n grid, stored face-wise.
///
/// Storage is 6 doubles per cell (four faces, the diagonal, and its reciprocal)
/// rather than a sparse matrix: no index arrays, no indirection, and the whole
/// stencil apply is a unit-stride streaming loop.
class Level {
  public:
    /// Below this many cells, a level runs its kernels serially.
    ///
    /// This matters far more than it looks. A multigrid hierarchy spends most
    /// of its *visits* on tiny levels -- a W-cycle reaches the coarsest grid
    /// 2^depth times -- and an OpenMP fork/join costs microseconds while a
    /// 32x32 sweep costs well under one. Without this guard the thread-team
    /// overhead dominates and the W-cycle measures ~30x a V-cycle per cycle
    /// instead of the ~2x the flop count predicts.
    static constexpr index_t kMinParallelCells = 4096;

    Level() = default;

    /// Build the finest level from a permeability field `a` (row-major, n x n).
    Level(const double* DARCY_RESTRICT a, index_t n)
        : n_(n), size_(n * n), parallel_(n * n >= kMinParallelCells) {
        allocate();
        const auto A = [&](index_t i, index_t j) { return a[i * n_ + j]; };
#pragma omp parallel for schedule(static) if (parallel_)
        for (index_t i = 0; i < n_; ++i) {
            for (index_t j = 0; j < n_; ++j) {
                const index_t k = i * n_ + j;
                const double c = A(i, j);
                e_[k] = (j + 1 < n_) ? harmonic_mean(c, A(i, j + 1)) : c;
                w_[k] = (j > 0) ? harmonic_mean(c, A(i, j - 1)) : c;
                n_north_[k] = (i + 1 < n_) ? harmonic_mean(c, A(i + 1, j)) : c;
                s_[k] = (i > 0) ? harmonic_mean(c, A(i - 1, j)) : c;
                const double d = e_[k] + w_[k] + n_north_[k] + s_[k];
                diag_[k] = d;
                inv_diag_[k] = 1.0 / d;
            }
        }
    }

    index_t n() const noexcept { return n_; }
    index_t size() const noexcept { return size_; }
    const std::vector<double>& diag() const noexcept { return diag_; }
    const std::vector<double>& inv_diag() const noexcept { return inv_diag_; }

    /// y = A x.
    void apply(const double* DARCY_RESTRICT x, double* DARCY_RESTRICT y) const noexcept {
#pragma omp parallel for schedule(static) if (parallel_)
        for (index_t i = 0; i < n_; ++i) {
            const index_t row = i * n_;
            for (index_t j = 0; j < n_; ++j) {
                const index_t k = row + j;
                double v = diag_[k] * x[k];
                if (j + 1 < n_) v -= e_[k] * x[k + 1];
                if (j > 0) v -= w_[k] * x[k - 1];
                if (i + 1 < n_) v -= n_north_[k] * x[k + n_];
                if (i > 0) v -= s_[k] * x[k - n_];
                y[k] = v;
            }
        }
    }

    /// r = b - A x.
    void residual(const double* DARCY_RESTRICT x, const double* DARCY_RESTRICT b,
                  double* DARCY_RESTRICT r) const noexcept {
#pragma omp parallel for schedule(static) if (parallel_)
        for (index_t i = 0; i < n_; ++i) {
            const index_t row = i * n_;
            for (index_t j = 0; j < n_; ++j) {
                const index_t k = row + j;
                double v = diag_[k] * x[k];
                if (j + 1 < n_) v -= e_[k] * x[k + 1];
                if (j > 0) v -= w_[k] * x[k - 1];
                if (i + 1 < n_) v -= n_north_[k] * x[k + n_];
                if (i > 0) v -= s_[k] * x[k - n_];
                r[k] = b[k] - v;
            }
        }
    }

    /// One red-black Gauss-Seidel sweep over cells with (i + j) % 2 == colour.
    ///
    /// Red-black ordering makes Gauss-Seidel -- normally inherently sequential
    /// -- embarrassingly parallel: same-colour cells only depend on cells of
    /// the other colour, so a whole colour updates concurrently and the result
    /// is bit-identical regardless of thread count.
    void gauss_seidel_sweep(double* DARCY_RESTRICT x, const double* DARCY_RESTRICT b,
                            int colour) const noexcept {
#pragma omp parallel for schedule(static) if (parallel_)
        for (index_t i = 0; i < n_; ++i) {
            const index_t row = i * n_;
            // Start at the first column of this colour, then stride by two.
            for (index_t j = static_cast<index_t>((i + colour) & 1); j < n_; j += 2) {
                const index_t k = row + j;
                double v = b[k];
                if (j + 1 < n_) v += e_[k] * x[k + 1];
                if (j > 0) v += w_[k] * x[k - 1];
                if (i + 1 < n_) v += n_north_[k] * x[k + n_];
                if (i > 0) v += s_[k] * x[k - n_];
                x[k] = v * inv_diag_[k];
            }
        }
    }

    /// Number of cells along one axis after one step of 2x2 agglomeration.
    index_t coarse_n() const noexcept { return (n_ + 1) / 2; }

    /// Galerkin coarse operator P^T A P for piecewise-constant P (see header).
    ///
    /// An odd n leaves a final aggregate of width one; the summation below
    /// handles that without a special case, so the hierarchy does not require
    /// power-of-two grids.
    Level coarsen() const {
        Level c;
        c.n_ = coarse_n();
        c.size_ = c.n_ * c.n_;
        c.parallel_ = c.size_ >= kMinParallelCells;
        c.allocate();

        const index_t nc = c.n_;
        const auto last_fine = [&](index_t I) { return (2 * I + 1 < n_) ? 2 * I + 1 : 2 * I; };

#pragma omp parallel for schedule(static) if (parallel_)
        for (index_t I = 0; I < nc; ++I) {
            const index_t i0 = 2 * I;
            const index_t i1 = last_fine(I);
            for (index_t J = 0; J < nc; ++J) {
                const index_t j0 = 2 * J;
                const index_t j1 = last_fine(J);
                double ce = 0.0, cw = 0.0, cn = 0.0, cs = 0.0;
                // East/west coarse faces: sum over the fine rows of the
                // aggregate, taking the fine faces on its right/left edge.
                for (index_t i = i0; i <= i1; ++i) {
                    ce += e_[i * n_ + j1];
                    cw += w_[i * n_ + j0];
                }
                // North/south coarse faces: same, over the fine columns.
                for (index_t j = j0; j <= j1; ++j) {
                    cn += n_north_[i1 * n_ + j];
                    cs += s_[i0 * n_ + j];
                }
                const index_t k = I * nc + J;
                c.e_[k] = ce;
                c.w_[k] = cw;
                c.n_north_[k] = cn;
                c.s_[k] = cs;
                const double d = ce + cw + cn + cs;
                c.diag_[k] = d;
                c.inv_diag_[k] = 1.0 / d;
            }
        }
        return c;
    }

    /// Restriction r_c = P^T r: sum the fine residual over each 2x2 aggregate.
    void restrict_to(const double* DARCY_RESTRICT fine, double* DARCY_RESTRICT coarse) const noexcept {
        const index_t nc = coarse_n();
#pragma omp parallel for schedule(static) if (parallel_)
        for (index_t I = 0; I < nc; ++I) {
            const index_t i1 = (2 * I + 1 < n_) ? 2 * I + 1 : 2 * I;
            for (index_t J = 0; J < nc; ++J) {
                const index_t j1 = (2 * J + 1 < n_) ? 2 * J + 1 : 2 * J;
                double s = 0.0;
                for (index_t i = 2 * I; i <= i1; ++i)
                    for (index_t j = 2 * J; j <= j1; ++j) s += fine[i * n_ + j];
                coarse[I * nc + J] = s;
            }
        }
    }

    /// Prolongation x += P x_c: inject each coarse value into its 2x2 children.
    void prolong_add(const double* DARCY_RESTRICT coarse, double* DARCY_RESTRICT fine) const noexcept {
        const index_t nc = coarse_n();
#pragma omp parallel for schedule(static) if (parallel_)
        for (index_t i = 0; i < n_; ++i) {
            const index_t I = i / 2;
            for (index_t j = 0; j < n_; ++j) fine[i * n_ + j] += coarse[I * nc + j / 2];
        }
    }

  private:
    void allocate() {
        e_.resize(static_cast<std::size_t>(size_));
        w_.resize(static_cast<std::size_t>(size_));
        n_north_.resize(static_cast<std::size_t>(size_));
        s_.resize(static_cast<std::size_t>(size_));
        diag_.resize(static_cast<std::size_t>(size_));
        inv_diag_.resize(static_cast<std::size_t>(size_));
    }

    index_t n_ = 0;
    index_t size_ = 0;
    bool parallel_ = false;
    // Face transmissibilities. `n_north_` is spelled out because `n_` is taken.
    std::vector<double> e_, w_, n_north_, s_, diag_, inv_diag_;
};

}  // namespace darcy
