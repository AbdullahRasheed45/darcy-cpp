// SPDX-License-Identifier: MIT
//
// Geometric-algebraic multigrid preconditioner.
//
// Jacobi-preconditioned CG on this operator needs O(n) iterations, because the
// condition number of the discrete elliptic operator grows like O(n^2) and a
// diagonal scaling does nothing about it. Multigrid fixes the actual problem:
// the smoother kills high-frequency error quickly but stalls on smooth error,
// and smooth error is exactly what a coarser grid represents well. Recursing
// gives a preconditioner whose quality does not degrade with n, so the CG
// iteration count becomes essentially grid-independent and the whole solve is
// O(N) instead of O(N^1.5).
//
// The cycle is kept SYMMETRIC (pre-smoothing red-then-black, post-smoothing
// black-then-red, symmetric coarse solve) because CG requires an SPD
// preconditioner -- an unsymmetric V-cycle silently destroys the convergence
// theory and shows up as erratic residual histories.
#pragma once

#include <algorithm>
#include <vector>

#include "darcy/blas.hpp"
#include "darcy/config.hpp"
#include "darcy/level.hpp"

namespace darcy {

struct MultigridOptions {
    /// Red-black Gauss-Seidel sweeps before restriction (one = red + black).
    int pre_sweeps = 1;
    /// Sweeps after prolongation. Kept equal to `pre_sweeps` for symmetry.
    int post_sweeps = 1;
    /// Recursive calls per level: 1 = V-cycle, 2 = W-cycle.
    ///
    /// Defaults to 2. Piecewise-constant (unsmoothed) aggregation gives a
    /// weaker coarse-grid correction than smoothed interpolation would, and a
    /// V-cycle inherits that weakness: its iteration count still creeps up with
    /// n (measured 25 -> 102 from n=32 to n=512). The W-cycle compensates and
    /// is genuinely near grid-independent (18 -> 28 over the same range) while
    /// costing only ~2x per cycle in 2D, so it wins on wall clock too.
    int cycle_index = 2;
    /// Stop coarsening once a level has at most this many cells per side.
    index_t min_coarse_n = 1;
    /// Hard cap on hierarchy depth.
    int max_levels = 32;
};

/// A multigrid hierarchy usable as an SPD preconditioner.
///
/// NOT thread-safe: `apply` writes into internal scratch buffers. Give each
/// thread its own instance (see `solve_batch`).
class Multigrid {
  public:
    Multigrid(Level fine, MultigridOptions opts) : opts_(opts) {
        levels_.push_back(std::move(fine));
        while (static_cast<int>(levels_.size()) < opts_.max_levels) {
            const Level& top = levels_.back();
            if (top.n() <= opts_.min_coarse_n || top.n() <= 1) break;
            levels_.push_back(top.coarsen());
        }
        x_.reserve(levels_.size());
        b_.reserve(levels_.size());
        r_.reserve(levels_.size());
        for (const Level& l : levels_) {
            const auto m = static_cast<std::size_t>(l.size());
            x_.emplace_back(m);
            b_.emplace_back(m);
            r_.emplace_back(m);
        }
    }

    int num_levels() const noexcept { return static_cast<int>(levels_.size()); }
    const Level& level(int i) const noexcept { return levels_[static_cast<std::size_t>(i)]; }

    /// Total doubles held by the hierarchy (operators + scratch).
    ///
    /// In 2D each level is a quarter of the one above, so the geometric series
    /// caps total storage at 4/3 of the fine level: multigrid is not the memory
    /// hog a direct factorisation is.
    std::size_t storage_doubles() const noexcept {
        std::size_t total = 0;
        for (const Level& l : levels_) total += static_cast<std::size_t>(l.size()) * 9;
        return total;
    }

    /// z = M^{-1} r, one cycle starting from a zero initial guess.
    void apply(const double* DARCY_RESTRICT r, double* DARCY_RESTRICT z) const {
        std::copy(r, r + levels_[0].size(), b_[0].begin());
        blas::fill(x_[0], 0.0);
        run_cycle(0);
        std::copy(x_[0].begin(), x_[0].end(), z);
    }

  private:
    /// Approximately solve A_l x_l = b_l, with x_l pre-initialised.
    void run_cycle(int l) const {
        const Level& A = levels_[static_cast<std::size_t>(l)];
        double* x = x_[static_cast<std::size_t>(l)].data();
        const double* b = b_[static_cast<std::size_t>(l)].data();

        if (l + 1 == static_cast<int>(levels_.size())) {
            coarsest_solve(A, x, b);
            return;
        }

        for (int s = 0; s < opts_.pre_sweeps; ++s) {
            A.gauss_seidel_sweep(x, b, 0);
            A.gauss_seidel_sweep(x, b, 1);
        }

        // Restrict the residual and solve the coarse correction equation.
        const auto lc = static_cast<std::size_t>(l + 1);
        A.residual(x, b, r_[static_cast<std::size_t>(l)].data());
        A.restrict_to(r_[static_cast<std::size_t>(l)].data(), b_[lc].data());
        // `run_cycle` refines whatever is already in x_[lc], so repeating it
        // with the same right-hand side is exactly the W-cycle recursion.
        blas::fill(x_[lc], 0.0);
        for (int g = 0; g < opts_.cycle_index; ++g) run_cycle(l + 1);
        A.prolong_add(x_[lc].data(), x);

        // Mirror the pre-smoothing colour order so the whole cycle is symmetric.
        for (int s = 0; s < opts_.post_sweeps; ++s) {
            A.gauss_seidel_sweep(x, b, 1);
            A.gauss_seidel_sweep(x, b, 0);
        }
    }

    void coarsest_solve(const Level& A, double* x, const double* b) const {
        if (A.size() == 1) {
            x[0] = b[0] * A.inv_diag()[0];  // 1x1 system: exact.
            return;
        }
        // A handful of symmetric sweeps is exact to round-off on a grid this
        // small, and costs nothing next to the fine levels.
        for (int s = 0; s < kCoarsestSweeps; ++s) {
            A.gauss_seidel_sweep(x, b, 0);
            A.gauss_seidel_sweep(x, b, 1);
        }
        for (int s = 0; s < kCoarsestSweeps; ++s) {
            A.gauss_seidel_sweep(x, b, 1);
            A.gauss_seidel_sweep(x, b, 0);
        }
    }

    static constexpr int kCoarsestSweeps = 8;

    MultigridOptions opts_;
    std::vector<Level> levels_;
    mutable std::vector<std::vector<double>> x_, b_, r_;
};

}  // namespace darcy
