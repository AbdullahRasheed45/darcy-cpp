// SPDX-License-Identifier: MIT
//
// Native tests for the numerical core.
//
// These assert the algebraic identities the multigrid hierarchy is built on --
// most importantly that the cheap coefficient-summing coarsening really equals
// the Galerkin product P^T A P. That is the load-bearing claim of the design,
// and checking it here (against a dense reference built from the operator
// itself) is far more direct than inferring it from convergence rates.
//
// Deliberately dependency-free: no test framework to install, so `ctest` works
// from a bare checkout.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "darcy/level.hpp"
#include "darcy/multigrid.hpp"
#include "darcy/solver.hpp"

namespace {

int failures = 0;
int checks = 0;

void check(bool condition, const std::string& what) {
    ++checks;
    if (!condition) {
        ++failures;
        std::printf("  FAIL  %s\n", what.c_str());
    }
}

void check_close(double got, double want, double tol, const std::string& what) {
    ++checks;
    const double scale = std::max({std::abs(got), std::abs(want), 1.0});
    if (!(std::abs(got - want) <= tol * scale)) {
        ++failures;
        std::printf("  FAIL  %s: got %.17g, want %.17g\n", what.c_str(), got, want);
    }
}

std::vector<double> random_permeability(darcy::index_t n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> phase(0.0, 1.0);
    std::vector<double> a(static_cast<std::size_t>(n * n));
    for (auto& value : a) value = phase(gen) < 0.5 ? 3.0 : 12.0;
    return a;
}

std::vector<double> random_vector(darcy::index_t size, unsigned seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<double> x(static_cast<std::size_t>(size));
    for (auto& value : x) value = normal(gen);
    return x;
}

/// Materialise a Level as a dense matrix by applying it to unit vectors.
std::vector<double> dense(const darcy::Level& level) {
    const auto m = static_cast<std::size_t>(level.size());
    std::vector<double> matrix(m * m, 0.0), unit(m, 0.0), column(m, 0.0);
    for (std::size_t j = 0; j < m; ++j) {
        unit[j] = 1.0;
        level.apply(unit.data(), column.data());
        for (std::size_t i = 0; i < m; ++i) matrix[i * m + j] = column[i];
        unit[j] = 0.0;
    }
    return matrix;
}

// -------------------------------------------------------------------------

void test_operator_is_symmetric() {
    const darcy::index_t n = 12;
    const auto a = random_permeability(n, 1);
    const darcy::Level level(a.data(), n);
    const auto x = random_vector(n * n, 2);
    const auto y = random_vector(n * n, 3);
    std::vector<double> ax(x.size()), ay(y.size());
    level.apply(x.data(), ax.data());
    level.apply(y.data(), ay.data());

    double left = 0.0, right = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        left += ax[i] * y[i];
        right += x[i] * ay[i];
    }
    check_close(left, right, 1e-12, "operator symmetry <Ax,y> == <x,Ay>");
}

void test_operator_is_positive_definite() {
    const darcy::index_t n = 12;
    const auto a = random_permeability(n, 4);
    const darcy::Level level(a.data(), n);
    for (unsigned seed = 0; seed < 20; ++seed) {
        const auto x = random_vector(n * n, 100 + seed);
        std::vector<double> ax(x.size());
        level.apply(x.data(), ax.data());
        double quadratic = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) quadratic += x[i] * ax[i];
        check(quadratic > 0.0, "operator positive definiteness <Ax,x> > 0");
    }
}

void test_restriction_is_the_transpose_of_prolongation() {
    // <R r, x_c> must equal <r, P x_c> exactly, or the coarse-grid correction
    // is not a Galerkin projection and the cycle stops being symmetric.
    for (darcy::index_t n : {8, 9, 16, 17}) {
        const auto a = random_permeability(n, 5);
        const darcy::Level level(a.data(), n);
        const darcy::index_t nc = level.coarse_n();

        const auto fine = random_vector(n * n, 6);
        const auto coarse = random_vector(nc * nc, 7);
        std::vector<double> restricted(static_cast<std::size_t>(nc * nc));
        std::vector<double> prolonged(static_cast<std::size_t>(n * n), 0.0);
        level.restrict_to(fine.data(), restricted.data());
        level.prolong_add(coarse.data(), prolonged.data());

        double left = 0.0, right = 0.0;
        for (std::size_t i = 0; i < restricted.size(); ++i) left += restricted[i] * coarse[i];
        for (std::size_t i = 0; i < fine.size(); ++i) right += fine[i] * prolonged[i];
        check_close(left, right, 1e-13, "R == P^T at n=" + std::to_string(n));
    }
}

void test_coarsening_equals_the_galerkin_product() {
    // The headline structural claim: summing face coefficients reproduces
    // P^T A P exactly, entry for entry -- including on odd grids, where the
    // trailing aggregate has width one.
    for (darcy::index_t n : {6, 7, 8, 13}) {
        const auto a = random_permeability(n, 8);
        const darcy::Level fine(a.data(), n);
        const darcy::Level coarse = fine.coarsen();
        const darcy::index_t nc = fine.coarse_n();
        const auto mc = static_cast<std::size_t>(nc * nc);

        const auto fine_dense = dense(fine);
        const auto coarse_dense = dense(coarse);
        const auto m = static_cast<std::size_t>(n * n);

        // (P^T A P)_{IJ} = sum over fine cells in aggregate I and J.
        const auto aggregate_of = [n, nc](std::size_t flat) {
            const auto i = static_cast<darcy::index_t>(flat) / n;
            const auto j = static_cast<darcy::index_t>(flat) % n;
            return static_cast<std::size_t>((i / 2) * nc + (j / 2));
        };
        std::vector<double> galerkin(mc * mc, 0.0);
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < m; ++j)
                galerkin[aggregate_of(i) * mc + aggregate_of(j)] += fine_dense[i * m + j];

        double worst = 0.0;
        for (std::size_t i = 0; i < mc * mc; ++i)
            worst = std::max(worst, std::abs(galerkin[i] - coarse_dense[i]));
        check_close(worst, 0.0, 1e-12, "coarsen() == P^T A P at n=" + std::to_string(n));
    }
}

void test_hierarchy_reaches_a_single_cell() {
    for (darcy::index_t n : {1, 2, 5, 16, 33, 64}) {
        const auto a = random_permeability(n, 9);
        darcy::Multigrid mg(darcy::Level(a.data(), n), darcy::MultigridOptions{});
        check(mg.level(mg.num_levels() - 1).n() == 1,
              "hierarchy coarsens to 1x1 from n=" + std::to_string(n));
    }
}

void test_preconditioner_is_symmetric() {
    const darcy::index_t n = 24;
    const auto a = random_permeability(n, 10);
    for (int cycle : {1, 2}) {
        darcy::MultigridOptions opts;
        opts.cycle_index = cycle;
        const darcy::Multigrid mg(darcy::Level(a.data(), n), opts);

        const auto r = random_vector(n * n, 11);
        const auto s = random_vector(n * n, 12);
        std::vector<double> mr(r.size()), ms(s.size());
        mg.apply(r.data(), mr.data());
        mg.apply(s.data(), ms.data());

        double left = 0.0, right = 0.0;
        for (std::size_t i = 0; i < r.size(); ++i) {
            left += mr[i] * s[i];
            right += ms[i] * r[i];
        }
        check_close(left, right, 1e-10, "preconditioner symmetry, cycle=" + std::to_string(cycle));
    }
}

void test_solver_converges_and_residual_is_honest() {
    const darcy::index_t n = 48;
    const auto a = random_permeability(n, 13);
    darcy::SolveOptions opts;
    opts.tol = 1e-11;
    const darcy::SolveResult result = darcy::solve_constant_rhs(a.data(), n, 1.0, opts);
    check(result.converged, "solver converges");

    // Recompute the residual from scratch rather than trusting CG's recursion.
    const darcy::Level level(a.data(), n);
    const double h2 = 1.0 / static_cast<double>((n + 1) * (n + 1));
    std::vector<double> au(result.u.size());
    level.apply(result.u.data(), au.data());
    double residual_norm = 0.0, rhs_norm = 0.0;
    for (std::size_t i = 0; i < au.size(); ++i) {
        const double d = h2 - au[i];
        residual_norm += d * d;
        rhs_norm += h2 * h2;
    }
    const double relative = std::sqrt(residual_norm) / std::sqrt(rhs_norm);
    check(relative < 10 * opts.tol, "recomputed residual matches the reported one");
}

void test_invalid_input_throws() {
    const std::vector<double> bad{1.0, 2.0, -3.0, 4.0};
    bool threw = false;
    try {
        darcy::SolveOptions opts;
        darcy::solve_constant_rhs(bad.data(), 2, 1.0, opts);
    } catch (const darcy::InvalidInput&) {
        threw = true;
    }
    check(threw, "non-positive permeability throws InvalidInput");
}

}  // namespace

int main() {
    std::printf("darcy core tests (OpenMP: %s, threads: %d)\n", darcy::has_openmp() ? "yes" : "no",
                darcy::max_threads());
    test_operator_is_symmetric();
    test_operator_is_positive_definite();
    test_restriction_is_the_transpose_of_prolongation();
    test_coarsening_equals_the_galerkin_product();
    test_hierarchy_reaches_a_single_cell();
    test_preconditioner_is_symmetric();
    test_solver_converges_and_residual_is_honest();
    test_invalid_input_throws();

    std::printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
