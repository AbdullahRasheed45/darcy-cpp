// SPDX-License-Identifier: MIT
//
// Minimal parallel BLAS-1 kernels. These are the only vector operations the
// Krylov solver needs, and keeping them here (rather than pulling in a BLAS
// dependency) keeps the build free of external libraries.
//
// All kernels are memory-bandwidth-bound; the OpenMP `static` schedule gives
// each thread a contiguous slab, which is what we want for first-touch NUMA
// locality and for streaming prefetchers.
#pragma once

#include <cmath>
#include <vector>

#include "darcy/config.hpp"

namespace darcy::blas {

using vec = std::vector<double>;

/// Below this length a kernel runs serially: forking a thread team costs more
/// than the work itself. Multigrid calls these on coarse levels of a handful
/// of cells, so the guard is not hypothetical.
inline constexpr index_t kMinParallelSize = 4096;

/// s = x . y
inline double dot(const vec& x, const vec& y) noexcept {
    const double* DARCY_RESTRICT px = x.data();
    const double* DARCY_RESTRICT py = y.data();
    const index_t n = static_cast<index_t>(x.size());
    double s = 0.0;
#pragma omp parallel for simd reduction(+ : s) schedule(static) if (n >= kMinParallelSize)
    for (index_t i = 0; i < n; ++i) s += px[i] * py[i];
    return s;
}

/// ||x||_2
inline double norm(const vec& x) noexcept { return std::sqrt(dot(x, x)); }

/// y += alpha * x
inline void axpy(double alpha, const vec& x, vec& y) noexcept {
    const double* DARCY_RESTRICT px = x.data();
    double* DARCY_RESTRICT py = y.data();
    const index_t n = static_cast<index_t>(x.size());
#pragma omp parallel for simd schedule(static) if (n >= kMinParallelSize)
    for (index_t i = 0; i < n; ++i) py[i] += alpha * px[i];
}

/// y = x + beta * y  (the CG search-direction update)
inline void xpby(const vec& x, double beta, vec& y) noexcept {
    const double* DARCY_RESTRICT px = x.data();
    double* DARCY_RESTRICT py = y.data();
    const index_t n = static_cast<index_t>(x.size());
#pragma omp parallel for simd schedule(static) if (n >= kMinParallelSize)
    for (index_t i = 0; i < n; ++i) py[i] = px[i] + beta * py[i];
}

/// Fused CG update: u += alpha*p, r -= alpha*Ap, and return ||r||^2.
///
/// Fusing the two axpys with the residual reduction turns three passes over
/// memory into one, which measurably helps on a bandwidth-bound solver.
inline double cg_update(double alpha, const vec& p, const vec& ap, vec& u, vec& r) noexcept {
    const double* DARCY_RESTRICT pp = p.data();
    const double* DARCY_RESTRICT pap = ap.data();
    double* DARCY_RESTRICT pu = u.data();
    double* DARCY_RESTRICT pr = r.data();
    const index_t n = static_cast<index_t>(u.size());
    double rr = 0.0;
#pragma omp parallel for simd reduction(+ : rr) schedule(static) if (n >= kMinParallelSize)
    for (index_t i = 0; i < n; ++i) {
        pu[i] += alpha * pp[i];
        const double ri = pr[i] - alpha * pap[i];
        pr[i] = ri;
        rr += ri * ri;
    }
    return rr;
}

inline void fill(vec& x, double value) noexcept {
    double* DARCY_RESTRICT px = x.data();
    const index_t n = static_cast<index_t>(x.size());
#pragma omp parallel for simd schedule(static) if (n >= kMinParallelSize)
    for (index_t i = 0; i < n; ++i) px[i] = value;
}

}  // namespace darcy::blas
