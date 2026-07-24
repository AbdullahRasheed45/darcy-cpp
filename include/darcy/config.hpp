// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>

#ifdef _OPENMP
#include <omp.h>
#define DARCY_HAS_OPENMP 1
#else
#define DARCY_HAS_OPENMP 0
#endif

#if defined(__GNUC__) || defined(__clang__)
#define DARCY_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define DARCY_RESTRICT __restrict
#else
#define DARCY_RESTRICT
#endif

// Portable OpenMP pragmas.
//
// MSVC implements only OpenMP 2.0, which has neither the `simd` clause nor the
// `parallel for simd` combined directive (both OpenMP 4.0). GCC and Clang do,
// and there the `simd` clause is what authorises vectorising a floating-point
// reduction that is otherwise non-associative. So we emit `simd` on the
// compilers that understand it and drop it on MSVC, via a token that expands to
// nothing there. `DARCY_OMP(...)` wraps the whole directive so the clauses
// (which contain commas) survive as a single `_Pragma` string; the two-level
// indirection is what lets `DARCY_SIMD` expand before stringisation.
#if DARCY_HAS_OPENMP
#define DARCY_DO_PRAGMA(x) _Pragma(#x)
#define DARCY_OMP(x) DARCY_DO_PRAGMA(omp x)
#if defined(_MSC_VER)
#define DARCY_SIMD
#else
#define DARCY_SIMD simd
#endif
#else
#define DARCY_OMP(x)
#define DARCY_SIMD
#endif

namespace darcy {

/// Signed index type used for all grid and vector loops.
///
/// Signed because OpenMP 2.0 (the level MSVC implements) only accepts signed
/// loop variables; 64-bit because an n = 100000 grid overflows `int` in the
/// flattened index.
using index_t = std::ptrdiff_t;

/// Number of OpenMP threads the library will use for a parallel region.
inline int max_threads() noexcept {
#if DARCY_HAS_OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

inline bool has_openmp() noexcept { return DARCY_HAS_OPENMP != 0; }

}  // namespace darcy
