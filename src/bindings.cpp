// SPDX-License-Identifier: MIT
//
// pybind11 bindings for the Darcy solver.
//
// Design notes:
//   * NumPy arrays are accepted with `forcecast`, so float32 / Fortran-order /
//     non-contiguous inputs work by converting once at the boundary rather
//     than failing on the user.
//   * The GIL is released around every solve. That is what makes `solve_batch`
//     scale and what lets callers overlap solves with Python-side work.
//   * C++ exceptions are translated to precise Python exception types, so
//     callers can catch `ValueError` rather than a generic `RuntimeError`.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <exception>
#include <string>
#include <vector>

#include "darcy/config.hpp"
#include "darcy/solver.hpp"

namespace py = pybind11;
using namespace darcy;

namespace {

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

/// Build the C++ options struct from the keyword arguments exposed to Python.
SolveOptions make_options(double tol, int max_iter, const std::string& preconditioner, int pre_sweeps,
                          int post_sweeps, int cycle_index, bool record_history) {
    SolveOptions opts;
    opts.tol = tol;
    opts.max_iter = max_iter;
    opts.preconditioner = preconditioner_from_string(preconditioner);
    opts.multigrid.pre_sweeps = pre_sweeps;
    opts.multigrid.post_sweeps = post_sweeps;
    opts.multigrid.cycle_index = cycle_index;
    opts.record_history = record_history;
    if (pre_sweeps < 1 || post_sweeps < 1)
        throw InvalidInput("pre_sweeps and post_sweeps must be at least 1");
    if (cycle_index < 1 || cycle_index > 2)
        throw InvalidInput("cycle_index must be 1 (V-cycle) or 2 (W-cycle)");
    return opts;
}

/// Turn the `f` argument -- a scalar or an n x n array -- into a dense buffer.
std::vector<double> make_rhs(const py::object& f, index_t n) {
    const auto size = static_cast<std::size_t>(n * n);
    if (py::isinstance<py::array>(f) || py::isinstance<py::list>(f) || py::isinstance<py::tuple>(f)) {
        Array arr = py::cast<Array>(f);
        auto buf = arr.request();
        if (buf.ndim != 2 || buf.shape[0] != n || buf.shape[1] != n) {
            throw InvalidInput("f must be a scalar or an array with the same shape as a");
        }
        const auto* ptr = static_cast<const double*>(buf.ptr);
        return std::vector<double>(ptr, ptr + size);
    }
    return std::vector<double>(size, py::cast<double>(f));
}

/// Copy a row-major buffer into a fresh NumPy (n, n) array.
py::array_t<double> to_numpy(const std::vector<double>& data, index_t n) {
    const std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(n)};
    py::array_t<double> out(shape);
    std::copy(data.begin(), data.end(), static_cast<double*>(out.request().ptr));
    return out;
}

index_t require_square_2d(const py::buffer_info& buf, const char* what) {
    if (buf.ndim != 2 || buf.shape[0] != buf.shape[1] || buf.shape[0] < 1) {
        throw InvalidInput(std::string(what) + " must be a square 2-D array with at least one cell");
    }
    return static_cast<index_t>(buf.shape[0]);
}

/// Python-facing view of SolveResult.
struct PyResult {
    py::array_t<double> u;
    int iterations;
    double residual;
    bool converged;
    int levels;
    std::vector<double> history;

    static PyResult from(const SolveResult& r) {
        return PyResult{to_numpy(r.u, r.n), r.iterations, r.residual, r.converged, r.levels, r.history};
    }
};

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() =
        "Matrix-free multigrid-preconditioned finite-volume solver for 2-D Darcy flow.\n"
        "Low-level extension module; prefer the `darcy` package API.";

    py::register_exception<InvalidInput>(m, "InvalidInput", PyExc_ValueError);

    py::class_<PyResult>(m, "SolveResult", "Outcome of a single Darcy solve.")
        .def_readonly("u", &PyResult::u, "Solution field, shape (n, n).")
        .def_readonly("iterations", &PyResult::iterations, "CG iterations performed.")
        .def_readonly("residual", &PyResult::residual, "Final relative residual.")
        .def_readonly("converged", &PyResult::converged, "True if the tolerance was reached.")
        .def_readonly("levels", &PyResult::levels, "Number of multigrid levels used.")
        .def_readonly("history", &PyResult::history, "Relative residual per iteration, if recorded.")
        .def("__repr__", [](const PyResult& r) {
            return "SolveResult(iterations=" + std::to_string(r.iterations) +
                   ", residual=" + std::to_string(r.residual) +
                   ", converged=" + (r.converged ? "True" : "False") +
                   ", levels=" + std::to_string(r.levels) + ")";
        });

    m.def(
        "solve",
        [](Array a, py::object f, double tol, int max_iter, const std::string& preconditioner, int pre_sweeps,
           int post_sweeps, int cycle_index, bool record_history) {
            auto buf = a.request();
            const index_t n = require_square_2d(buf, "a");
            const SolveOptions opts =
                make_options(tol, max_iter, preconditioner, pre_sweeps, post_sweeps, cycle_index, record_history);
            std::vector<double> rhs = make_rhs(f, n);
            const auto* a_ptr = static_cast<const double*>(buf.ptr);

            SolveResult result;
            {
                py::gil_scoped_release release;  // no Python objects touched below
                result = solve(a_ptr, rhs.data(), n, opts);
            }
            return PyResult::from(result);
        },
        py::arg("a"), py::arg("f") = 1.0, py::arg("tol") = 1e-10, py::arg("max_iter") = 10000,
        py::arg("preconditioner") = "multigrid", py::arg("pre_sweeps") = 1, py::arg("post_sweeps") = 1,
        py::arg("cycle_index") = 2, py::arg("record_history") = false,
        "Solve -div(a grad u) = f with homogeneous Dirichlet boundary conditions.");

    m.def(
        "solve_batch",
        [](Array a, py::object f, double tol, int max_iter, const std::string& preconditioner, int pre_sweeps,
           int post_sweeps, int cycle_index, int threads) {
            auto buf = a.request();
            if (buf.ndim != 3 || buf.shape[1] != buf.shape[2] || buf.shape[1] < 1)
                throw InvalidInput("a must have shape (batch, n, n)");
            const auto batch = static_cast<index_t>(buf.shape[0]);
            const auto n = static_cast<index_t>(buf.shape[1]);
            const index_t stride = n * n;
            const SolveOptions opts =
                make_options(tol, max_iter, preconditioner, pre_sweeps, post_sweeps, cycle_index, false);
            std::vector<double> rhs = make_rhs(f, n);
            const auto* a_ptr = static_cast<const double*>(buf.ptr);

            const std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(batch), static_cast<py::ssize_t>(n),
                                                 static_cast<py::ssize_t>(n)};
            py::array_t<double> out(shape);
            auto* out_ptr = static_cast<double*>(out.request().ptr);
            std::vector<int> iters(static_cast<std::size_t>(batch), 0);
            std::vector<double> residuals(static_cast<std::size_t>(batch), 0.0);
            std::vector<char> converged(static_cast<std::size_t>(batch), 0);

            // One exception slot: OpenMP forbids unwinding out of a parallel
            // region, so failures are captured and rethrown after the join.
            std::exception_ptr error;
            {
                py::gil_scoped_release release;
#if DARCY_HAS_OPENMP
                const int nthreads = threads > 0 ? threads : omp_get_max_threads();
#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
#else
                (void)threads;
#endif
                for (index_t s = 0; s < batch; ++s) {
                    try {
                        // Parallelism lives here, across independent samples,
                        // so each solve runs its own kernels serially.
                        const SolveResult r = solve(a_ptr + s * stride, rhs.data(), n, opts);
                        std::copy(r.u.begin(), r.u.end(), out_ptr + s * stride);
                        iters[static_cast<std::size_t>(s)] = r.iterations;
                        residuals[static_cast<std::size_t>(s)] = r.residual;
                        converged[static_cast<std::size_t>(s)] = r.converged ? 1 : 0;
                    } catch (...) {
#pragma omp critical
                        if (!error) error = std::current_exception();
                    }
                }
            }
            if (error) std::rethrow_exception(error);

            std::vector<bool> conv(converged.begin(), converged.end());
            return py::make_tuple(out, iters, residuals, conv);
        },
        py::arg("a"), py::arg("f") = 1.0, py::arg("tol") = 1e-10, py::arg("max_iter") = 10000,
        py::arg("preconditioner") = "multigrid", py::arg("pre_sweeps") = 1, py::arg("post_sweeps") = 1,
        py::arg("cycle_index") = 2, py::arg("threads") = 0,
        "Solve a batch of independent Darcy problems in parallel across samples.");

    m.def(
        "apply_operator",
        [](Array a, Array x) {
            auto abuf = a.request();
            const index_t n = require_square_2d(abuf, "a");
            auto xbuf = x.request();
            if (require_square_2d(xbuf, "x") != n) throw InvalidInput("x must have the same shape as a");
            validate_permeability(static_cast<const double*>(abuf.ptr), n * n);
            Level level(static_cast<const double*>(abuf.ptr), n);
            std::vector<double> y(static_cast<std::size_t>(n * n));
            {
                py::gil_scoped_release release;
                level.apply(static_cast<const double*>(xbuf.ptr), y.data());
            }
            return to_numpy(y, n);
        },
        py::arg("a"), py::arg("x"),
        "Apply the finite-volume operator A to x (matrix-free). Exposed for testing symmetry and definiteness.");

    m.def(
        "apply_preconditioner",
        [](Array a, Array r, int pre_sweeps, int post_sweeps, int cycle_index) {
            auto abuf = a.request();
            const index_t n = require_square_2d(abuf, "a");
            auto rbuf = r.request();
            if (require_square_2d(rbuf, "r") != n) throw InvalidInput("r must have the same shape as a");
            validate_permeability(static_cast<const double*>(abuf.ptr), n * n);

            MultigridOptions opts;
            opts.pre_sweeps = pre_sweeps;
            opts.post_sweeps = post_sweeps;
            opts.cycle_index = cycle_index;
            if (pre_sweeps < 1 || post_sweeps < 1) throw InvalidInput("sweeps must be at least 1");
            if (cycle_index < 1 || cycle_index > 2) throw InvalidInput("cycle_index must be 1 or 2");

            std::vector<double> z(static_cast<std::size_t>(n * n));
            {
                py::gil_scoped_release release;
                Multigrid mg(Level(static_cast<const double*>(abuf.ptr), n), opts);
                mg.apply(static_cast<const double*>(rbuf.ptr), z.data());
            }
            return to_numpy(z, n);
        },
        py::arg("a"), py::arg("r"), py::arg("pre_sweeps") = 1, py::arg("post_sweeps") = 1,
        py::arg("cycle_index") = 2,
        "Apply one multigrid cycle to r from a zero initial guess. Exposed so tests can verify "
        "directly that the preconditioner is symmetric positive definite, which CG requires.");

    m.def("num_threads", &max_threads, "Maximum number of OpenMP threads available.");
    m.def("has_openmp", &has_openmp, "Whether the extension was compiled with OpenMP support.");
    m.def(
        "set_num_threads",
        [](int t) {
#if DARCY_HAS_OPENMP
            if (t > 0) omp_set_num_threads(t);
#else
            (void)t;
#endif
        },
        py::arg("threads"), "Set the OpenMP thread count for subsequent solves.");

    m.attr("__version__") = DARCY_VERSION;
    m.attr("__compiler__") =
#if defined(__clang__)
        "clang " __clang_version__;
#elif defined(__GNUC__)
        "gcc " __VERSION__;
#elif defined(_MSC_VER)
        "msvc";
#else
        "unknown";
#endif
}
