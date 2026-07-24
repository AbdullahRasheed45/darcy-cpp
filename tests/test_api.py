"""Public API behaviour: input handling, validation, batching, threading."""

from __future__ import annotations

import numpy as np
import pytest

import darcy
from conftest import two_phase


class TestInputHandling:
    """Awkward-but-valid inputs should be converted, not rejected."""

    def test_accepts_float32(self) -> None:
        a = two_phase(32, seed=0)
        assert np.allclose(darcy.solve(a.astype(np.float32)).u, darcy.solve(a).u, rtol=1e-6)

    def test_accepts_integer_dtype(self) -> None:
        a = np.full((16, 16), 4, dtype=np.int32)
        result = darcy.solve(a)
        assert result.converged

    def test_accepts_fortran_order(self) -> None:
        a = two_phase(32, seed=1)
        assert np.allclose(darcy.solve(np.asfortranarray(a)).u, darcy.solve(a).u, atol=1e-12)

    def test_accepts_non_contiguous_view(self) -> None:
        a = two_phase(64, seed=2)[::2, ::2]
        assert darcy.solve(a).converged

    def test_accepts_nested_lists(self) -> None:
        assert darcy.solve([[1.0, 2.0], [2.0, 1.0]]).converged

    def test_does_not_mutate_its_input(self) -> None:
        a = two_phase(32, seed=3)
        original = a.copy()
        darcy.solve(a)
        assert np.array_equal(a, original)

    def test_output_is_a_fresh_writable_array(self) -> None:
        result = darcy.solve(two_phase(16, seed=4))
        assert result.u.flags.writeable and result.u.flags.c_contiguous
        result.u[0, 0] = 123.0  # must not corrupt anything internal
        assert darcy.solve(two_phase(16, seed=4)).u[0, 0] != 123.0

    def test_scalar_and_uniform_array_forcing_agree(self) -> None:
        a = two_phase(24, seed=5)
        u_scalar = darcy.solve(a, 2.5, tol=1e-12).u
        u_array = darcy.solve(a, np.full((24, 24), 2.5), tol=1e-12).u
        assert np.allclose(u_scalar, u_array, atol=1e-13)


class TestValidation:
    """Invalid input must fail loudly at the boundary, as ValueError."""

    def test_rejects_non_square(self) -> None:
        with pytest.raises(ValueError, match="square"):
            darcy.solve(np.ones((4, 8)))

    def test_rejects_wrong_dimensionality(self) -> None:
        with pytest.raises(ValueError):
            darcy.solve(np.ones((4, 4, 4)))

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            darcy.solve(np.ones((0, 0)))

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_permeability(self, bad: float) -> None:
        a = np.full((8, 8), 2.0)
        a[3, 4] = bad
        with pytest.raises(ValueError, match="positive"):
            darcy.solve(a)

    def test_rejects_mismatched_forcing_shape(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            darcy.solve(np.ones((8, 8)), np.ones((4, 4)))

    def test_rejects_unknown_preconditioner(self) -> None:
        with pytest.raises(ValueError, match="unknown preconditioner"):
            darcy.solve(np.ones((8, 8)), preconditioner="ilu")

    @pytest.mark.parametrize("kwargs", [{"tol": 0.0}, {"tol": -1e-3}, {"max_iter": 0}])
    def test_rejects_bad_solver_parameters(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            darcy.solve(np.ones((8, 8)), **kwargs)

    @pytest.mark.parametrize("kwargs", [{"pre_sweeps": 0}, {"post_sweeps": 0}, {"cycle_index": 3}])
    def test_rejects_bad_multigrid_parameters(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            darcy.solve(np.ones((8, 8)), **kwargs)


class TestNonConvergence:
    def test_reports_rather_than_raises(self) -> None:
        """Hitting max_iter returns converged=False; it is not an exception.

        Callers doing large sweeps need to inspect and continue, not handle an
        exception per sample.
        """
        result = darcy.solve(two_phase(128, seed=0), tol=1e-14, max_iter=2)
        assert not result.converged
        assert result.iterations == 2
        assert result.residual > 1e-14
        assert np.isfinite(result.u).all()


class TestBatch:
    def test_matches_serial_solves(self) -> None:
        batch_a = np.stack([two_phase(32, seed=s) for s in range(6)])
        batch = darcy.solve_batch(batch_a, tol=1e-12)
        assert batch.all_converged
        assert batch.u.shape == batch_a.shape
        for i, a in enumerate(batch_a):
            assert np.allclose(batch.u[i], darcy.solve(a, tol=1e-12).u, atol=1e-11)

    def test_is_deterministic_across_thread_counts(self) -> None:
        """Results must not depend on how the work was scheduled.

        Red-black Gauss-Seidel and static-scheduled reductions are what make
        this hold; a data race in the smoother would show up here.
        """
        batch_a = np.stack([two_phase(48, seed=s) for s in range(4)])
        single = darcy.solve_batch(batch_a, tol=1e-12, threads=1)
        multi = darcy.solve_batch(batch_a, tol=1e-12, threads=8)
        assert np.array_equal(single.iterations, multi.iterations)
        assert np.allclose(single.u, multi.u, atol=1e-13)

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match=r"batch, n, n"):
            darcy.solve_batch(np.ones((8, 8)))

    def test_propagates_validation_errors(self) -> None:
        batch_a = np.stack([np.full((8, 8), 2.0) for _ in range(3)])
        batch_a[1, 0, 0] = -1.0
        with pytest.raises(ValueError, match="positive"):
            darcy.solve_batch(batch_a)

    def test_single_sample_batch(self) -> None:
        a = two_phase(16, seed=1)
        batch = darcy.solve_batch(a[None], tol=1e-12)
        assert np.allclose(batch.u[0], darcy.solve(a, tol=1e-12).u, atol=1e-12)

    def test_reports_per_sample_diagnostics(self) -> None:
        batch = darcy.solve_batch(np.stack([two_phase(32, seed=s) for s in range(5)]), max_iter=2)
        assert not batch.all_converged
        assert batch.iterations.shape == (5,)
        assert batch.residuals.shape == (5,)
        assert batch.converged.dtype == bool


class TestIntrospection:
    def test_build_info_is_complete(self) -> None:
        info = darcy.build_info()
        assert set(info) == {"version", "compiler", "openmp", "max_threads", "numpy"}
        assert isinstance(info["openmp"], bool)
        assert info["max_threads"] >= 1

    def test_version_is_exposed(self) -> None:
        assert darcy.__version__.count(".") == 2

    def test_thread_count_round_trips(self) -> None:
        original = darcy.num_threads()
        try:
            darcy.set_num_threads(2)
            assert darcy.num_threads() == (2 if darcy.has_openmp() else 1)
        finally:
            darcy.set_num_threads(original)

    def test_result_repr_is_informative(self) -> None:
        text = repr(darcy.solve(two_phase(16, seed=0)))
        for token in ("SolveResult", "iterations", "residual", "converged"):
            assert token in text

    def test_history_is_empty_unless_requested(self) -> None:
        assert darcy.solve(two_phase(16, seed=0)).history.size == 0
        assert darcy.solve(two_phase(16, seed=0), record_history=True).history.size > 0


class TestFields:
    def test_gaussian_random_field_is_reproducible(self) -> None:
        assert np.array_equal(
            darcy.gaussian_random_field(32, seed=42), darcy.gaussian_random_field(32, seed=42)
        )

    def test_different_seeds_differ(self) -> None:
        assert not np.array_equal(
            darcy.gaussian_random_field(32, seed=1), darcy.gaussian_random_field(32, seed=2)
        )

    def test_accepts_a_generator(self) -> None:
        field = darcy.gaussian_random_field(16, seed=np.random.default_rng(0))
        assert field.shape == (16, 16)

    def test_field_is_mean_zero(self) -> None:
        assert abs(darcy.gaussian_random_field(128, seed=0).mean()) < 1e-10

    def test_higher_alpha_gives_smoother_fields(self) -> None:
        """Smoothness is measured by total variation of the sampled field."""
        rough = darcy.gaussian_random_field(64, alpha=1.5, seed=0)
        smooth = darcy.gaussian_random_field(64, alpha=4.0, seed=0)

        def roughness(field: np.ndarray) -> float:
            return float(np.abs(np.diff(field, axis=0)).sum() / np.abs(field).sum())

        assert roughness(smooth) < roughness(rough)

    def test_two_phase_takes_exactly_two_values(self) -> None:
        a = darcy.two_phase_permeability(darcy.gaussian_random_field(64, seed=0), hi=9.0, lo=2.0)
        assert set(np.unique(a)) == {2.0, 9.0}

    def test_rejects_invalid_parameters(self) -> None:
        with pytest.raises(ValueError):
            darcy.gaussian_random_field(0)
        with pytest.raises(ValueError, match="positive"):
            darcy.two_phase_permeability(np.zeros((8, 8)), hi=1.0, lo=-1.0)
