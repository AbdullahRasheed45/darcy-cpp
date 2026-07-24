"""Shared fixtures and helpers for the test suite."""

from __future__ import annotations

import numpy as np
import pytest

import darcy


def two_phase(n: int, seed: int = 0, hi: float = 12.0, lo: float = 3.0) -> np.ndarray:
    """A thresholded-GRF permeability field, the benchmark's standard medium."""
    return darcy.two_phase_permeability(darcy.gaussian_random_field(n, seed=seed), hi=hi, lo=lo)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20240724)


@pytest.fixture(params=["multigrid", "jacobi", "none"])
def preconditioner(request: pytest.FixtureRequest) -> str:
    """Every correctness property must hold for every preconditioner.

    They solve the same linear system by different means, so any disagreement
    is a bug in the preconditioner rather than a property of the discretisation.
    """
    return str(request.param)
