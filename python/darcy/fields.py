"""Random permeability fields.

The sampling procedure matches the standard Darcy-flow benchmark from Li et
al., *Fourier Neural Operator for Parametric Partial Differential Equations*
(ICLR 2021): a Gaussian random field is drawn from
``N(0, (-Laplacian + tau^2 I)^(-alpha))`` and thresholded into a two-phase
medium. Reproducing that setup exactly is what makes this solver a drop-in
replacement for the SciPy data generator used to train neural operators.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["gaussian_random_field", "two_phase_permeability"]

SeedLike = Union[int, np.random.Generator, None]


def _as_generator(seed: SeedLike) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def gaussian_random_field(
    n: int,
    alpha: float = 2.0,
    tau: float = 3.0,
    seed: SeedLike = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Sample a mean-zero Gaussian random field on an ``n x n`` grid.

    The spectral density is ``tau^(alpha-1) (pi^2 (k1^2 + k2^2) + tau^2)^(-alpha/2)``.
    Larger ``alpha`` gives smoother fields; larger ``tau`` gives a shorter
    correlation length.

    Parameters
    ----------
    n:
        Grid size. Must be positive.
    alpha, tau:
        Covariance parameters (see above).
    seed:
        Integer seed or a ``numpy.random.Generator``. Passing an integer makes
        the sample reproducible.
    rng:
        Deprecated alias for ``seed``, kept so existing scripts keep working.
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    generator = _as_generator(rng if rng is not None else seed)

    k = np.arange(n)
    k1, k2 = np.meshgrid(k, k, indexing="ij")
    coef = (np.pi**2) * (k1**2 + k2**2) + tau**2
    sqrt_eig = (tau ** (alpha - 1.0)) * coef ** (-alpha / 2.0)
    sqrt_eig[0, 0] = 0.0  # drop the mean mode so the field is centred

    xi = generator.standard_normal((n, n))
    from scipy.fft import idctn  # imported lazily: SciPy is an optional dep

    return np.asarray(idctn(sqrt_eig * xi, norm="ortho"), dtype=np.float64)


def two_phase_permeability(
    grf: NDArray[np.float64], hi: float = 12.0, lo: float = 3.0
) -> NDArray[np.float64]:
    """Threshold a Gaussian random field into a two-phase permeability field.

    Values at or above zero become ``hi``, the rest ``lo``. The resulting jump
    discontinuity is exactly what makes harmonic face averaging necessary and
    naive discretisations inaccurate.
    """
    if hi <= 0 or lo <= 0:
        raise ValueError("permeability phases must be strictly positive")
    return np.where(np.asarray(grf) >= 0.0, hi, lo).astype(np.float64)
