"""Type stubs for the compiled extension module."""

from typing import Any, List, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

__version__: str
__compiler__: str

class InvalidInput(ValueError): ...

class SolveResult:
    @property
    def u(self) -> NDArray[np.float64]: ...
    @property
    def iterations(self) -> int: ...
    @property
    def residual(self) -> float: ...
    @property
    def converged(self) -> bool: ...
    @property
    def levels(self) -> int: ...
    @property
    def history(self) -> List[float]: ...

def solve(
    a: ArrayLike,
    f: float | ArrayLike = ...,
    *,
    tol: float = ...,
    max_iter: int = ...,
    preconditioner: str = ...,
    pre_sweeps: int = ...,
    post_sweeps: int = ...,
    cycle_index: int = ...,
    record_history: bool = ...,
) -> SolveResult: ...
def solve_batch(
    a: ArrayLike,
    f: float | ArrayLike = ...,
    *,
    tol: float = ...,
    max_iter: int = ...,
    preconditioner: str = ...,
    pre_sweeps: int = ...,
    post_sweeps: int = ...,
    cycle_index: int = ...,
    threads: int = ...,
) -> Tuple[NDArray[np.float64], List[int], List[float], List[bool]]: ...
def apply_operator(a: ArrayLike, x: ArrayLike) -> NDArray[np.float64]: ...
def apply_preconditioner(
    a: ArrayLike,
    r: ArrayLike,
    *,
    pre_sweeps: int = ...,
    post_sweeps: int = ...,
    cycle_index: int = ...,
) -> NDArray[np.float64]: ...
def num_threads() -> int: ...
def set_num_threads(threads: int) -> None: ...
def has_openmp() -> bool: ...

_: Any
