"""BlkDiagPrecondLO_tools"""

from typing import Any

from numpy.typing import NDArray

def BDPLO_mult_QU(
    new_npix: int,
    weighted_sin_sq: NDArray[Any],
    weighted_cos_sq: NDArray[Any],
    weighted_sincos: NDArray[Any],
    one_over_determinant: NDArray[Any],
    vec: NDArray[Any],
    prod: NDArray[Any],
) -> None: ...
def BDPLO_mult_IQU(
    new_npix: int,
    weighted_counts: NDArray[Any],
    weighted_sin_sq: NDArray[Any],
    weighted_cos_sq: NDArray[Any],
    weighted_sincos: NDArray[Any],
    weighted_sin: NDArray[Any],
    weighted_cos: NDArray[Any],
    one_over_determinant: NDArray[Any],
    vec: NDArray[Any],
    prod: NDArray[Any],
) -> None: ...
