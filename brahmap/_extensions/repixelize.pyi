"""repixelize"""

from typing import Any

import numpy
from numpy.typing import NDArray

def repixelize_pol_I(
    new_npix: int,
    observed_pixels: NDArray[Any],
    hit_counts: NDArray[Any],
    weighted_counts: NDArray[Any],
) -> None: ...
def repixelize_pol_QU(
    new_npix: int,
    observed_pixels: NDArray[Any],
    hit_counts: NDArray[Any],
    weighted_counts: NDArray[Any],
    weighted_sin_sq: NDArray[Any],
    weighted_cos_sq: NDArray[Any],
    weighted_sincos: NDArray[Any],
    one_over_determinant: NDArray[Any],
) -> None: ...
def repixelize_pol_IQU(
    new_npix: int,
    observed_pixels: NDArray[Any],
    hit_counts: NDArray[Any],
    weighted_counts: NDArray[Any],
    weighted_sin_sq: NDArray[Any],
    weighted_cos_sq: NDArray[Any],
    weighted_sincos: NDArray[Any],
    weighted_sin: NDArray[Any],
    weighted_cos: NDArray[Any],
    one_over_determinant: NDArray[Any],
) -> None: ...
def flag_bad_pixel_samples(
    nsamples: int,
    pixel_flag: NDArray[numpy.bool_],
    old2new_pixel: NDArray[Any],
    pointings: NDArray[Any],
    pointings_flag: NDArray[numpy.bool_] | None,
) -> None: ...
