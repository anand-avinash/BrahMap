"""compute_weights"""

from typing import Any

import numpy
from numpy.typing import NDArray

def compute_weights_pol_I(
    npix: int,
    nsamples: int,
    pointings: NDArray[Any],
    pointings_flag: NDArray[numpy.bool_] | None,
    noise_weights: NDArray[Any],
    hit_counts: NDArray[Any],
    weighted_counts: NDArray[Any],
    observed_pixels: NDArray[Any],
    __old2new_pixel: NDArray[Any],  # type: ignore
    pixel_flag: NDArray[numpy.bool_],
    comm: object,
) -> int: ...
def compute_weights_pol_QU(
    npix: int,
    nsamples: int,
    pointings: NDArray[Any],
    pointings_flag: NDArray[numpy.bool_] | None,
    noise_weights: NDArray[Any],
    pol_angles: NDArray[Any],
    hit_counts: NDArray[Any],
    weighted_counts: NDArray[Any],
    sin2phi: NDArray[Any],
    cos2phi: NDArray[Any],
    weighted_sin_sq: NDArray[Any],
    weighted_cos_sq: NDArray[Any],
    weighted_sincos: NDArray[Any],
    one_over_determinant: NDArray[Any],
    comm: object,
) -> None: ...
def compute_weights_pol_IQU(
    npix: int,
    nsamples: int,
    pointings: NDArray[Any],
    pointings_flag: NDArray[numpy.bool_] | None,
    noise_weights: NDArray[Any],
    pol_angles: NDArray[Any],
    hit_counts: NDArray[Any],
    weighted_counts: NDArray[Any],
    sin2phi: NDArray[Any],
    cos2phi: NDArray[Any],
    weighted_sin_sq: NDArray[Any],
    weighted_cos_sq: NDArray[Any],
    weighted_sincos: NDArray[Any],
    weighted_sin: NDArray[Any],
    weighted_cos: NDArray[Any],
    one_over_determinant: NDArray[Any],
    comm: object,
) -> None: ...
def get_pixel_mask_pol(
    solver_type: int,
    npix: int,
    threshold: float,
    hit_counts: NDArray[Any],
    one_over_determinant: NDArray[Any],
    observed_pixels: NDArray[Any],
    __old2new_pixel: NDArray[Any],  # type: ignore
    pixel_flag: NDArray[numpy.bool_],
) -> int: ...
