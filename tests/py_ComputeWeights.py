import numpy as np
from mpi4py import MPI


def computeweights_pol_I(
    npix: int,
    nsamples: int,
    pointings: np.ndarray,
    pointings_flag: np.ndarray,
    noise_weights: np.ndarray,
    dtype_float,
    comm,
):
    weighted_counts = np.zeros(npix, dtype=dtype_float)

    for idx in range(nsamples):
        pixel = pointings[idx]

        if pointings_flag[idx]:
            weighted_counts[pixel] += noise_weights[idx]

    comm.Allreduce(MPI.IN_PLACE, weighted_counts, MPI.SUM)

    observed_pixels = np.where(weighted_counts > 0)[0]

    new_npix = len(observed_pixels)

    observed_pixels = observed_pixels.astype(dtype=pointings.dtype)
    old2new_pixel = np.zeros(npix, dtype=pointings.dtype)
    pixel_flag = np.zeros(npix, dtype=bool)

    for idx in range(npix):
        if idx in observed_pixels:
            new_idx = np.where(observed_pixels == idx)[0]
            old2new_pixel[idx] = new_idx[0]
            pixel_flag[idx] = True

    return new_npix, weighted_counts, observed_pixels, old2new_pixel, pixel_flag


def computeweights_pol_QU(
    npix: int,
    nsamples: int,
    pointings: np.ndarray,
    pointings_flag: np.ndarray,
    noise_weights: np.ndarray,
    pol_angles: np.ndarray,
    dtype_float,
    comm,
):
    weighted_counts = np.zeros(npix, dtype=dtype_float)
    weighted_sin_sq = np.zeros(npix, dtype=dtype_float)
    weighted_cos_sq = np.zeros(npix, dtype=dtype_float)
    weighted_sincos = np.zeros(npix, dtype=dtype_float)
    one_over_determinant = np.zeros(npix, dtype=dtype_float)

    sin2phi = np.sin(2.0 * pol_angles)
    cos2phi = np.cos(2.0 * pol_angles)

    for idx in range(nsamples):
        pixel = pointings[idx]

        if pointings_flag[idx]:
            weighted_counts[pixel] += noise_weights[idx]
            weighted_sin_sq[pixel] += noise_weights[idx] * sin2phi[idx] * sin2phi[idx]
            weighted_cos_sq[pixel] += noise_weights[idx] * cos2phi[idx] * cos2phi[idx]
            weighted_sincos[pixel] += noise_weights[idx] * sin2phi[idx] * cos2phi[idx]

    comm.Allreduce(MPI.IN_PLACE, weighted_counts, MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, weighted_sin_sq, MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, weighted_cos_sq, MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, weighted_sincos, MPI.SUM)

    one_over_determinant = (weighted_cos_sq * weighted_sin_sq) - (
        weighted_sincos * weighted_sincos
    )

    return (
        weighted_counts,
        sin2phi,
        cos2phi,
        weighted_sin_sq,
        weighted_cos_sq,
        weighted_sincos,
        one_over_determinant,
    )


def computeweights_pol_IQU(
    npix: int,
    nsamples: int,
    pointings: np.ndarray,
    pointings_flag: np.ndarray,
    noise_weights: np.ndarray,
    pol_angles: np.ndarray,
    dtype_float,
    comm,
):
    weighted_counts = np.zeros(npix, dtype=dtype_float)
    weighted_sin_sq = np.zeros(npix, dtype=dtype_float)
    weighted_cos_sq = np.zeros(npix, dtype=dtype_float)
    weighted_sincos = np.zeros(npix, dtype=dtype_float)
    weighted_sin = np.zeros(npix, dtype=dtype_float)
    weighted_cos = np.zeros(npix, dtype=dtype_float)
    one_over_determinant = np.zeros(npix, dtype=dtype_float)

    sin2phi = np.sin(2.0 * pol_angles)
    cos2phi = np.cos(2.0 * pol_angles)

    for idx in range(nsamples):
        pixel = pointings[idx]

        if pointings_flag[idx]:
            weighted_counts[pixel] += noise_weights[idx]
            weighted_sin_sq[pixel] += noise_weights[idx] * sin2phi[idx] * sin2phi[idx]
            weighted_cos_sq[pixel] += noise_weights[idx] * cos2phi[idx] * cos2phi[idx]
            weighted_sincos[pixel] += noise_weights[idx] * sin2phi[idx] * cos2phi[idx]
            weighted_sin[pixel] += noise_weights[idx] * sin2phi[idx]
            weighted_cos[pixel] += noise_weights[idx] * cos2phi[idx]

    comm.Allreduce(MPI.IN_PLACE, weighted_counts, MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, weighted_sin, MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, weighted_cos, MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, weighted_sin_sq, MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, weighted_cos_sq, MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, weighted_sincos, MPI.SUM)

    one_over_determinant = (
        weighted_counts
        * (weighted_cos_sq * weighted_sin_sq - weighted_sincos * weighted_sincos)
        - weighted_cos * weighted_cos * weighted_sin_sq
        - weighted_sin * weighted_sin * weighted_cos_sq
        + 2.0 * weighted_cos * weighted_sin * weighted_sincos
    )

    return (
        weighted_counts,
        sin2phi,
        cos2phi,
        weighted_sin_sq,
        weighted_cos_sq,
        weighted_sincos,
        weighted_sin,
        weighted_cos,
        one_over_determinant,
    )


def get_pix_mask_pol(
    npix: int,
    solver_type: int,
    threshold: float,
    weighted_counts: np.ndarray,
    one_over_determinant: np.ndarray,
    dtype_int,
):
    determinant_mask = np.where(one_over_determinant > threshold)[0]
    count_mask = np.where(weighted_counts > (solver_type - 1))[0]
    observed_pixels = np.intersect1d(count_mask, determinant_mask)
    new_npix = len(observed_pixels)

    observed_pixels = observed_pixels.astype(dtype=dtype_int)
    old2new_pixel = np.zeros(npix, dtype=dtype_int)
    pixel_flag = np.zeros(npix, dtype=bool)

    for idx in range(npix):
        if idx in observed_pixels:
            new_idx = np.where(observed_pixels == idx)[0]
            old2new_pixel[idx] = new_idx[0]
            pixel_flag[idx] = True

    return new_npix, observed_pixels, old2new_pixel, pixel_flag
