import numpy as np


def computeweights_pol_I(
    npix,
    nsamples,
    pointings,
    pointings_flag,
    noise_weights,
    dtype_float,
):
    weighted_counts = np.zeros(npix, dtype=dtype_float)

    for idx in range(nsamples):
        pixel = pointings[idx]

        if pointings_flag[idx]:
            weighted_counts[pixel] += noise_weights[idx]

    pixel_mask = np.where(weighted_counts > 0)[0]

    new_npix = len(pixel_mask)

    return new_npix, weighted_counts, pixel_mask


def computeweights_pol_QU(
    npix,
    nsamples,
    pointings,
    pointings_flag,
    noise_weights,
    pol_angles,
    dtype_float,
):
    weighted_counts = np.zeros(npix, dtype=dtype_float)
    weighted_sin_sq = np.zeros(npix, dtype=dtype_float)
    weighted_cos_sq = np.zeros(npix, dtype=dtype_float)
    weighted_sincos = np.zeros(npix, dtype=dtype_float)

    sin2phi = np.sin(2.0 * pol_angles)
    cos2phi = np.cos(2.0 * pol_angles)

    for idx in range(nsamples):
        pixel = pointings[idx]

        if pointings_flag[idx]:
            weighted_counts[pixel] += noise_weights[idx]
            weighted_sin_sq[pixel] += noise_weights[idx] * sin2phi[idx] * sin2phi[idx]
            weighted_cos_sq[pixel] += noise_weights[idx] * cos2phi[idx] * cos2phi[idx]
            weighted_sincos[pixel] += noise_weights[idx] * sin2phi[idx] * cos2phi[idx]

    return (
        weighted_counts,
        sin2phi,
        cos2phi,
        weighted_sin_sq,
        weighted_cos_sq,
        weighted_sincos,
    )


def computeweights_pol_IQU(
    npix,
    nsamples,
    pointings,
    pointings_flag,
    noise_weights,
    pol_angles,
    dtype_float,
):
    weighted_counts = np.zeros(npix, dtype=dtype_float)
    weighted_sin_sq = np.zeros(npix, dtype=dtype_float)
    weighted_cos_sq = np.zeros(npix, dtype=dtype_float)
    weighted_sincos = np.zeros(npix, dtype=dtype_float)
    weighted_sin = np.zeros(npix, dtype=dtype_float)
    weighted_cos = np.zeros(npix, dtype=dtype_float)

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

    return (
        weighted_counts,
        sin2phi,
        cos2phi,
        weighted_sin_sq,
        weighted_cos_sq,
        weighted_sincos,
        weighted_sin,
        weighted_cos,
    )


def get_pix_mask_pol(
    solver_type,
    threshold,
    weighted_counts,
    weighted_sin_sq,
    weighted_cos_sq,
    weighted_sincos,
):
    determinant = (weighted_sin_sq * weighted_cos_sq) - (
        weighted_sincos * weighted_sincos
    )
    trace = weighted_sin_sq + weighted_cos_sq
    sqrtf = np.sqrt(trace * trace / 4.0 - determinant)
    lambda_max = trace / 2.0 + sqrtf
    lambda_min = trace / 2.0 - sqrtf
    cond_num = np.abs(lambda_max / lambda_min)
    cond_num_mask = np.where(cond_num <= threshold)[0]
    count_mask = np.where(weighted_counts > (solver_type - 1))[0]
    pixel_mask = np.intersect1d(count_mask, cond_num_mask)
    new_npix = len(pixel_mask)
    return new_npix, pixel_mask
