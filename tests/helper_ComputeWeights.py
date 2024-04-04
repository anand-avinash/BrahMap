import numpy as np


def computeweights_pol_I(
    npix: int,
    nsamples: int,
    pointings: np.ndarray,
    pointings_flag: np.ndarray,
    noise_weights: np.ndarray,
    dtype_float,
):
    weighted_counts = np.zeros(npix, dtype=dtype_float)

    for idx in range(nsamples):
        pixel = pointings[idx]

        if pointings_flag[idx]:
            weighted_counts[pixel] += noise_weights[idx]

    pixel_mask = np.where(weighted_counts > 0)[0]

    new_npix = len(pixel_mask)

    pixel_mask = pixel_mask.astype(dtype=pointings.dtype)
    old2new_pixel = np.zeros(npix, dtype=pointings.dtype)
    pixel_flag = np.zeros(npix, dtype=bool)

    for idx in range(npix):
        if idx in pixel_mask:
            new_idx = np.where(pixel_mask == idx)[0]
            old2new_pixel[idx] = new_idx
            pixel_flag[idx] = True

    return new_npix, weighted_counts, pixel_mask, old2new_pixel, pixel_flag


def computeweights_pol_QU(
    npix: int,
    nsamples: int,
    pointings: np.ndarray,
    pointings_flag: np.ndarray,
    noise_weights: np.ndarray,
    pol_angles: np.ndarray,
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
    npix: int,
    nsamples: int,
    pointings: np.ndarray,
    pointings_flag: np.ndarray,
    noise_weights: np.ndarray,
    pol_angles: np.ndarray,
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
    npix: int,
    solver_type: int,
    threshold: float,
    weighted_counts: np.ndarray,
    weighted_sin_sq: np.ndarray,
    weighted_cos_sq: np.ndarray,
    weighted_sincos: np.ndarray,
    dtype_int,
):
    determinant = (weighted_sin_sq * weighted_cos_sq) - (
        weighted_sincos * weighted_sincos
    )
    trace = weighted_sin_sq + weighted_cos_sq
    sqrtf = np.sqrt(trace * trace / 4.0 - determinant)
    lambda_max = trace / 2.0 + sqrtf
    lambda_min = trace / 2.0 - sqrtf
    # cond_num = np.abs(lambda_max / lambda_min)
    # ^ This division produced `divided by zero warning` in multiple cases. So replaced it with `np.divide`. see <https://stackoverflow.com/a/54364060>
    cond_num = 1.0e-7 * np.ones(len(lambda_max))
    np.divide(lambda_max, lambda_min, out=cond_num, where=lambda_min != 0)
    cond_num = np.abs(cond_num)

    cond_num_mask = np.where(cond_num <= threshold)[0]
    count_mask = np.where(weighted_counts > (solver_type - 1))[0]
    pixel_mask = np.intersect1d(count_mask, cond_num_mask)
    new_npix = len(pixel_mask)

    pixel_mask = pixel_mask.astype(dtype=dtype_int)
    old2new_pixel = np.zeros(npix, dtype=dtype_int)
    pixel_flag = np.zeros(npix, dtype=bool)

    for idx in range(npix):
        if idx in pixel_mask:
            new_idx = np.where(pixel_mask == idx)[0]
            old2new_pixel[idx] = new_idx
            pixel_flag[idx] = True

    return new_npix, pixel_mask, old2new_pixel, pixel_flag
