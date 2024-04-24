import numpy as np


def repixelize_pol_I(
    new_npix: int, pixel_mask: np.ndarray, weighted_counts: np.ndarray
):
    for idx in range(new_npix):
        pixel = pixel_mask[idx]
        weighted_counts[idx] = weighted_counts[pixel]

    weighted_counts.resize(new_npix, refcheck=False)

    return weighted_counts


def repixelize_pol_QU(
    new_npix: int,
    pixel_mask: np.ndarray,
    weighted_counts: np.ndarray,
    weighted_sin_sq: np.ndarray,
    weighted_cos_sq: np.ndarray,
    weighted_sincos: np.ndarray,
    one_over_determinant: np.ndarray,
):
    for idx in range(new_npix):
        pixel = pixel_mask[idx]
        weighted_counts[idx] = weighted_counts[pixel]
        weighted_sin_sq[idx] = weighted_sin_sq[pixel]
        weighted_cos_sq[idx] = weighted_cos_sq[pixel]
        weighted_sincos[idx] = weighted_sincos[pixel]
        one_over_determinant[idx] = 1.0 / one_over_determinant[pixel]

    weighted_counts.resize(new_npix, refcheck=False)
    weighted_sin_sq.resize(new_npix, refcheck=False)
    weighted_cos_sq.resize(new_npix, refcheck=False)
    weighted_sincos.resize(new_npix, refcheck=False)
    one_over_determinant.resize(new_npix, refcheck=False)

    return (
        weighted_counts,
        weighted_sin_sq,
        weighted_cos_sq,
        weighted_sincos,
        one_over_determinant,
    )


def repixelize_pol_IQU(
    new_npix: int,
    pixel_mask: np.ndarray,
    weighted_counts: np.ndarray,
    weighted_sin_sq: np.ndarray,
    weighted_cos_sq: np.ndarray,
    weighted_sincos: np.ndarray,
    weighted_sin: np.ndarray,
    weighted_cos: np.ndarray,
    one_over_determinant: np.ndarray,
):
    for idx in range(new_npix):
        pixel = pixel_mask[idx]
        weighted_counts[idx] = weighted_counts[pixel]
        weighted_sin_sq[idx] = weighted_sin_sq[pixel]
        weighted_cos_sq[idx] = weighted_cos_sq[pixel]
        weighted_sincos[idx] = weighted_sincos[pixel]
        weighted_sin[idx] = weighted_sin[pixel]
        weighted_cos[idx] = weighted_cos[pixel]
        one_over_determinant[idx] = 1.0 / one_over_determinant[pixel]

    weighted_counts.resize(new_npix, refcheck=False)
    weighted_sin_sq.resize(new_npix, refcheck=False)
    weighted_cos_sq.resize(new_npix, refcheck=False)
    weighted_sincos.resize(new_npix, refcheck=False)
    weighted_sin.resize(new_npix, refcheck=False)
    weighted_cos.resize(new_npix, refcheck=False)
    one_over_determinant.resize(new_npix, refcheck=False)

    return (
        weighted_counts,
        weighted_sin_sq,
        weighted_cos_sq,
        weighted_sincos,
        weighted_sin,
        weighted_cos,
        one_over_determinant,
    )


def flag_bad_pixel_samples(
    nsamples: int,
    pixel_flag: np.ndarray,
    old2new_pixel: np.ndarray,
    pointings: np.ndarray,
    pointings_flag: np.ndarray,
):
    for idx in range(nsamples):
        pixel = pointings[idx]
        if pixel_flag[pixel]:
            pointings[idx] = old2new_pixel[pixel]
        else:
            pointings[idx] = 0
            pointings_flag[idx] = False
