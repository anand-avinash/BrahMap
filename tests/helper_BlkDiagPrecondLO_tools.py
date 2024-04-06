import numpy as np


def BDPLO_mult_I(weighted_counts: np.ndarray, vec: np.ndarray):
    prod = vec / weighted_counts

    return prod


def BDPLO_mult_QU(
    solver_type: int,
    new_npix: int,
    weighted_sin_sq: np.ndarray,
    weighted_cos_sq: np.ndarray,
    weighted_sincos: np.ndarray,
    vec: np.ndarray,
):
    prod = np.zeros(new_npix * solver_type, dtype=vec.dtype)
    determinant = (weighted_cos_sq * weighted_sin_sq) - (
        weighted_sincos * weighted_sincos
    )
    mask = np.ma.masked_greater(abs(determinant), 1.0e-5).mask

    for idx in range(new_npix):
        if mask[idx]:
            prod[2 * idx] = (
                weighted_sin_sq[idx] * vec[2 * idx]
                - weighted_sincos[idx] * vec[2 * idx + 1]
            ) / determinant[idx]
            prod[2 * idx + 1] = (
                -weighted_sincos[idx] * vec[2 * idx]
                + weighted_cos_sq[idx] * vec[2 * idx + 1]
            ) / determinant[idx]
        else:
            continue

    return prod


def BDPLO_mult_IQU(
    solver_type: int,
    new_npix: int,
    weighted_counts: np.ndarray,
    weighted_sin_sq: np.ndarray,
    weighted_cos_sq: np.ndarray,
    weighted_sincos: np.ndarray,
    weighted_sin: np.ndarray,
    weighted_cos: np.ndarray,
    vec: np.ndarray,
):
    prod = np.zeros(new_npix * solver_type, dtype=vec.dtype)
    determinant = (
        weighted_counts
        * (weighted_cos_sq * weighted_sin_sq - weighted_sincos * weighted_sincos)
        - weighted_cos * weighted_cos * weighted_sin_sq
        - weighted_sin * weighted_sin * weighted_cos_sq
        + 2.0 * weighted_cos * weighted_sin * weighted_sincos
    )
    mask = np.ma.masked_greater(abs(determinant), 1.0e-5).mask

    for idx in range(new_npix):
        if mask[idx]:
            prod[3 * idx] = (
                (
                    weighted_cos_sq[idx] * weighted_sin_sq[idx]
                    - weighted_sincos[idx] * weighted_sincos[idx]
                )
                * vec[3 * idx]
                + (
                    weighted_sin[idx] * weighted_sincos[idx]
                    - weighted_cos[idx] * weighted_sin_sq[idx]
                )
                * vec[3 * idx + 1]
                + (
                    weighted_cos[idx] * weighted_sincos[idx]
                    - weighted_sin[idx] * weighted_cos_sq[idx]
                )
                * vec[3 * idx + 2]
            ) / determinant[idx]
            prod[3 * idx + 1] = (
                (
                    weighted_sin[idx] * weighted_sincos[idx]
                    - weighted_cos[idx] * weighted_sin_sq[idx]
                )
                * vec[3 * idx]
                + (
                    weighted_counts[idx] * weighted_sin_sq[idx]
                    - weighted_sin[idx] * weighted_sin[idx]
                )
                * vec[3 * idx + 1]
                + (
                    weighted_sin[idx] * weighted_cos[idx]
                    - weighted_counts[idx] * weighted_sincos[idx]
                )
                * vec[3 * idx + 2]
            ) / determinant[idx]
            prod[3 * idx + 2] = (
                (
                    weighted_cos[idx] * weighted_sincos[idx]
                    - weighted_sin[idx] * weighted_cos_sq[idx]
                )
                * vec[3 * idx]
                + (
                    -weighted_counts[idx] * weighted_sincos[idx]
                    + weighted_cos[idx] * weighted_sin[idx]
                )
                * vec[3 * idx + 1]
                + (
                    weighted_counts[idx] * weighted_cos_sq[idx]
                    - weighted_cos[idx] * weighted_cos[idx]
                )
                * vec[3 * idx + 2]
            ) / determinant[idx]
        else:
            continue

    return prod
