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
    one_over_determinant: np.ndarray,
    vec: np.ndarray,
):
    prod = np.zeros(new_npix * solver_type, dtype=vec.dtype)

    for idx in range(new_npix):
        prod[2 * idx] = (
            weighted_sin_sq[idx] * vec[2 * idx]
            - weighted_sincos[idx] * vec[2 * idx + 1]
        ) * one_over_determinant[idx]
        prod[2 * idx + 1] = (
            -weighted_sincos[idx] * vec[2 * idx]
            + weighted_cos_sq[idx] * vec[2 * idx + 1]
        ) * one_over_determinant[idx]

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
    one_over_determinant: np.ndarray,
    vec: np.ndarray,
):
    prod = np.zeros(new_npix * solver_type, dtype=vec.dtype)

    for idx in range(new_npix):
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
        ) * one_over_determinant[idx]
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
        ) * one_over_determinant[idx]
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
        ) * one_over_determinant[idx]

    return prod
