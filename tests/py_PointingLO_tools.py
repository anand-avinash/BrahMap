import numpy as np
from mpi4py import MPI


def PLO_mult_I(
    nrows,
    pointings: np.ndarray,
    pointings_flags: np.ndarray,
    vec: np.ndarray,
):
    prod = np.zeros(nrows, dtype=vec.dtype)

    for idx in range(nrows):
        pixel = pointings[idx]
        if pointings_flags[idx]:
            prod[idx] += vec[pixel]

    return prod


def PLO_rmult_I(
    nrows,
    ncols,
    pointings: np.ndarray,
    pointings_flags: np.ndarray,
    vec: np.ndarray,
    comm,
):
    prod = np.zeros(ncols, dtype=vec.dtype)

    for idx in range(nrows):
        pixel = pointings[idx]
        if pointings_flags[idx]:
            prod[pixel] += vec[idx]

    comm.Allreduce(MPI.IN_PLACE, prod, MPI.SUM)

    return prod


def PLO_mult_QU(
    nrows,
    pointings: np.ndarray,
    pointings_flags: np.ndarray,
    sin2phi: np.ndarray,
    cos2phi: np.ndarray,
    vec: np.ndarray,
):
    prod = np.zeros(nrows, dtype=vec.dtype)

    for idx in range(nrows):
        pixel = pointings[idx]
        if pointings_flags[idx]:
            prod[idx] += (
                vec[2 * pixel] * cos2phi[idx] + vec[2 * pixel + 1] * sin2phi[idx]
            )

    return prod


def PLO_rmult_QU(
    nrows,
    ncols,
    pointings: np.ndarray,
    pointings_flags: np.ndarray,
    sin2phi: np.ndarray,
    cos2phi: np.ndarray,
    vec: np.ndarray,
    comm,
):
    prod = np.zeros(ncols, dtype=vec.dtype)

    for idx in range(nrows):
        pixel = pointings[idx]
        if pointings_flags[idx]:
            prod[2 * pixel] += vec[idx] * cos2phi[idx]
            prod[2 * pixel + 1] += vec[idx] * sin2phi[idx]

    comm.Allreduce(MPI.IN_PLACE, prod, MPI.SUM)

    return prod


def PLO_mult_IQU(
    nrows,
    pointings: np.ndarray,
    pointings_flags: np.ndarray,
    sin2phi: np.ndarray,
    cos2phi: np.ndarray,
    vec: np.ndarray,
):
    prod = np.zeros(nrows, dtype=vec.dtype)

    for idx in range(nrows):
        pixel = pointings[idx]
        if pointings_flags[idx]:
            prod[idx] += (
                vec[3 * pixel]
                + vec[3 * pixel + 1] * cos2phi[idx]
                + vec[3 * pixel + 2] * sin2phi[idx]
            )

    return prod


def PLO_rmult_IQU(
    nrows,
    ncols,
    pointings: np.ndarray,
    pointings_flags: np.ndarray,
    sin2phi: np.ndarray,
    cos2phi: np.ndarray,
    vec: np.ndarray,
    comm,
):
    prod = np.zeros(ncols, dtype=vec.dtype)

    for idx in range(nrows):
        pixel = pointings[idx]
        if pointings_flags[idx]:
            prod[3 * pixel] += vec[idx]
            prod[3 * pixel + 1] += vec[idx] * cos2phi[idx]
            prod[3 * pixel + 2] += vec[idx] * sin2phi[idx]

    comm.Allreduce(MPI.IN_PLACE, prod, MPI.SUM)

    return prod
