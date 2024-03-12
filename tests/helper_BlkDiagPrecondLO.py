import numpy as np


def BlkDiagPrecondLO_mult_qu(npix, sin2, cos2, sincos, vec):
    prod = np.zeros_like(vec)
    determinant = (cos2 * sin2) - (sincos * sincos)
    mask = np.ma.masked_greater(abs(determinant), 1.0e-5).mask

    for idx in range(npix):
        if mask[idx]:
            prod[2 * idx] = (
                sin2[idx] * vec[2 * idx] - sincos[idx] * vec[2 * idx + 1]
            ) / determinant[idx]
            prod[2 * idx + 1] = (
                -sincos[idx] * vec[2 * idx] + cos2[idx] * vec[2 * idx + 1]
            ) / determinant[idx]
        else:
            continue

    return prod


def BlkDiagPrecondLO_mult_iqu(npix, counts, sine, cosine, sin2, cos2, sincos, vec):
    prod = np.zeros_like(vec)
    determinant = (
        counts * (cos2 * sin2 - sincos * sincos)
        - cosine * cosine * sin2
        - sine * sine * cos2
        + 2.0 * cosine * sine * sincos
    )
    mask = np.ma.masked_greater(abs(determinant), 1.0e-5).mask

    for idx in range(npix):
        if mask[idx]:
            prod[3 * idx] = (
                (cos2[idx] * sin2[idx] - sincos[idx] * sincos[idx]) * vec[3 * idx]
                + (sine[idx] * sincos[idx] - cosine[idx] * sin2[idx]) * vec[3 * idx + 1]
                + (cosine[idx] * sincos[idx] - sine[idx] * cos2[idx]) * vec[3 * idx + 2]
            ) / determinant[idx]
            prod[3 * idx + 1] = (
                (sine[idx] * sincos[idx] - cosine[idx] * sin2[idx]) * vec[3 * idx]
                + (counts[idx] * sin2[idx] - sine[idx] * sine[idx]) * vec[3 * idx + 1]
                + (sine[idx] * cosine[idx] - counts[idx] * sincos[idx])
                * vec[3 * idx + 2]
            ) / determinant[idx]
            prod[3 * idx + 2] = (
                (cosine[idx] * sincos[idx] - sine[idx] * cos2[idx]) * vec[3 * idx]
                + (-counts[idx] * sincos[idx] + cosine[idx] * sine[idx])
                * vec[3 * idx + 1]
                + (counts[idx] * cos2[idx] - cosine[idx] * cosine[idx])
                * vec[3 * idx + 2]
            ) / determinant[idx]
        else:
            continue

    return prod
