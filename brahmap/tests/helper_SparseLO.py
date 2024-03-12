import numpy as np


def SparseLO_mult(nrows, pixs, vec):
    prod = np.zeros(nrows)
    for idx in range(nrows):
        if pixs[idx] == -1:
            continue
        prod[idx] += vec[pixs[idx]]
    return prod


def SparseLO_rmult(nrows, ncols, pixs, vec):
    prod = np.zeros(ncols)
    for idx in range(nrows):
        if pixs[idx] == -1:
            continue
        prod[pixs[idx]] += vec[idx]
    return prod


def SparseLO_mult_qu(nrows, pixs, sin, cos, vec):
    prod = np.zeros(nrows)
    for idx in range(nrows):
        if pixs[idx] == -1:
            continue
        prod[idx] += vec[2 * pixs[idx]] * cos[idx] + vec[2 * pixs[idx] + 1] * sin[idx]
    return prod


def SparseLO_rmult_qu(nrows, ncols, pixs, sin, cos, vec):
    prod = np.zeros(ncols * 2)
    for idx in range(nrows):
        if pixs[idx] == -1:
            continue
        prod[2 * pixs[idx]] += vec[idx] * cos[idx]
        prod[2 * pixs[idx] + 1] += vec[idx] * sin[idx]
    return prod


def SparseLO_mult_iqu(nrows, pixs, sin, cos, vec):
    prod = np.zeros(nrows)
    for idx in range(nrows):
        if pixs[idx] == -1:
            continue
        prod[idx] += (
            vec[3 * pixs[idx]]
            + vec[3 * pixs[idx] + 1] * cos[idx]
            + vec[3 * pixs[idx] + 2] * sin[idx]
        )
    return prod


def SparseLO_rmult_iqu(nrows, ncols, pixs, sin, cos, vec):
    prod = np.zeros(ncols * 3)
    for idx in range(nrows):
        if pixs[idx] == -1:
            continue
        prod[3 * pixs[idx]] += vec[idx]
        prod[3 * pixs[idx] + 1] += vec[idx] * cos[idx]
        prod[3 * pixs[idx] + 2] += vec[idx] * sin[idx]
    return prod
