#
#   LINEAR_ALGEBRA_FUNCS.PY
#   interface function to blas routines
#   date: 2016-12-02
#   author: GIUSEPPE PUGLISI
#
#   Copyright (C) 2016   Giuseppe Puglisi    giuspugl@sissa.it
#


from scipy.linalg import get_blas_funcs
from scipy.special import legendre
import numpy as np


def dgemm(A, B):
    """
    Compute Matrix-Matrix multiplication from the BLAS routine DGEMM
    If ``A ,B``  are ordered as lists it convert them
    as matrices via the `` numpy.asarray`` function.
    """
    if isinstance(A, list):
        A = np.asarray(A, order="F")
    if isinstance(B, list):
        B = np.asarray(B, order="F")

    matdot = get_blas_funcs("gemm", (A, B))

    return matdot(alpha=1.0, a=A.T, b=B, trans_b=True, trans_a=False)


def norm2(q):
    """
    Compute the euclidean norm of an array ``q`` by calling the BLAS routine
    """
    q = np.asarray(q)
    nrm2 = get_blas_funcs("nrm2", dtype=q.dtype)
    return nrm2(q)


def scalprod(a, b):
    """
    Scalar product of two vectors ``a`` and ``b``.
    """
    dot = get_blas_funcs("dot", (a, b))
    return dot(a, b)


def get_legendre_polynomials(polyorder, size):
    """
    Returns a ``size x polyorder`` matrix whose   columns contain
    the respective Legendre polynomial in :math:`\left[ -1,1 \right` normalized.
    """

    legendres = np.empty([size, polyorder + 1])
    x = np.linspace(-1, 1, size)
    for i in range(polyorder + 1):
        L = legendre(i)
        legendres[:, i] = L(x) / norm2(L(x))

    return legendres
