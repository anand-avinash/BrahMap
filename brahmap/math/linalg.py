from typing import Callable
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg

from brahmap import MPI_UTILS
from ..base import LinearOperator


def parallel_norm(x: np.ndarray) -> float:
    """A replacement of `np.linalg.norm` to compute 2-norm of a vector
    distributed among multiple MPI processes

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    float
        The norm of vector `x`
    """
    sqnorm = x.dot(x)
    sqnorm = MPI_UTILS.comm.allreduce(sqnorm)
    ret = np.sqrt(sqnorm)
    return ret


def cg(
    A: LinearOperator,
    b: np.ndarray,
    x0: np.ndarray = None,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-12,
    maxiter: int = 100,
    M: LinearOperator = None,
    callback: Callable = None,
    parallel: bool = True,
):
    """A replacement of `scipy.sparse.linalg.cg` where `np.linalg.norm` is
    replaced with `brahmap.math.parallel_norm` when the parameter `parallel`
    is set `True`

    Parameters
    ----------
    A : LinearOperator
        _description_
    b : np.ndarray
        _description_
    x0 : np.ndarray, optional
        _description_, by default None
    rtol : float, optional
        _description_, by default 1.0e-12
    atol : float, optional
        _description_, by default 1.0e-12
    maxiter : int, optional
        _description_, by default 100
    M : LinearOperator, optional
        _description_, by default None
    callback : Callable, optional
        _description_, by default None
    parallel : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_
    """
    A, M, x, b, postprocess = scipy.sparse.linalg._isolve.utils.make_system(
        A,
        M,
        x0,
        b,
    )

    if parallel:
        norm_function: Callable = parallel_norm
    else:
        norm_function: Callable = np.linalg.norm

    b_norm = norm_function(b)

    atol, _ = scipy.sparse.linalg._isolve.iterative._get_atol_rtol(
        "cg",
        b_norm,
        atol,
        rtol,
    )

    if b_norm == 0:
        return postprocess(b), 0

    dotprod = np.vdot if np.iscomplexobj(x) else np.dot

    # r = b - A@x if x has any non-zero element, otherwise r = b
    r = b - A * x if x.any() else b.copy()

    # Dummy initialization
    rho_prev, p = None, None

    norm_residual = 1.0

    for iteration in range(maxiter):
        if norm_residual < atol:
            return postprocess(x), 0

        z = M * r
        rho_cur = dotprod(r, z)
        if iteration > 0:
            beta = rho_cur / rho_prev
            p *= beta
            p += z
        else:
            p = np.empty_like(r)
            p[:] = z[:]

        q = A * p
        alpha = rho_cur / dotprod(p, q)
        x += alpha * p
        r -= alpha * q
        rho_prev = rho_cur

        norm_residual = norm_function(r) / b_norm

        if callback:
            callback(x, r, norm_residual)

    else:
        return postprocess(x), maxiter
