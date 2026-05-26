from typing import Callable
import numpy as np
import numpy.typing as npt
import scipy
import scipy.sparse
import scipy.sparse.linalg

from ..mpi import MPI_UTILS
from ..base import LinearOperator


def parallel_norm(x: npt.NDArray[np.number]) -> float:
    """Computes the 2-norm of a vector distributed among multiple MPI
    processes as a replacement for
    [`np.linalg.norm()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html).

    Parameters
    ----------
    x : npt.NDArray[np.number]
        The input array to compute the norm for

    Returns
    -------
    float
        The final computed 2-norm of the vector $x$
    """
    sqnorm = x.dot(x)
    sqnorm = MPI_UTILS.comm.allreduce(sqnorm)
    ret = np.sqrt(sqnorm)
    return ret


def cg(
    A: LinearOperator,
    b: npt.NDArray[np.number],
    x0: npt.NDArray[np.number] | None = None,
    atol: float = 1.0e-12,
    maxiter: int = 100,
    M: LinearOperator | None = None,
    callback: Callable | None = None,
    parallel: bool = False,
) -> tuple[npt.NDArray[np.number], int]:
    """An MPI-parallelized replacement of
    [`scipy.sparse.linalg.cg()`](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.sparse.linalg.cg.html).

    It provides the conjugate gradient (CG) solver for the linear
    equation $A \\cdot x = b$ with an optional preconditioner $M$ and an
    initial guess $x0$.

    This function replaces
    [`np.linalg.norm()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)
    with [`brahmap.math.parallel_norm()`][brahmap.math.parallel_norm] when
    the `parallel` parameter is set to `True`. All matrices and vectors are assumed
    to be real.

    Parameters
    ----------
    A : LinearOperator
        The primary linear operator or matrix $A$
    b : npt.NDArray[np.number]
        The right-hand side vector (RHS) $b$
    x0 : npt.NDArray[np.number] | None, optional
        The initial guess for the solution vector $x0$, by default `None`
    atol : float, optional
        The absolute tolerance for convergence, by default `1.0e-12`
    maxiter : int, optional
        The maximum number of iterations allowed, by default `100`
    M : LinearOperator | None, optional
        The preconditioner linear operator to accelerate convergence, by default `None`
    callback : Callable | None, optional
        A callback function to be called after each iteration, by default `None`
    parallel : bool, optional
        Whether to enable MPI parallelized computation of the 2-norm, by
        default `False`

    Returns
    -------
    tuple[npt.NDArray[np.number], int]
        A tuple containing the final computed output vector and the
        convergence status code. The status code 0 implies a successful
        convergence
    """
    temp_tuple = scipy.sparse.linalg._isolve.utils.make_system(
        A,
        M,
        x0,
        b,
    )

    # Starting from SciPy 1.16.0, `make_system` returns 4 objects instead of 5.
    # Even in earlier versions, only the first 4 objects were needed for our
    # use. The following unpacking ensures compatibility across all versions.
    # This logic can be simplified once support for versions below 1.16.0 is
    # dropped.
    A = temp_tuple[0]  # type: ignore
    M = temp_tuple[1]  # type: ignore
    x = temp_tuple[2]
    b = temp_tuple[3]

    if parallel:
        norm_function: Callable = parallel_norm
    else:

        def norm_function(x: npt.NDArray[np.number]) -> float:
            return np.sqrt(x.dot(x))

    b_norm = norm_function(b)

    if b_norm == 0:
        return b, 0

    # r = b - A@x if x has any non-zero element, otherwise r = b
    r = b - A * x if x.any() else b.copy()

    # Dummy initialization
    rho_prev, p = None, None

    norm_residual = 1.0

    for iteration in range(maxiter):
        if norm_residual < atol:
            return x, 0

        z = M * r  # type: ignore
        rho_cur = np.dot(r, z)  # type: ignore
        if iteration > 0:
            beta = rho_cur / rho_prev
            p *= beta
            p += z
        else:
            p = z.copy()  # type: ignore

        q = A * p
        alpha = rho_cur / np.dot(p, q)
        x += alpha * p
        r -= alpha * q
        rho_prev = rho_cur

        norm_residual = norm_function(r) / b_norm

        if callback:
            callback(x, r, norm_residual)

    else:
        return x, maxiter
