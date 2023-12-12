#
#   DEFLATIONLIB.PY
#   is a package of functions interfacing krypy routines for KRYLOV  subspaces
#   date: 2016-12-02
#   author: GIUSEPPE PUGLISI
#
#   Copyright (C) 2016   Giuseppe Puglisi    giuspugl@sissa.it
#

from scipy.linalg import get_blas_funcs
import scipy.sparse.linalg as spla
import krypy as kp

import numpy as np
from ..utilities import *


def arnoldi(A, b, x0=None, tol=1e-5, maxiter=1000, inner_m=30):
    """
    Computes an orthonormal basis to get the approximated eigenvalues
    (Ritz eigenvalues) and eigenvector.

    The basis comes from a Gram-Schmidt orthonormalization of the Krylov
    subspace  defined as:

    .. math::

        K_m = span( b, Ab, ..., A^{m-1} b )

    at the :math:`m`-th iteration.

    **Parameters**

    - ``A`` : {sparse matrix , linear operator}
            matrix we want to approximate eigenvectors;
    - ``b`` : {array}
            array to build the Krylov subspace ;
    - ``x0`` : {array}
            initial guess vector to compute residuals;
    - ``tol`` : {float}
            tolerance threshold to the Ritz eigenvalue computation;
    - ``maxiter`` : {int}
            to validate the result one can compute ``maxiter`` times the
            eigenvalues, to seek the stability of the algorithm;
    - ``inner_m`` :  {int}
            maximum number of iterations within the Arnoldi algorithm,

            .. Warning::

                ``inner_m <=N_pix``

    **Returns**

    - ``w`` : {list of arrays}
            the orthonormal basis ``m x N_pix``;
    - ``h`` : {list of arrays}
            the elements of the :math:`H_m` Hessenberg matrix.
            At the ``m``-th iteration  :math:`h_m` has got :math:`m+1` elements.

    """
    if not np.isfinite(b).all():
        raise ValueError("RHS must contain only finite numbers")

    matvec = A.matvec
    axpy, dot, scal = None, None, None

    b_norm = norm2(b)
    if b_norm == 0:
        b_norm = 1

    r_outer = b - matvec(x0)
    # -- determine input type routines
    if axpy is None:
        if np.iscomplexobj(r_outer) and not np.iscomplexobj(x0):
            x0 = x0.astype(r_outer.dtype)
        axpy, dot, scal = get_blas_funcs(["axpy", "dot", "scal"], (x0, r_outer))

    # -- check stopping condition
    r_norm = norm2(r_outer)
    if r_norm < tol * b_norm or r_norm < tol:
        print(
            "Arnoldi exited at the first iteration\nr_norm < tol * b_norm or r_norm < tol"
        )
        return None, None, 0
    # -- ARNOLDI ALGRITHM
    vs0 = scal(1.0 / r_norm, r_outer)  # q=x/||x||
    hs = []
    vs = [vs0]
    v_new = None

    for j in range(1, 1 + inner_m):
        v_new = matvec(vs[j - 1])  # r=A q
        v_new2 = v_new.copy()
        #     ++ orthogonalize
        hcur = []
        for v in vs:
            alpha = dot(v, v_new)  # alpha= (q,r)
            hcur.append(alpha)
            v_new = axpy(v, v_new2, v.shape[0], -alpha)  # v_new -= alpha*v
        hcur.append(norm2(v_new))
        #       ++ normalize
        v_new = scal(1.0 / hcur[-1], v_new)
        if abs(v_new[j] * hcur[-1]) <= tol:
            print("--------------------------------------")
            print("Computed  %d Ritz eigenvalues within the tolerance %.1g " % (j, tol))
            print("--------------------------------------")

            hs.append(hcur)
            return vs, hs, j

        vs.append(v_new)
        hs.append(hcur)
        if j == inner_m:
            raise RuntimeError("Convergence not achieved within the Arnoldi algorithm")
            return None, None, j


def build_hess(h, m):
    """
    Compute  and store (as a Hessenberg matrix) the :math:`H_m` matrix from the
    output list ``h`` of the :func:`arnoldi` routine.

    **Parameters**

    - ``h`` : {list of arrays}
            matrix coefficients ;
    - ``m`` : {int}
            size of ``H``

    **Returns**

    - ``H`` :{numpy.matrix}

    """
    hess = np.zeros((m, m))
    for q in range(m - 1):
        hess[: (q + 2), q] = h[q]
    hess[:m, m - 1] = h[-1][:m]

    return hess


def build_Z(z, y, w, eps):
    """
    Build the deflation matrix :math:`Z`. Its columns are the :math:`r`
    selected eigenvectors :math:`Z_i=w_m*y_i` s.t. their eigenvalues  :math:`z_i`
    are smaller than a certain threshold ``eps``.

    **Parameters**

    - ``z`` : {array}
        eigenvalues of :math:`H_m`;
    - ``y`` : {list of arrays}
        eigenvectors of :math:`H_m`;
    - ``w`` : {list of arrays}
        orthonormal basis (computed with the Arnoldi algorithm);
    - ``eps`` : {float}
        threshold to select the smallest eigenvalues.


    **Returns**

    - ``Z`` : {matrix}
            deflation subspace matrix;
    - ``r`` :  {int}
            :math:`rank(Z)`.

    """

    m = len(z)

    npix = len(w[0])

    select_eigvec = []
    for i in range(m):
        if abs(z[i]) <= eps:
            select_eigvec.append(y[i])
    r = len(select_eigvec)
    if r == 0:
        raise RuntimeError(
            "No Ritz eigenvalue are found smaller than fixed threshold %.1g " % eps
        )
    print("++++++++++++++++++++++++++++++++++++")
    print(
        "Found  eigenvectors below the threshold %.1g!\nThe deflation subspace  has dim(Z)=%d "
        % (eps, r)
    )
    print("++++++++++++++++++++++++++++++++++++")
    z = np.matrix(select_eigvec)

    Z = dgemm(w.T, z)
    return Z, r


def run_krypy_arnoldi(A, x0, M, tol, maxiter=None):
    N = len(x0)
    if maxiter is None:
        nmax = N
    else:
        nmax = maxiter
    x0 = x0.reshape((N, 1))
    Aop = spla.aslinearoperator(A)

    if M is not None:
        prec = spla.aslinearoperator(M)
        v, h, p = kp.utils.arnoldi(Aop, x0, M=prec, maxiter=nmax)
    else:
        v, h = kp.utils.arnoldi(Aop, x0, maxiter=nmax)

    m = v.shape[1]
    print(
        "Residual after  %d Arnoldi iterations, r^(k)= %g \nExiting Arnoldi ..."
        % (m, norm2(v[:, -1]))
    )
    return v, h, m


def find_ritz_eigenvalues(h, v, threshold=1.0e-2, eigenvalues=False, filename=None):
    eig, u, resnorm, z = kp.utils.ritz(h, V=v, hermitian=True)
    # orthonormalize eigenvectors
    # Q,R=kp.utils.qr(z)
    selected = np.ma.masked_less(eig, threshold)
    r = len(eig[selected.mask])
    print("//" * 30)
    print("Found   %d Ritz eigenvalues smaller than %.1g " % (r, threshold))
    print(eig[:r])
    print("//" * 30)
    if not filename is None:
        write_ritz_eigenvectors_to_hdf5(z, filename, eigvals=eig)
    if eigenvalues:
        return z[:, selected.mask], r, eig[selected.mask]
    else:
        return z[:, :r], r
