import numpy as np
from ..linop import linop as lp
from ..linop import blkop as blk

import SparseLO_tools
import BlkDiagPrecondLO_tools


class SparseLO(lp.LinearOperator):
    r"""Derived class from the one from the  :class:`LinearOperator` in :mod:`linop`.
    It constitutes an interface for dealing with the projection operator
    (pointing matrix).

    Since this can be represented as a sparse matrix, it is initialized \
    by an array of observed pixels which resembles the  ``(i,j)`` positions \
    of the non-null elements of  the matrix,``obs_pixs``.

    **Parameters**

    - ``n`` : {int}
        size of the pixel domain ;
    - ``m`` : {int}
        size of  time domain;
        (or the non-null elements in a row of :math:`A_{i,j}`);
    - ``pix_samples`` : {array}
        list of pixels observed in the time domain,
        (or the non-null elements in a row of :math:`A_{i,j}`);
    - ``pol`` : {int,[*default* `pol=1`]}
        process an intensity only (``pol=1``), polarization only ``pol=2``
        and intensity+polarization map (``pol=3``);
    - ``angle_processed``: {:class:`ProcessTimeSamples`}
        the class (instantiated before :class:`SparseLO`)has already computed
        the :math:`\cos 2\phi` and :math:`\sin 2\phi`, we link the ``cos`` and ``sin``
        attributes of this class to the  :class:`ProcessTimeSamples` ones ;

    """

    def mult(self, v):
        r"""
        Performs the product of a sparse matrix :math:`Av`,\
         with :math:`v` a  :mod:`numpy`  array (:math:`dim(v)=n_{pix}`)  .

        It extracts the components of :math:`v` corresponding  to the non-null \
        elements of the operator.

        """
        x = SparseLO_tools.py_SparseLO_mult(self.nrows, self.pairs, v)

        return x

    def rmult(self, v):
        r"""
        Performs the product for the transpose operator :math:`A^T`.

        """
        x = SparseLO_tools.py_SparseLO_rmult(self.nrows, self.ncols, self.pairs, v)

        return x

    def mult_qu(self, v):
        r"""Performs :math:`A * v` with :math:`v` being a *polarization* vector.
        The output array will encode a linear combination of the two Stokes
        parameters,  (whose components are stored contiguously).

        .. math::
            d_t=  Q_p \cos(2\phi_t)+ U_p \sin(2\phi_t).
        """
        x = SparseLO_tools.py_SparseLO_mult_qu(
            self.nrows, self.pairs, self.sin, self.cos, v
        )

        return x

    def rmult_qu(self, v):
        r"""
        Performs :math:`A^T * v`. The output vector will be a QU-map-like array.
        """
        vec_out = SparseLO_tools.py_SparseLO_rmult_qu(
            self.nrows, self.ncols, self.pairs, self.sin, self.cos, v
        )

        return vec_out

    def mult_iqu(self, v):
        r"""Performs the product of a sparse matrix :math:`Av`,
        with ``v`` a  :mod:`numpy` array containing the
        three Stokes parameters [IQU] .

        .. note::
            Compared to the operation ``mult`` this routine returns a
            :math:`n_t`-size vector defined as:

            .. math::
                d_t= I_p + Q_p \cos(2\phi_t)+ U_p \sin(2\phi_t).

            with :math:`p` is the pixel observed at time :math:`t` with polarization angle
            :math:`\phi_t`.
        """
        x = SparseLO_tools.py_SparseLO_mult_iqu(
            self.nrows, self.pairs, self.sin, self.cos, v
        )

        return x

    def rmult_iqu(self, v):
        r"""
        Performs the product for the transpose operator :math:`A^T` to get a IQU map-like vector.
        Since this vector resembles the pixel of 3 maps it has 3 times the size ``Npix``.
        IQU values referring to the same pixel are  contiguously stored in the memory.

        """
        x = SparseLO_tools.py_SparseLO_rmult_iqu(
            self.nrows, self.ncols, self.pairs, self.sin, self.cos, v
        )

        return x

    def __init__(self, n, m, pix_samples, pol=1, angle_processed=None):
        self.ncols = n
        self.nrows = m
        self.pol = pol
        self.pairs = pix_samples
        if self.pol > 1:
            self.cos = angle_processed.cos
            self.sin = angle_processed.sin

        if pol == 3:
            self.__runcase = "IQU"
            super(SparseLO, self).__init__(
                nargin=self.pol * self.ncols,
                nargout=self.nrows,
                matvec=self.mult_iqu,
                symmetric=False,
                rmatvec=self.rmult_iqu,
            )
        elif pol == 1:
            self.__runcase = "I"
            super(SparseLO, self).__init__(
                nargin=self.pol * self.ncols,
                nargout=self.nrows,
                matvec=self.mult,
                symmetric=False,
                rmatvec=self.rmult,
            )
        elif pol == 2:
            self.__runcase = "QU"
            super(SparseLO, self).__init__(
                nargin=self.pol * self.ncols,
                nargout=self.nrows,
                matvec=self.mult_qu,
                symmetric=False,
                rmatvec=self.rmult_qu,
            )
        else:
            raise RuntimeError(
                "No valid polarization key set!\t=>\tpol=%d \n \
                                    Possible values are pol=%d(I),%d(QU), %d(IQU)."
                % (pol, 1, 2, 3)
            )

    @property
    def maptype(self):
        """
        Return a string depending on the map you are processing
        """
        return self.__runcase


class ToeplitzLO(lp.LinearOperator):
    r"""
    Derived Class from a LinearOperator. It exploit the symmetries of an ``dim x dim``
    Toeplitz matrix.
    This particular kind of matrices satisfy the following relation:

    .. math::

        A_{i,j}=A_{i+1,j+1}=a_{i-j}

    Therefore, it is enough to initialize ``A`` by mean of an array ``a`` of ``size = dim``.

    **Parameters**

    - ``a`` : {array, list}
        the array which resembles all the elements of the Toeplitz matrix;
    - ``size`` : {int}
        size of the block.

    """

    def mult(self, v):
        """
        Performs the product of a Toeplitz matrix with a vector ``x``.

        """
        val = self.array[0]
        y = val * v
        for i in range(1, len(self.array)):
            val = self.array[i]
            temp = val * v
            y[:-i] += temp[i:]
            y[i:] += temp[:-i]

        return y

    def __init__(self, a, size):
        super(ToeplitzLO, self).__init__(
            nargin=size, nargout=size, matvec=self.mult, symmetric=True
        )
        self.array = a


class BlockLO(blk.BlockDiagonalLinearOperator):
    r"""
    Derived class from  :mod:`blkop.BlockDiagonalLinearOperator`.
    It basically relies on the definition of a block diagonal operator,
    composed by ``nblocks``  diagonal operators with equal size .
    If it does not have any  off-diagonal terms (*default case* ), each block is a multiple  of
    the identity characterized by the  values listed in ``t`` and therefore is
    initialized by the :func:`BlockLO.build_blocks` as a :class:`linop.DiagonalOperator`.

    **Parameters**

    - ``blocksize`` : {int or list }
        size of each diagonal block, if `int` it is : :math:`blocksize= n/nblocks`.
    - ``t`` : {array}
        noise values for each block
    - ``offdiag`` : {bool, default ``False``}
        strictly  related to the way  the array ``t`` is passed (see notes ).

        .. note::

            - True : ``t`` is a list of array,
                    ``shape(t)= [nblocks,bandsize]``, to have a Toeplitz band diagonal operator,
                    :math:`bandsize != blocksize`
            - False : ``t`` is an array, ``shape(t)=[nblocks]``.
                    each block is identified by a scalar value in the diagonal.

    """

    def build_blocks(self):
        r"""
        Build each block of the operator either with or
        without off diagonal terms.
        Each block is initialized as a Toeplitz (either **band** or **diagonal**)
        linear operator.

        .. see also::

        ``self.diag``: {numpy array}
            the array resuming the :math:`diag(N^{-1})`.


        """

        tmplist = []
        self.blocklist = []

        for idx, elem in enumerate(self.covnoise):
            d = np.ones(self.blocksize[idx])

            # d = np.empty(self.blocksize)
            if self.isoffdiag:
                d.fill(elem[0])
                tmplist.append(d)
                self.blocklist.append(ToeplitzLO(elem, self.blocksize[idx]))
            else:
                d.fill(elem)
                tmplist.append(d)
                self.blocklist.append(lp.DiagonalOperator(d))

            # for j, i in enumerate(self.covnoise):
        self.diag = np.concatenate(tmplist)

    def __init__(self, blocksize, t, offdiag=False):
        self.__isoffdiag = offdiag
        self.blocksize = blocksize
        self.covnoise = t
        self.build_blocks()
        super(BlockLO, self).__init__(self.blocklist)

    @property
    def isoffdiag(self):
        """
        Property saying whether or not the operator has
        off-diagonal terms.
        """
        return self.__isoffdiag


class BlockDiagonalLO(lp.LinearOperator):
    r"""
    Explicit implementation of :math:`A diag(N^{-1}) A^T`, in order to save time
    in the application of the two matrices onto a vector (in this way the leading dimension  will be :math:`n_{pix}`
    instead of  :math:`n_t`).

    .. note::
        it is initialized as the  :class:`BlockDiagonalPreconditionerLO` since it involves
        computation with  the same matrices.
    """

    def __init__(self, processed_samples, n, pol=1):
        self.size = pol * n
        self.pol = pol
        super(BlockDiagonalLO, self).__init__(
            nargin=self.size, nargout=self.size, matvec=self.mult, symmetric=True
        )
        self.pixels = np.arange(n)
        if pol == 1:
            self.counts = processed_samples.weighted_counts
        elif pol > 1:
            self.sin2 = processed_samples.weighted_sin_sq
            self.sincos = processed_samples.weighted_sincos
            self.cos2 = processed_samples.weighted_cos_sq
            if pol == 3:
                self.counts = processed_samples.weighted_counts
                self.cos = processed_samples.weighted_cos
                self.sin = processed_samples.weighted_sin

    def mult(self, x):
        r"""
        Multiplication of  :math:`A diag(N^{-1}) A^T` on to a vector math:`x`
        ( :math:`n_{pix}` array).
        """
        y = x * 0.0
        if self.pol == 1:
            y = x * self.counts
        elif self.pol == 3:
            for pix, s2, c2, cs, c, s, hits in zip(
                self.pixels,
                self.sin2,
                self.cos2,
                self.sincos,
                self.cos,
                self.sin,
                self.counts,
            ):
                y[3 * pix] = hits * x[3 * pix] + c * x[3 * pix + 1] + s * x[3 * pix + 2]
                y[3 * pix + 1] = (
                    c * x[3 * pix] + c2 * x[3 * pix + 1] + cs * x[3 * pix + 2]
                )
                y[3 * pix + 2] = (
                    s * x[3 * pix] + cs * x[3 * pix + 1] + s2 * x[3 * pix + 2]
                )
        elif self.pol == 2:
            for pix, s2, c2, cs in zip(self.pixels, self.sin2, self.cos2, self.sincos):
                y[pix * 2] = c2 * x[2 * pix] + cs * x[pix * 2 + 1]
                y[pix * 2 + 1] = cs * x[2 * pix] + s2 * x[pix * 2 + 1]
        return y


class BlockDiagonalPreconditionerLO(lp.LinearOperator):
    r"""
    Standard preconditioner defined as:

    .. math::

        M_{BD}=( A diag(N^{-1}) A^T)^{-1}

    where :math:`A` is the *pointing matrix* (see  :class:`SparseLO`).
    Such inverse operator  could be easily computed given the structure of the
    matrix :math:`A`. It could be  sparse in the case of Intensity only analysis (`pol=1`),
    block-sparse if polarization is included (`pol=3,2`).


    **Parameters**

    - ``n``:{int}
        the size of the problem, ``npix``;
    - ``CES``:{:class:`ProcessTimeSamples`}
        the linear operator related to the data sample processing. Its members (`counts`, `masks`,
        `sine`, `cosine`, etc... ) are  needed to explicitly compute the inverse of the
        :math:`n_{pix}` blocks of :math:`M_{BD}`.
    - ``pol``:{int}
        the size of each block of the matrix.
    """

    def mult(self, x):
        r"""
        Action of :math:`y=( A  diag(N^{-1}) A^T)^{-1} x`,
        where :math:`x` is   an :math:`n_{pix}` array.
        """
        y = x * 0.0
        npix = int(self.size / self.pol)

        if self.pol == 1:
            nan = np.ma.masked_greater(self.counts, 0)

            y[nan.mask] = x[nan.mask] / self.counts[nan.mask]
        elif self.pol == 2:
            y = BlkDiagPrecondLO_tools.py_BlkDiagPrecondLO_mult_qu(
                npix, self.sin2, self.cos2, self.sincos, x
            )

        elif self.pol == 3:
            y = BlkDiagPrecondLO_tools.py_BlkDiagPrecondLO_mult_iqu(
                npix,
                self.counts,
                self.sine,
                self.cosine,
                self.sin2,
                self.cos2,
                self.sincos,
                x,
            )

        return y

    def __init__(self, processed_samples, n, pol=1):
        self.size = pol * n
        self.pixels = np.arange(n)
        self.pol = pol
        if pol == 1:
            self.counts = processed_samples.weighted_counts
        elif pol > 1:
            self.sin2 = processed_samples.weighted_sin_sq
            self.cos2 = processed_samples.weighted_cos_sq
            self.sincos = processed_samples.weighted_sincos
            if pol == 3:
                self.counts = processed_samples.weighted_counts
                self.cosine = processed_samples.weighted_cos
                self.sine = processed_samples.weighted_sin

        super(BlockDiagonalPreconditionerLO, self).__init__(
            nargin=self.size, nargout=self.size, matvec=self.mult, symmetric=True
        )


class InverseLO(lp.LinearOperator):
    r"""
    Construct the inverse operator of a matrix :math:`A`, as a linear operator.

    **Parameters**

    - ``A`` : {linear operator}
        the linear operator of the linear system to invert;
    - ``method`` : {function }
        the method to compute ``A^-1`` (see below);
    - ``P`` : {linear operator } (optional)
        the preconditioner for the computation of the inverse operator.

    """

    def mult(self, x):
        r"""
        It returns  :math:`y=A^{-1}x` by solving the linear system :math:`Ay=x`
        with a certain :mod:`scipy` routine (e.g. :func:`scipy.sparse.linalg.cg`)
        defined above as ``method``.
        """

        y, info = self.method(self.A, x, M=self.preconditioner)
        self.isconverged(info)
        return y

    def isconverged(self, info):
        r"""
        It returns a Boolean value  depending on the
        exit status of the solver.

        **Parameters**

        - ``info`` : {int}
            output of the solver method (usually :func:`scipy.sparse.cg`).



        """
        self.__converged = info
        if info == 0:
            return True
        else:
            return False

    def __init__(self, A, method=None, preconditioner=None):
        super(InverseLO, self).__init__(
            nargin=A.shape[0], nargout=A.shape[1], matvec=self.mult, symmetric=True
        )
        self.A = A
        self.__method = method
        self.__preconditioner = preconditioner
        self.__converged = None

    @property
    def method(self):
        r"""
        The method to compute the inverse of A. \
        It can be any :mod:`scipy.sparse.linalg` solver, namely :func:`scipy.sparse.linalg.cg`,
        :func:`scipy.sparse.linalg.bicg`, etc.

        """
        return self.__method

    @property
    def converged(self):
        r"""
        provides convergence information:

        - 0 : successful exit;
        - >0 : convergence to tolerance not achieved, number of iterations;
        - <0 : illegal input or breakdown.

        """
        return self.__converged

    @property
    def preconditioner(self):
        """
        Preconditioner for the solver.
        """
        return self.__preconditioner
