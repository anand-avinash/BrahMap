import numpy as np
import warnings
from ..linop import linop as lp
from ..linop import blkop as blk
from ..utilities import ProcessTimeSamples, TypeChangeWarning

import PointingLO_tools
import BlkDiagPrecondLO_tools
import InvNoiseCov_tools


class PointingLO(lp.LinearOperator):
    r"""Derived class from the one from the  :class:`LinearOperator` in :mod:`linop`.
    It constitutes an interface for dealing with the projection operator
    (pointing matrix).

    Since this can be represented as a sparse matrix, it is initialized \
    by an array of observed pixels which resembles the  ``(i,j)`` positions \
    of the non-null elements of  the matrix,``obs_pixs``.

    **Parameters**

    - ``processed_samples``: {:class:`ProcessTimeSamples`}
        the class (instantiated before :class:`PointingLO`)has already computed
        the :math:`\cos 2\phi` and :math:`\sin 2\phi`, we link the ``cos2phi`` and ``sin2phi``
        attributes of this class to the  :class:`ProcessTimeSamples` ones ;

    """

    def __init__(
        self,
        processed_samples: ProcessTimeSamples,
    ):
        self.solver_type = processed_samples.solver_type

        self.ncols = processed_samples.new_npix * self.solver_type
        self.nrows = processed_samples.nsamples

        self.pointings = processed_samples.pointings
        self.pointings_flag = processed_samples.pointings_flag

        if self.solver_type > 1:
            self.sin2phi = processed_samples.sin2phi
            self.cos2phi = processed_samples.cos2phi

        if self.solver_type == 1:
            super(PointingLO, self).__init__(
                nargin=self.ncols,
                nargout=self.nrows,
                symmetric=False,
                matvec=self._mult_I,
                rmatvec=self._rmult_I,
                dtype=processed_samples.dtype_float,
            )
        elif self.solver_type == 2:
            super(PointingLO, self).__init__(
                nargin=self.ncols,
                nargout=self.nrows,
                symmetric=False,
                matvec=self._mult_QU,
                rmatvec=self._rmult_QU,
                dtype=processed_samples.dtype_float,
            )
        else:
            super(PointingLO, self).__init__(
                nargin=self.ncols,
                nargout=self.nrows,
                matvec=self._mult_IQU,
                symmetric=False,
                rmatvec=self._rmult_IQU,
                dtype=processed_samples.dtype_float,
            )

    def _mult_I(self, vec: np.ndarray):
        r"""
        Performs the product of a sparse matrix :math:`Av`,\
         with :math:`v` a  :mod:`numpy`  array (:math:`dim(v)=n_{pix}`)  .

        It extracts the components of :math:`v` corresponding  to the non-null \
        elements of the operator.

        """

        if len(vec) != self.ncols:
            raise ValueError(
                f"Dimensions of `vec` is not compatible with the dimension of this `PointingLO` instance.\nShape of `PointingLO` instance: {self.shape}\nShape of `vec`: {vec.shape}"
            )

        if vec.dtype != self.dtype:
            warnings.warn(
                f"dtype of `vec` will be changed to {self.dtype}",
                TypeChangeWarning,
            )
            vec = vec.astype(dtype=self.dtype, copy=False)

        prod = np.zeros(self.nrows, dtype=self.dtype)

        PointingLO_tools.PLO_mult_I(
            nsamples=self.nrows,
            pointings=self.pointings,
            pointings_flag=self.pointings_flag,
            vec=vec,
            prod=prod,
        )

        return prod

    def _rmult_I(self, vec: np.ndarray):
        r"""
        Performs the product for the transpose operator :math:`A^T`.

        """

        if len(vec) != self.nrows:
            raise ValueError(
                f"Dimensions of `vec` is not compatible with the dimension of this `PointingLO` instance.\nShape of `PointingLO` instance: {self.shape}\nShape of `vec`: {vec.shape}"
            )

        if vec.dtype != self.dtype:
            warnings.warn(
                f"dtype of `vec` will be changed to {self.dtype}",
                TypeChangeWarning,
            )
            vec = vec.astype(dtype=self.dtype, copy=False)

        prod = np.zeros(self.ncols, dtype=self.dtype)

        PointingLO_tools.PLO_rmult_I(
            nsamples=self.nrows,
            pointings=self.pointings,
            pointings_flag=self.pointings_flag,
            vec=vec,
            prod=prod,
        )

        return prod

    def _mult_QU(self, vec: np.ndarray):
        r"""Performs :math:`A * v` with :math:`v` being a *polarization* vector.
        The output array will encode a linear combination of the two Stokes
        parameters,  (whose components are stored contiguously).

        .. math::
            d_t=  Q_p \cos(2\phi_t)+ U_p \sin(2\phi_t).
        """

        if len(vec) != self.ncols:
            raise ValueError(
                f"Dimensions of `vec` is not compatible with the dimension of this `PointingLO` instance.\nShape of `PointingLO` instance: {self.shape}\nShape of `vec`: {vec.shape}"
            )

        if vec.dtype != self.dtype:
            warnings.warn(
                f"dtype of `vec` will be changed to {self.dtype}",
                TypeChangeWarning,
            )
            vec = vec.astype(dtype=self.dtype, copy=False)

        prod = np.zeros(self.nrows, dtype=self.dtype)

        PointingLO_tools.PLO_mult_QU(
            nsamples=self.nrows,
            pointings=self.pointings,
            pointings_flag=self.pointings_flag,
            sin2phi=self.sin2phi,
            cos2phi=self.cos2phi,
            vec=vec,
            prod=prod,
        )

        return prod

    def _rmult_QU(self, vec: np.ndarray):
        r"""
        Performs :math:`A^T * v`. The output vector will be a QU-map-like array.
        """

        if len(vec) != self.nrows:
            raise ValueError(
                f"Dimensions of `vec` is not compatible with the dimension of this `PointingLO` instance.\nShape of `PointingLO` instance: {self.shape}\nShape of `vec`: {vec.shape}"
            )

        if vec.dtype != self.dtype:
            warnings.warn(
                f"dtype of `vec` will be changed to {self.dtype}",
                TypeChangeWarning,
            )
            vec = vec.astype(dtype=self.dtype, copy=False)

        prod = np.zeros(self.ncols, dtype=self.dtype)

        PointingLO_tools.PLO_rmult_QU(
            nsamples=self.nrows,
            pointings=self.pointings,
            pointings_flag=self.pointings_flag,
            sin2phi=self.sin2phi,
            cos2phi=self.cos2phi,
            vec=vec,
            prod=prod,
        )

        return prod

    def _mult_IQU(self, vec: np.ndarray):
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

        if len(vec) != self.ncols:
            raise ValueError(
                f"Dimensions of `vec` is not compatible with the dimension of this `PointingLO` instance.\nShape of `PointingLO` instance: {self.shape}\nShape of `vec`: {vec.shape}"
            )

        if vec.dtype != self.dtype:
            warnings.warn(
                f"dtype of `vec` will be changed to {self.dtype}",
                TypeChangeWarning,
            )
            vec = vec.astype(dtype=self.dtype, copy=False)

        prod = np.zeros(self.nrows, dtype=self.dtype)

        PointingLO_tools.PLO_mult_IQU(
            nsamples=self.nrows,
            pointings=self.pointings,
            pointings_flag=self.pointings_flag,
            sin2phi=self.sin2phi,
            cos2phi=self.cos2phi,
            vec=vec,
            prod=prod,
        )

        return prod

    def _rmult_IQU(self, vec: np.ndarray):
        r"""
        Performs the product for the transpose operator :math:`A^T` to get a IQU map-like vector.
        Since this vector resembles the pixel of 3 maps it has 3 times the size ``Npix``.
        IQU values referring to the same pixel are  contiguously stored in the memory.

        """

        if len(vec) != self.nrows:
            raise ValueError(
                f"Dimensions of `vec` is not compatible with the dimension of this `PointingLO` instance.\nShape of `PointingLO` instance: {self.shape}\nShape of `vec`: {vec.shape}"
            )

        if vec.dtype != self.dtype:
            warnings.warn(
                f"dtype of `vec` will be changed to {self.dtype}",
                TypeChangeWarning,
            )
            vec = vec.astype(dtype=self.dtype, copy=False)

        prod = np.zeros(self.ncols, dtype=self.dtype)

        PointingLO_tools.PLO_rmult_IQU(
            nsamples=self.nrows,
            pointings=self.pointings,
            pointings_flag=self.pointings_flag,
            sin2phi=self.sin2phi,
            cos2phi=self.cos2phi,
            vec=vec,
            prod=prod,
        )

        return prod

    def solver_string(self):
        """
        Return a string depending on the map you are processing
        """
        if self.solver_type == 1:
            return "I"
        elif self.solver_type == 2:
            return "QU"
        else:
            return "IQU"


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


class InvNoiseCovLO_Uncorrelated(lp.LinearOperator):
    """
    Class representing a inverse noise covariance operator for uncorrelated noise.

    A diagonal linear operator defined by its diagonal `diag` (a Numpy array.)
    The type must be specified in the `diag` argument, e.g.,
    `np.ones(5, dtype=np.complex)` or `np.ones(5).astype(np.complex)`.

    """

    def __init__(self, diag: np.ndarray, dtype=None):
        if dtype is not None:
            self.diag = np.asarray(diag, dtype=dtype)
        elif isinstance(diag, np.ndarray):
            dtype = self.diag.dtype
        else:
            dtype = np.float64
            self.diag = np.asarray(diag, dtype=dtype)

        if self.diag.ndim != 1:
            msg = "diag array must be 1-d"
            raise ValueError(msg)

        super(InvNoiseCovLO_Uncorrelated, self).__init__(
            nargin=self.diag.shape[0],
            nargout=self.diag.shape[0],
            symmetric=True,
            matvec=self._mult,
            rmatvec=self._mult,
            dtype=dtype,
        )

    def _mult(self, vec: np.ndarray):
        if len(vec) != self.diag.shape[0]:
            raise ValueError(
                f"Dimensions of `vec` is not compatible with the dimensions of this `InvNoiseCovLO_Uncorrelated` instance.\nShape of `InvNoiseCovLO_Uncorrelated` instance: {self.shape}\nShape of `vec`: {vec.shape}"
            )

        if vec.dtype != self.dtype:
            warnings.warn(
                f"dtype of `vec` will be changed to {self.dtype}",
                TypeChangeWarning,
            )
            vec = vec.astype(dtype=self.dtype, copy=False)

        prod = np.zeros(self.diag.shape[0], dtype=self.dtype)

        InvNoiseCov_tools.uncorrelated_mult(
            nsamples=self.diag.shape[0],
            diag=self.diag,
            vec=vec,
            prod=prod,
        )

        return prod


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


class BlockDiagonalPreconditionerLO(lp.LinearOperator):
    r"""
    Standard preconditioner defined as:

    .. math::

        M_{BD}=( A diag(N^{-1}) A^T)^{-1}

    where :math:`A` is the *pointing matrix* (see  :class:`PointingLO`).
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

    def __init__(self, processed_samples: ProcessTimeSamples):
        self.solver_type = processed_samples.solver_type
        self.new_npix = processed_samples.new_npix
        self.size = processed_samples.new_npix * self.solver_type

        if self.solver_type == 1:
            self.weighted_counts = processed_samples.weighted_counts
        else:
            self.weighted_sin_sq = processed_samples.weighted_sin_sq
            self.weighted_cos_sq = processed_samples.weighted_cos_sq
            self.weighted_sincos = processed_samples.weighted_sincos
            if self.solver_type == 3:
                self.weighted_counts = processed_samples.weighted_counts
                self.weighted_sin = processed_samples.weighted_sin
                self.weighted_cos = processed_samples.weighted_cos

        if self.solver_type == 1:
            super(BlockDiagonalPreconditionerLO, self).__init__(
                nargin=self.size,
                nargout=self.size,
                symmetric=True,
                matvec=self._mult_I,
                dtype=processed_samples.dtype_float,
            )
        elif self.solver_type == 2:
            super(BlockDiagonalPreconditionerLO, self).__init__(
                nargin=self.size,
                nargout=self.size,
                symmetric=True,
                matvec=self._mult_QU,
                dtype=processed_samples.dtype_float,
            )
        else:
            super(BlockDiagonalPreconditionerLO, self).__init__(
                nargin=self.size,
                nargout=self.size,
                symmetric=True,
                matvec=self._mult_IQU,
                dtype=processed_samples.dtype_float,
            )

    def _mult_I(self, vec: np.ndarray):
        r"""
        Action of :math:`y=( A  diag(N^{-1}) A^T)^{-1} x`,
        where :math:`x` is   an :math:`n_{pix}` array.
        """

        if len(vec) != self.size:
            raise ValueError(
                f"Dimenstions of `vec` is not compatible with the dimension of this `BlockDiagonalPreconditionerLO` instance.\nShape of `BlockDiagonalPreconditionerLO` instance: {self.shape}\nShape of `vec`: {vec.shape}"
            )

        if vec.dtype != self.dtype:
            warnings.warn(
                f"dtype of `vec` will be changed to {self.dtype}",
                TypeChangeWarning,
            )
            vec = vec.astype(dtype=self.dtype, copy=False)

        prod = vec / self.weighted_counts

        return prod

    def _mult_QU(self, vec: np.ndarray):
        r"""
        Action of :math:`y=( A  diag(N^{-1}) A^T)^{-1} x`,
        where :math:`x` is   an :math:`n_{pix}` array.
        """

        if len(vec) != self.size:
            raise ValueError(
                f"Dimenstions of `vec` is not compatible with the dimension of this `BlockDiagonalPreconditionerLO` instance.\nShape of `BlockDiagonalPreconditionerLO` instance: {self.shape}\nShape of `vec`: {vec.shape}"
            )

        if vec.dtype != self.dtype:
            warnings.warn(
                f"dtype of `vec` will be changed to {self.dtype}",
                TypeChangeWarning,
            )
            vec = vec.astype(dtype=self.dtype, copy=False)

        prod = np.zeros(self.size, dtype=self.dtype)

        BlkDiagPrecondLO_tools.BDPLO_mult_QU(
            new_npix=self.new_npix,
            weighted_sin_sq=self.weighted_sin_sq,
            weighted_cos_sq=self.weighted_cos_sq,
            weighted_sincos=self.weighted_sincos,
            vec=vec,
            prod=prod,
        )

        return prod

    def _mult_IQU(self, vec: np.ndarray):
        r"""
        Action of :math:`y=( A  diag(N^{-1}) A^T)^{-1} x`,
        where :math:`x` is   an :math:`n_{pix}` array.
        """

        if len(vec) != self.size:
            raise ValueError(
                f"Dimenstions of `vec` is not compatible with the dimension of this `BlockDiagonalPreconditionerLO` instance.\nShape of `BlockDiagonalPreconditionerLO` instance: {self.shape}\nShape of `vec`: {vec.shape}"
            )

        if vec.dtype != self.dtype:
            warnings.warn(
                f"dtype of `vec` will be changed to {self.dtype}",
                TypeChangeWarning,
            )
            vec = vec.astype(dtype=self.dtype, copy=False)

        prod = np.zeros(self.size, dtype=self.dtype)

        BlkDiagPrecondLO_tools.BDPLO_mult_IQU(
            new_npix=self.new_npix,
            weighted_counts=self.weighted_counts,
            weighted_sin_sq=self.weighted_sin_sq,
            weighted_cos_sq=self.weighted_cos_sq,
            weighted_sincos=self.weighted_sincos,
            weighted_sin=self.weighted_sin,
            weighted_cos=self.weighted_cos,
            vec=vec,
            prod=prod,
        )

        return prod

    def solver_string(self):
        """
        Return a string depending on the map you are processing
        """
        if self.solver_type == 1:
            return "I"
        elif self.solver_type == 2:
            return "QU"
        else:
            return "IQU"


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
