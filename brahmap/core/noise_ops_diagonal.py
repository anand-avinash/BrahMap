import numpy as np
import warnings
from typing import List, Union

from ..base import LinearOperator, DiagonalOperator, BlockDiagonalLinearOperator

from ..utilities import TypeChangeWarning

from ..math import DTypeFloat, linalg_tools

from ..mpi import MPI_RAISE_EXCEPTION

from ..base import NoiseCovLinearOperator, InvNoiseCovLinearOperator

from brahmap import MPI_UTILS


class NoiseCovLO_Diagonal(NoiseCovLinearOperator):
    def __init__(
        self,
        size: int,
        input: Union[np.ndarray, List, DTypeFloat] = 1.0,
        input_type="covariance",
        dtype: DTypeFloat = np.float64,
    ):
        if isinstance(input, float) and input_type == "covariance":
            self.__noise_covariance = np.full(shape=size, fill_value=input, dtype=dtype)
        elif input_type == "covariance":
            self.__noise_covariance = np.asarray(a=input, dtype=dtype)
        elif input_type == "power_spectrum":
            self.__noise_covariance = np.fft.ifft(input).real.astype(
                dtype=dtype, copy=False
            )

        MPI_RAISE_EXCEPTION(
            condition=(self.__noise_covariance.ndim != 1),
            exception=ValueError,
            message="The `input` array must be a 1-d vector",
        )
        MPI_RAISE_EXCEPTION(
            condition=(size != self.__noise_covariance.shape[0]),
            exception=ValueError,
            message="The input array size must be same as the size of the linear operator",
        )

        super(NoiseCovLO_Diagonal, self).__init__(
            nargin=size,
            matvec=self._mult,
            input_type=input_type,
            dtype=dtype,
        )

    @property
    def diag(self) -> np.ndarray:
        return self.__noise_covariance

    def get_inverse(self):
        inv_noise_cov = InvNoiseCovLO_Diagonal(
            size=self.shape[0],
            input=self.__noise_covariance,
            input_type="covariance",
            dtype=self.dtype,
        )
        return inv_noise_cov

    def _mult(self, vec: np.ndarray):
        MPI_RAISE_EXCEPTION(
            condition=(len(vec) != self.shape[0]),
            exception=ValueError,
            message=f"Dimensions of `vec` is not compatible with the dimensions of this `InvNoiseCovLO_Diagonal` instance.\nShape of `InvNoiseCovLO_Diagonal` instance: {self.shape}\nShape of `vec`: {vec.shape}",
        )

        if vec.dtype != self.dtype:
            if MPI_UTILS.rank == 0:
                warnings.warn(
                    f"dtype of `vec` will be changed to {self.dtype}",
                    TypeChangeWarning,
                )
            vec = vec.astype(dtype=self.dtype, copy=False)

        prod = np.zeros(self.shape[0], dtype=self.dtype)

        linalg_tools.multiply_array(
            nsamples=self.shape[0],
            diag=self.__noise_covariance,
            vec=vec,
            prod=prod,
        )

        return prod


class InvNoiseCovLO_Diagonal(InvNoiseCovLinearOperator):
    def __init__(
        self,
        size: int,
        input: Union[np.ndarray, List, DTypeFloat] = 1.0,
        input_type="covariance",
        dtype: DTypeFloat = np.float64,
    ):
        if isinstance(input, float) and input_type == "covariance":
            self.__inv_noise_cov = np.full(
                shape=size, fill_value=1.0 / input, dtype=dtype
            )
        elif input_type == "covariance":
            self.__inv_noise_cov = 1.0 / np.asarray(a=input, dtype=dtype)
        elif input_type == "power_spectrum":
            self.__inv_noise_cov = 1.0 / np.fft.ifft(input).real.astype(
                dtype=dtype, copy=False
            )

        MPI_RAISE_EXCEPTION(
            condition=(self.__inv_noise_cov.ndim != 1),
            exception=ValueError,
            message="The `input` array must be a 1-d vector",
        )
        MPI_RAISE_EXCEPTION(
            condition=(size != self.__inv_noise_cov.shape[0]),
            exception=ValueError,
            message="The input array size must be same as the size of the linear operator",
        )

        super(InvNoiseCovLO_Diagonal, self).__init__(
            nargin=size,
            matvec=self._mult,
            input_type=input_type,
            dtype=dtype,
        )

    @property
    def diag(self) -> np.ndarray:
        return self.__inv_noise_cov

    def get_inverse(self):
        noise_cov = NoiseCovLO_Diagonal(
            size=self.shape[0],
            input=1.0 / self.__inv_noise_cov,
            input_type="covariance",
            dtype=self.dtype,
        )
        return noise_cov

    def _mult(self, vec: np.ndarray):
        MPI_RAISE_EXCEPTION(
            condition=(len(vec) != self.shape[0]),
            exception=ValueError,
            message=f"Dimensions of `vec` is not compatible with the dimensions of this `InvNoiseCovLO_Diagonal` instance.\nShape of `InvNoiseCovLO_Diagonal` instance: {self.shape}\nShape of `vec`: {vec.shape}",
        )

        if vec.dtype != self.dtype:
            if MPI_UTILS.rank == 0:
                warnings.warn(
                    f"dtype of `vec` will be changed to {self.dtype}",
                    TypeChangeWarning,
                )
            vec = vec.astype(dtype=self.dtype, copy=False)

        prod = np.zeros(self.shape[0], dtype=self.dtype)

        linalg_tools.multiply_array(
            nsamples=self.shape[0],
            diag=self.__inv_noise_cov,
            vec=vec,
            prod=prod,
        )

        return prod


class ToeplitzLO(LinearOperator):
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

    def __init__(self, a, size, dtype=None):
        if dtype is None:
            dtype = np.float64
        else:
            dtype = dtype

        self.__size = size

        super(ToeplitzLO, self).__init__(
            nargin=self.__size,
            nargout=self.__size,
            matvec=self.mult,
            symmetric=True,
            dtype=dtype,
        )
        self.array = a


class BlockLO(BlockDiagonalLinearOperator):
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
                self.blocklist.append(DiagonalOperator(d))

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
