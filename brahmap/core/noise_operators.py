import numpy as np
import warnings
from typing import List, Union

from ..base import LinearOperator, DiagonalOperator, BlockDiagonalLinearOperator

from ..utilities import TypeChangeWarning

from .._extensions import InvNoiseCov_tools

from ..math import DTypeFloat

from ..mpi import MPI_RAISE_EXCEPTION

from brahmap import MPI_UTILS


class NoiseCovLinearOperator(LinearOperator):
    def __init__(
        self, nargin, matvec, input, input_type="covariance", dtype=np.float64, **kwargs
    ):
        MPI_RAISE_EXCEPTION(
            condition=(input_type not in ["covariance", "power_spectrum"]),
            exception=ValueError,
            message="Please provide only one of `covariance` or `power_spectrum`",
        )

        if input_type == "covariance":
            self.__noise_covariance = input
            self.__power_spectrum = None
        elif input_type == "power_spectrum":
            self.__noise_covariance = None
            self.__power_spectrum = input

        super(NoiseCovLinearOperator, self).__init__(
            nargin=nargin,
            nargout=nargin,
            matvec=matvec,
            symmetric=True,
            dtype=dtype,
            **kwargs,
        )

    @property
    def power_spectrum(self) -> np.ndarray:
        if self.__power_spectrum is None:
            return np.fft.fft(self.__noise_covariance).real.astype(dtype=self.dtype)
        else:
            return self.__power_spectrum

    @property
    def noise_covariance(self) -> np.ndarray:
        if self.__noise_covariance is None:
            return np.fft.ifft(self.__power_spectrum).real.astype(dtype=self.dtype)
        else:
            return self.__noise_covariance

    @property
    def diag(self) -> np.ndarray:
        MPI_RAISE_EXCEPTION(
            condition=True,
            exception=NotImplementedError,
            message="Please subclass to implement `diag`",
        )

    def get_inverse(self) -> "InvNoiseCovLinearOperator":
        MPI_RAISE_EXCEPTION(
            condition=True,
            exception=NotImplementedError,
            message="Please subclass to implement `get_inverse`",
        )


class InvNoiseCovLinearOperator(NoiseCovLinearOperator):
    def __init__(
        self, nargin, matvec, input, input_type="covariance", dtype=np.float64, **kwargs
    ):
        super(InvNoiseCovLinearOperator, self).__init__(
            nargin, matvec, input, input_type, dtype, **kwargs
        )


class InvNoiseCovLO_Diagonal(InvNoiseCovLinearOperator):
    def __init__(
        self,
        size: int,
        input: Union[np.ndarray, List, DTypeFloat] = 1.0,
        input_type="covariance",
        dtype: DTypeFloat = np.float64,
    ):
        if isinstance(input, float):
            input = np.full(shape=size, fill_value=input, dtype=dtype)
        else:
            input = np.asarray(a=input, dtype=dtype)
            MPI_RAISE_EXCEPTION(
                condition=(input.ndim != 1),
                exception=ValueError,
                message="The `input` array must be a 1-d vector",
            )
            MPI_RAISE_EXCEPTION(
                condition=(size != input.shape[0]),
                exception=ValueError,
                message="The input array size must be same as the size of the linear operator",
            )

        super(InvNoiseCovLO_Diagonal, self).__init__(
            nargin=size,
            matvec=self._mult,
            input=input,
            input_type=input_type,
            dtype=dtype,
        )

    @property
    def diag(self) -> np.ndarray:
        return 1.0 / self.noise_covariance

    def get_inverse(self):
        noise_cov = NoiseCovLO_Diagonal(
            size=self.shape[0],
            input=self.noise_covariance,
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

        InvNoiseCov_tools.uncorrelated_mult(
            nsamples=self.shape[0],
            diag=1.0 / self.noise_covariance,
            vec=vec,
            prod=prod,
        )

        return prod


class NoiseCovLO_Diagonal(InvNoiseCovLinearOperator):
    def __init__(
        self,
        size: int,
        input: Union[np.ndarray, List, DTypeFloat] = 1.0,
        input_type="covariance",
        dtype: DTypeFloat = np.float64,
    ):
        if isinstance(input, float):
            input = np.full(shape=size, fill_value=input, dtype=dtype)
        else:
            input = np.asarray(a=input, dtype=dtype)
            MPI_RAISE_EXCEPTION(
                condition=(input.ndim != 1),
                exception=ValueError,
                message="The `input` array must be a 1-d vector",
            )
            MPI_RAISE_EXCEPTION(
                condition=(size != input.shape[0]),
                exception=ValueError,
                message="The input array size must be same as the size of the linear operator",
            )

        super(NoiseCovLO_Diagonal, self).__init__(
            nargin=size,
            matvec=self._mult,
            input=input,
            input_type=input_type,
            dtype=dtype,
        )

    @property
    def diag(self) -> np.ndarray:
        return self.noise_covariance

    def get_inverse(self):
        inv_noise_cov = InvNoiseCovLO_Diagonal(
            size=self.shape[0],
            input=self.noise_covariance,
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

        InvNoiseCov_tools.uncorrelated_mult(
            nsamples=self.shape[0],
            diag=self.noise_covariance,
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
