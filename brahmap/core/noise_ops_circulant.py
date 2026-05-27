import numpy as np
import numpy.typing as npt
import scipy.fft
from typing import Literal, cast

from ..base import NoiseCovLinearOperator, InvNoiseCovLinearOperator
from ..math import DTypeFloat
from ..mpi import MPI_UTILS


class NoiseCovLO_Circulant(NoiseCovLinearOperator):
    """A linear operator representing a circulant noise covariance matrix $N$.

    Parameters
    ----------
    size : int
        The size (dimension) of the linear operator
    input : npt.ArrayLike
        The input array or data defining the operator
    input_type : Literal["covariance", "power_spectrum"], optional
        Specifies whether the `input` is a covariance array or a power
        spectrum array, by default "power_spectrum"
    dtype : DTypeFloat, optional
        The data type of the operator, by default np.float64
    """

    def __init__(
        self,
        size: int,
        input: npt.ArrayLike,
        input_type: Literal["covariance", "power_spectrum"] = "power_spectrum",
        dtype: DTypeFloat = np.float64,
    ) -> None:
        input = np.asarray(a=input, dtype=dtype)

        if input.ndim != 1:
            raise ValueError("The `input` array must be a 1-d vector")

        if input_type == "covariance":
            if size != input.shape[0]:
                raise ValueError(
                    "The input array size must be same as the size of the linear operator"
                )
            self.__input = scipy.fft.rfft(  # type: ignore
                input,
                workers=MPI_UTILS.nthreads_per_process,
            ).real.astype(dtype=dtype, copy=False)
        elif input_type == "power_spectrum":
            if size != input.shape[0] and input.shape[0] != size // 2 + 1:
                raise ValueError(
                    "The input array size must be same as the size of the linear operator, or exactly half-size (N//2 + 1)"
                )
            self.__input = input[: size // 2 + 1]

        super().__init__(
            nargin=size,
            matvec=self._mult,
            input_type=input_type,
            dtype=dtype,
        )

    @property
    def diag(self) -> npt.NDArray[np.number]:
        """The diagonal elements of the noise covariance operator.

        Returns
        -------
        npt.NDArray[np.number]
            A 1-d array containing the diagonal elements
        """
        if self.size % 2 == 0:
            total_sum = 2 * np.sum(self.__input) - self.__input[0] - self.__input[-1]
        else:
            total_sum = 2 * np.sum(self.__input) - self.__input[0]

        factor = total_sum / self.size
        return factor * np.ones(self.size, dtype=self.dtype)

    def get_inverse(self) -> "InvNoiseCovLO_Circulant":
        """Returns the inverse of this circulant noise covariance operator.

        Returns
        -------
        InvNoiseCovLO_Circulant
            The inverse operator $N^{-1}$
        """
        inv_noise_cov = InvNoiseCovLO_Circulant(
            size=self.size,
            input=self.__input,
            input_type="power_spectrum",
            dtype=cast(DTypeFloat, self.dtype),
        )
        return inv_noise_cov

    def _mult(self, vec: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
        r"""Performs the matrix-vector product $N v$.

        Parameters
        ----------
        vec : npt.NDArray[np.number]
            The input vector $v$

        Returns
        -------
        npt.NDArray[np.number]
            The resulting vector
        """
        prod = scipy.fft.rfft(
            vec,
            workers=MPI_UTILS.nthreads_per_process,
        )
        prod = prod * self.__input
        prod = scipy.fft.irfft(
            prod,
            n=len(vec),
            workers=MPI_UTILS.nthreads_per_process,
        )

        return prod.astype(dtype=self.dtype, copy=False)  # type: ignore


class InvNoiseCovLO_Circulant(InvNoiseCovLinearOperator):
    """A linear operator representing the inverse of a circulant noise
    covariance matrix $N^{-1}$.

    Parameters
    ----------
    size : int
        The size (dimension) of the linear operator
    input : npt.ArrayLike
        The input array or data defining the operator
    input_type : Literal["covariance", "power_spectrum"], optional
        Specifies whether the `input` is a covariance array or a power
        spectrum array, by default "power_spectrum"
    dtype : DTypeFloat, optional
        The data type of the operator, by default np.float64
    """

    def __init__(
        self,
        size: int,
        input: npt.ArrayLike,
        input_type: Literal["covariance", "power_spectrum"] = "power_spectrum",
        dtype: DTypeFloat = np.float64,
    ) -> None:
        input = np.asarray(a=input, dtype=dtype)

        if input.ndim != 1:
            raise ValueError("The `input` array must be a 1-d vector")

        if input_type == "covariance":
            if size != input.shape[0]:
                raise ValueError(
                    "The input array size must be same as the size of the linear operator"
                )
            self.__input = 1.0 / scipy.fft.rfft(  # type: ignore
                input,
                workers=MPI_UTILS.nthreads_per_process,
            ).real.astype(dtype=dtype, copy=False)
        elif input_type == "power_spectrum":
            if size != input.shape[0] and input.shape[0] != size // 2 + 1:
                raise ValueError(
                    "The input array size must be same as the size of the linear operator, or exactly half-size (N//2 + 1)"
                )
            self.__input = 1.0 / input[: size // 2 + 1]

        super().__init__(
            nargin=size,
            matvec=self._mult,
            input_type=input_type,
            dtype=dtype,
        )

    @property
    def diag(self) -> npt.NDArray[np.number]:
        """The diagonal elements of the inverse noise covariance operator.

        Returns
        -------
        npt.NDArray[np.number]
            A 1-d array containing the diagonal elements
        """
        if self.size % 2 == 0:
            total_sum = 2 * np.sum(self.__input) - self.__input[0] - self.__input[-1]
        else:
            total_sum = 2 * np.sum(self.__input) - self.__input[0]

        factor = total_sum / self.size
        return factor * np.ones(self.size, dtype=self.dtype)

    def get_inverse(self) -> "NoiseCovLO_Circulant":  # type: ignore
        """Returns the inverse of this operator, which is the original
        noise covariance operator.

        Returns
        -------
        NoiseCovLO_Circulant
            The noise covariance operator $N$
        """
        noise_cov = NoiseCovLO_Circulant(
            size=self.size,
            input=1.0 / self.__input,
            input_type="power_spectrum",
            dtype=cast(DTypeFloat, self.dtype),
        )
        return noise_cov

    def _mult(self, vec: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
        r"""Performs the matrix-vector product $N^{-1} v$.

        Parameters
        ----------
        vec : npt.NDArray[np.number]
            The input vector $v$

        Returns
        -------
        npt.NDArray[np.number]
            The resulting vector
        """
        prod = scipy.fft.rfft(
            vec,
            workers=MPI_UTILS.nthreads_per_process,
        )
        prod = prod * self.__input
        prod = scipy.fft.irfft(
            prod,
            n=len(vec),
            workers=MPI_UTILS.nthreads_per_process,
        )

        return prod.astype(dtype=self.dtype, copy=False)  # type: ignore
