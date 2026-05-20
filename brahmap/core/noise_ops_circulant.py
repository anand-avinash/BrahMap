import numpy as np
import scipy.fft
import warnings
from typing import List, Literal, cast

from ..base import TypeChangeWarning
from ..base import NoiseCovLinearOperator, InvNoiseCovLinearOperator
from ..math import DTypeFloat
from ..mpi import MPI_RAISE_EXCEPTION, MPI_UTILS


class NoiseCovLO_Circulant(NoiseCovLinearOperator):
    """Linear operator for Circulant noise covariance

    Parameters
    ----------
    size : int
        _description_
    input : Union[np.ndarray, List]
        _description_
    input_type : Literal["covariance", "power_spectrum"], optional
        _description_, by default "power_spectrum"
    dtype : DTypeFloat, optional
        _description_, by default np.float64
    """

    def __init__(
        self,
        size: int,
        input: np.ndarray | List,
        input_type: Literal["covariance", "power_spectrum"] = "power_spectrum",
        dtype: DTypeFloat = np.float64,
    ) -> None:
        input = np.asarray(a=input, dtype=dtype)

        MPI_RAISE_EXCEPTION(
            condition=(input.ndim != 1),
            exception=ValueError,
            message="The `input` array must be a 1-d vector",
        )

        if input_type == "covariance":
            MPI_RAISE_EXCEPTION(
                condition=(size != input.shape[0]),
                exception=ValueError,
                message="The input array size must be same as the size of the linear operator",
            )
            self.__input = scipy.fft.rfft(
                input,
                workers=MPI_UTILS.nthreads_per_process,
            ).real.astype(dtype=dtype, copy=False)
        elif input_type == "power_spectrum":
            MPI_RAISE_EXCEPTION(
                condition=(size != input.shape[0] and input.shape[0] != size // 2 + 1),
                exception=ValueError,
                message="The input array size must be same as the size of the linear operator, or exactly half-size (N//2 + 1)",
            )
            self.__input = input[: size // 2 + 1]

        super().__init__(
            nargin=size,
            matvec=self._mult,
            input_type=input_type,
            dtype=dtype,
        )

    @property
    def diag(self) -> np.ndarray:
        if self.size % 2 == 0:
            total_sum = 2 * np.sum(self.__input) - self.__input[0] - self.__input[-1]
        else:
            total_sum = 2 * np.sum(self.__input) - self.__input[0]

        factor = total_sum / self.size
        return factor * np.ones(self.size, dtype=self.dtype)

    def get_inverse(self) -> "InvNoiseCovLO_Circulant":
        inv_noise_cov = InvNoiseCovLO_Circulant(
            size=self.size,
            input=self.__input,
            input_type="power_spectrum",
            dtype=cast(DTypeFloat, self.dtype),
        )
        return inv_noise_cov

    def _mult(self, vec: np.ndarray) -> np.ndarray:
        MPI_RAISE_EXCEPTION(
            condition=(len(vec) != self.shape[0]),
            exception=ValueError,
            message=f"Dimensions of `vec` is not compatible with the dimensions of this `NoiseCovLO_Circulant` instance.\nShape of `NoiseCovLO_Circulant` instance: {self.shape}\nShape of `vec`: {vec.shape}",
        )

        if vec.dtype != self.dtype:
            if MPI_UTILS.rank == 0:
                warnings.warn(
                    f"dtype of `vec` will be changed to {self.dtype}",
                    TypeChangeWarning,
                )
            vec = vec.astype(dtype=self.dtype, copy=False)

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

        return prod.astype(dtype=self.dtype, copy=False)


class InvNoiseCovLO_Circulant(InvNoiseCovLinearOperator):
    """Linear operator for the inverse of Circulant noise covariance

    Parameters
    ----------
    size : int
        _description_
    input : Union[np.ndarray, List]
        _description_
    input_type : Literal["covariance", "power_spectrum"], optional
        _description_, by default "power_spectrum"
    dtype : DTypeFloat, optional
        _description_, by default np.float64
    """

    def __init__(
        self,
        size: int,
        input: np.ndarray | List,
        input_type: Literal["covariance", "power_spectrum"] = "power_spectrum",
        dtype: DTypeFloat = np.float64,
    ) -> None:
        input = np.asarray(a=input, dtype=dtype)

        MPI_RAISE_EXCEPTION(
            condition=(input.ndim != 1),
            exception=ValueError,
            message="The `input` array must be a 1-d vector",
        )

        if input_type == "covariance":
            MPI_RAISE_EXCEPTION(
                condition=(size != input.shape[0]),
                exception=ValueError,
                message="The input array size must be same as the size of the linear operator",
            )
            self.__input = 1.0 / scipy.fft.rfft(
                input,
                workers=MPI_UTILS.nthreads_per_process,
            ).real.astype(dtype=dtype, copy=False)
        elif input_type == "power_spectrum":
            MPI_RAISE_EXCEPTION(
                condition=(size != input.shape[0] and input.shape[0] != size // 2 + 1),
                exception=ValueError,
                message="The input array size must be same as the size of the linear operator, or exactly half-size (N//2 + 1)",
            )
            self.__input = 1.0 / input[: size // 2 + 1]

        super().__init__(
            nargin=size,
            matvec=self._mult,
            input_type=input_type,
            dtype=dtype,
        )

    @property
    def diag(self) -> np.ndarray:
        if self.size % 2 == 0:
            total_sum = 2 * np.sum(self.__input) - self.__input[0] - self.__input[-1]
        else:
            total_sum = 2 * np.sum(self.__input) - self.__input[0]

        factor = total_sum / self.size
        return factor * np.ones(self.size, dtype=self.dtype)

    def get_inverse(self) -> "NoiseCovLO_Circulant":  # type: ignore
        noise_cov = NoiseCovLO_Circulant(
            size=self.size,
            input=1.0 / self.__input,
            input_type="power_spectrum",
            dtype=cast(DTypeFloat, self.dtype),
        )
        return noise_cov

    def _mult(self, vec: np.ndarray) -> np.ndarray:
        MPI_RAISE_EXCEPTION(
            condition=(len(vec) != self.shape[0]),
            exception=ValueError,
            message=f"Dimensions of `vec` is not compatible with the dimensions of this `InvNoiseCovLO_Circulant` instance.\nShape of `InvNoiseCovLO_Circulant` instance: {self.shape}\nShape of `vec`: {vec.shape}",
        )

        if vec.dtype != self.dtype:
            if MPI_UTILS.rank == 0:
                warnings.warn(
                    f"dtype of `vec` will be changed to {self.dtype}",
                    TypeChangeWarning,
                )
            vec = vec.astype(dtype=self.dtype, copy=False)

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

        return prod.astype(dtype=self.dtype, copy=False)
