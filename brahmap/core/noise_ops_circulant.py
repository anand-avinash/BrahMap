import numpy as np
import warnings
from typing import List, Union, Literal

from ..utilities import TypeChangeWarning
from ..base import NoiseCovLinearOperator, InvNoiseCovLinearOperator
from ..math import DTypeFloat
from ..mpi import MPI_RAISE_EXCEPTION

from brahmap import MPI_UTILS


class NoiseCovLO_Circulant(NoiseCovLinearOperator):
    def __init__(
        self,
        size: int,
        input: Union[np.ndarray, List],
        input_type: Literal["covariance", "power_spectrum"] = "power_spectrum",
        dtype: DTypeFloat = np.float64,
    ):
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

        if input_type == "covariance":
            self.__input = np.fft.fft(input).real.astype(dtype=dtype, copy=False)
        elif input_type == "power_spectrum":
            self.__input = input

        super(NoiseCovLO_Circulant, self).__init__(
            nargin=size,
            matvec=self._mult,
            input_type=input_type,
            dtype=dtype,
        )

    @property
    def diag(self) -> np.ndarray:
        factor = np.average(self.__input)
        return factor * np.ones(self.size, dtype=self.dtype)

    def get_inverse(self):
        inv_noise_cov = InvNoiseCovLO_Circulant(
            size=self.size,
            input=self.__input,
            input_type="power_spectrum",
            dtype=self.dtype,
        )
        return inv_noise_cov

    def _mult(self, vec: np.ndarray):
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

        prod = np.fft.ifft(vec)
        prod = prod * self.__input
        prod = np.real(np.fft.fft(prod))

        return prod


class InvNoiseCovLO_Circulant(InvNoiseCovLinearOperator):
    def __init__(
        self,
        size: int,
        input: Union[np.ndarray, List],
        input_type: Literal["covariance", "power_spectrum"] = "power_spectrum",
        dtype: DTypeFloat = np.float64,
    ):
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

        if input_type == "covariance":
            self.__input = 1.0 / np.fft.fft(input).real.astype(dtype=dtype, copy=False)
        elif input_type == "power_spectrum":
            self.__input = 1.0 / input

        super(InvNoiseCovLO_Circulant, self).__init__(
            nargin=size,
            matvec=self._mult,
            input_type=input_type,
            dtype=dtype,
        )

    @property
    def diag(self) -> np.ndarray:
        factor = np.average(self.__input)
        return factor * np.ones(self.size, dtype=self.dtype)

    def get_inverse(self):
        noise_cov = NoiseCovLO_Circulant(
            size=self.size,
            input=1.0 / self.__input,
            input_type="power_spectrum",
            dtype=self.dtype,
        )
        return noise_cov

    def _mult(self, vec: np.ndarray):
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

        prod = np.fft.ifft(vec)
        prod = prod * self.__input
        prod = np.real(np.fft.fft(prod))

        return prod
