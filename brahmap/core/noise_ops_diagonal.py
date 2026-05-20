import numpy as np
import scipy.fft
from numbers import Number
from typing import List, Literal, cast


from ..math import DTypeFloat, linalg_tools

from ..mpi import MPI_RAISE_EXCEPTION

from ..base import NoiseCovLinearOperator, InvNoiseCovLinearOperator


class NoiseCovLO_Diagonal(NoiseCovLinearOperator):
    """Linear operator for diagonal noise covariance

    Parameters
    ----------
    size : int
        _description_
    input : Union[np.ndarray, List, float], optional
        _description_, by default 1.0
    input_type : Literal["covariance", "power_spectrum"], optional
        _description_, by default "covariance"
    dtype : DTypeFloat, optional
        _description_, by default np.float64
    """

    def __init__(
        self,
        size: int,
        input: np.ndarray | List | float = 1.0,
        input_type: Literal["covariance", "power_spectrum"] = "covariance",
        dtype: DTypeFloat = np.float64,
    ):
        if isinstance(input, Number) and input_type == "covariance":
            self.__noise_covariance = np.full(shape=size, fill_value=input, dtype=dtype)
        elif input_type == "covariance":
            self.__noise_covariance = np.ascontiguousarray(a=input, dtype=dtype)
        elif input_type == "power_spectrum":
            self.__noise_covariance = np.ascontiguousarray(
                scipy.fft.ifft(input).real, dtype=dtype
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

        super().__init__(
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
            dtype=cast(DTypeFloat, self.dtype),
        )
        return inv_noise_cov

    def _mult(self, vec: np.ndarray):
        MPI_RAISE_EXCEPTION(
            condition=(len(vec) != self.shape[0]),
            exception=ValueError,
            message=f"Dimensions of `vec` is not compatible with the dimensions of this `InvNoiseCovLO_Diagonal` instance.\nShape of `InvNoiseCovLO_Diagonal` instance: {self.shape}\nShape of `vec`: {vec.shape}",
        )

        vec = np.ascontiguousarray(vec, dtype=self.dtype)
        prod = np.zeros(self.shape[0], dtype=self.dtype)

        linalg_tools.multiply_array(
            nsamples=self.shape[0],
            diag=self.__noise_covariance,
            vec=vec,
            prod=prod,
        )

        return prod


class InvNoiseCovLO_Diagonal(InvNoiseCovLinearOperator):
    """Linear operator for the inverse of diagonal noise covariance

    Parameters
    ----------
    size : int
        _description_
    input : Union[np.ndarray, List, float], optional
        _description_, by default 1.0
    input_type : Literal["covariance", "power_spectrum"], optional
        _description_, by default "covariance"
    dtype : DTypeFloat, optional
        _description_, by default np.float64
    """

    def __init__(
        self,
        size: int,
        input: np.ndarray | List | float = 1.0,
        input_type: Literal["covariance", "power_spectrum"] = "covariance",
        dtype: DTypeFloat = np.float64,
    ):
        if isinstance(input, Number) and input_type == "covariance":
            self.__inv_noise_cov = np.full(
                shape=size, fill_value=1.0 / input, dtype=dtype
            )
        elif input_type == "covariance":
            self.__inv_noise_cov = np.ascontiguousarray(
                1.0 / np.asarray(a=input, dtype=dtype), dtype=dtype
            )
        elif input_type == "power_spectrum":
            self.__inv_noise_cov = np.ascontiguousarray(
                1.0 / scipy.fft.ifft(input).real, dtype=dtype
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

        super().__init__(
            nargin=size,
            matvec=self._mult,
            input_type=input_type,
            dtype=dtype,
        )

    @property
    def diag(self) -> np.ndarray:
        return self.__inv_noise_cov

    def get_inverse(self):  # type: ignore
        noise_cov = NoiseCovLO_Diagonal(
            size=self.shape[0],
            input=1.0 / self.__inv_noise_cov,
            input_type="covariance",
            dtype=cast(DTypeFloat, self.dtype),
        )
        return noise_cov

    def _mult(self, vec: np.ndarray):
        MPI_RAISE_EXCEPTION(
            condition=(len(vec) != self.shape[0]),
            exception=ValueError,
            message=f"Dimensions of `vec` is not compatible with the dimensions of this `InvNoiseCovLO_Diagonal` instance.\nShape of `InvNoiseCovLO_Diagonal` instance: {self.shape}\nShape of `vec`: {vec.shape}",
        )

        vec = np.ascontiguousarray(vec, dtype=self.dtype)

        prod = np.zeros(self.shape[0], dtype=self.dtype)

        linalg_tools.multiply_array(
            nsamples=self.shape[0],
            diag=self.__inv_noise_cov,
            vec=vec,
            prod=prod,
        )

        return prod
