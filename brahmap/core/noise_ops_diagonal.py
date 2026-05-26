import numpy as np
import numpy.typing as npt
import scipy.fft
from numbers import Number
from typing import Literal, cast


from ..math import DTypeFloat, linalg_tools


from ..base import NoiseCovLinearOperator, InvNoiseCovLinearOperator


class NoiseCovLO_Diagonal(NoiseCovLinearOperator):
    """A linear operator representing a diagonal noise covariance matrix $N$.

    Parameters
    ----------
    size : int
        The size (dimension) of the linear operator
    input : npt.ArrayLike, optional
        The input array or data defining the operator. If `input` is a
        single number, it is taken as a constant variance. By default `1.0`
    input_type : Literal["covariance", "power_spectrum"], optional
        Specifies whether the `input` is a covariance array or a power
        spectrum array, by default `"covariance"`
    dtype : DTypeFloat, optional
        The data type of the operator, by default `np.float64`
    """

    def __init__(
        self,
        size: int,
        input: npt.ArrayLike = 1.0,
        input_type: Literal["covariance", "power_spectrum"] = "covariance",
        dtype: DTypeFloat = np.float64,
    ) -> None:
        if isinstance(input, Number) and input_type == "covariance":
            self.__noise_covariance = np.full(
                shape=size,
                fill_value=input,
                dtype=dtype,
            )
        elif input_type == "covariance":
            self.__noise_covariance = np.ascontiguousarray(a=input, dtype=dtype)
        elif input_type == "power_spectrum":
            self.__noise_covariance = np.ascontiguousarray(
                scipy.fft.ifft(input).real,  # type: ignore
                dtype=dtype,
            )

        if self.__noise_covariance.ndim != 1:
            raise ValueError("The `input` array must be a 1-d vector")
        if size != self.__noise_covariance.shape[0]:
            raise ValueError(
                "The input array size must be same as the size of the linear operator"
            )

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
        return self.__noise_covariance

    def get_inverse(self) -> "InvNoiseCovLO_Diagonal":
        """Returns the inverse of this diagonal noise covariance operator.

        Returns
        -------
        InvNoiseCovLO_Diagonal
            The inverse operator $N^{-1}$
        """
        inv_noise_cov = InvNoiseCovLO_Diagonal(
            size=self.shape[0],
            input=self.__noise_covariance,
            input_type="covariance",
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
    """A linear operator representing the inverse of a diagonal noise
    covariance matrix $N^{-1}$.

    Parameters
    ----------
    size : int
        The size (dimension) of the linear operator
    input : npt.ArrayLike, optional
        The input array or data defining the operator. If `input` is a
        single number, it is taken as a constant variance. By default `1.0`
    input_type : Literal["covariance", "power_spectrum"], optional
        Specifies whether the `input` is a covariance array or a power
        spectrum array, by default `"covariance"`
    dtype : DTypeFloat, optional
        The data type of the operator, by default `np.float64`
    """

    def __init__(
        self,
        size: int,
        input: npt.ArrayLike = 1.0,
        input_type: Literal["covariance", "power_spectrum"] = "covariance",
        dtype: DTypeFloat = np.float64,
    ) -> None:
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
                1.0 / scipy.fft.ifft(input).real,  # type: ignore
                dtype=dtype,
            )

        if self.__inv_noise_cov.ndim != 1:
            raise ValueError("The `input` array must be a 1-d vector")
        if size != self.__inv_noise_cov.shape[0]:
            raise ValueError(
                "The input array size must be same as the size of the linear operator"
            )

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
        return self.__inv_noise_cov

    def get_inverse(self) -> "NoiseCovLO_Diagonal":  # type: ignore
        """Returns the inverse of this operator, which is the original
        noise covariance operator.

        Returns
        -------
        NoiseCovLO_Diagonal
            The noise covariance operator $N$
        """
        noise_cov = NoiseCovLO_Diagonal(
            size=self.shape[0],
            input=1.0 / self.__inv_noise_cov,
            input_type="covariance",
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
        vec = np.ascontiguousarray(vec, dtype=self.dtype)

        prod = np.zeros(self.shape[0], dtype=self.dtype)

        linalg_tools.multiply_array(
            nsamples=self.shape[0],
            diag=self.__inv_noise_cov,
            vec=vec,
            prod=prod,
        )

        return prod
