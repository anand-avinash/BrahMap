import numpy as np
import warnings
from typing import List, Union, Literal

from ..base import TypeChangeWarning
from ..base import LinearOperator, NoiseCovLinearOperator, InvNoiseCovLinearOperator
from ..math import DTypeFloat, cg
from ..mpi import MPI_RAISE_EXCEPTION
from ..core import InvNoiseCovLO_Circulant

from brahmap import MPI_UTILS


class NoiseCovLO_Toeplitz01(NoiseCovLinearOperator):
    """The input covariance array must be at least of the size n. The input power spectrum array must be of the size 2n-2 or 2n-1."""

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

        if input_type == "covariance":
            MPI_RAISE_EXCEPTION(
                condition=(size > input.shape[0]),
                exception=ValueError,
                message="The input noise covariance array must be at least of the size of the linear operator",
            )
            covariance = input[:size]
        elif input_type == "power_spectrum":
            MPI_RAISE_EXCEPTION(
                condition=(
                    (2 * size - 1 != input.shape[0])
                    and (2 * size - 2 != input.shape[0])
                ),
                exception=ValueError,
                message="The input power spectrum array must be of the size 2n-2 or 2n-1, where n is the size of the linear operator",
            )
            covariance = np.fft.ifft(input)[:size]
            covariance = covariance.real.astype(dtype=dtype)

        self.__diag_factor = covariance[0]
        self.__input = np.concatenate([covariance, np.roll(covariance[::-1], 1)])
        self.__input = np.fft.fft(self.__input)

        del covariance

        super(NoiseCovLO_Toeplitz01, self).__init__(
            nargin=size,
            matvec=self._mult,
            input_type=input_type,
            dtype=dtype,
        )

    @property
    def diag(self) -> np.ndarray:
        return self.__diag_factor * np.ones(self.size, dtype=self.dtype)

    def get_inverse(self):
        covariance = np.fft.ifft(self.__input)[: self.size]
        covariance = covariance.real.astype(dtype=self.dtype)
        inv_noise_cov = InvNoiseCovLO_Toeplitz01(
            size=self.size,
            input=covariance,
            input_type="covariance",
            dtype=self.dtype,
        )
        return inv_noise_cov

    def _mult(self, vec: np.ndarray):
        MPI_RAISE_EXCEPTION(
            condition=(len(vec) != self.shape[0]),
            exception=ValueError,
            message=f"Dimensions of `vec` is not compatible with the dimensions of this `NoiseCovLO_Toeplitz` instance.\nShape of `NoiseCovLO_Toeplitz` instance: {self.shape}\nShape of `vec`: {vec.shape}",
        )

        if vec.dtype != self.dtype:
            if MPI_UTILS.rank == 0:
                warnings.warn(
                    f"dtype of `vec` will be changed to {self.dtype}",
                    TypeChangeWarning,
                )
            vec = vec.astype(dtype=self.dtype, copy=False)

        prod = np.pad(vec, pad_width=((0, self.size)), mode="constant")

        prod = np.fft.ifft(prod)
        prod = prod * self.__input
        prod = np.fft.fft(prod)[: self.size]

        return prod.real.astype(dtype=self.dtype, copy=False)


class InvNoiseCovLO_Toeplitz01(InvNoiseCovLinearOperator):
    """The input covariance array must be at least of the size n. The input power spectrum array must be of the size 2n-2 or 2n-1."""

    def __init__(
        self,
        size: int,
        input: Union[np.ndarray, List],
        input_type: Literal["covariance", "power_spectrum"] = "power_spectrum",
        precond_op: Union[
            LinearOperator, Literal[None, "Strang", "TChan", "RChan", "KK2"]
        ] = None,
        precond_maxiter=50,
        precond_rtol=1.0e-10,
        precond_atol=1.0e-10,
        precond_callback=None,
        dtype: DTypeFloat = np.float64,
    ):
        self.__toeplitz_op = NoiseCovLO_Toeplitz01(
            size=size,
            input=input,
            input_type=input_type,
            dtype=dtype,
        )

        self.__precond_rtol = precond_rtol
        self.__precond_atol = precond_atol
        self.__precond_maxiter = precond_maxiter
        self.__precond_callback = precond_callback

        if precond_op is None:
            self.__precond_op = None
        elif isinstance(precond_op, LinearOperator) or isinstance(
            precond_op, np.ndarray
        ):
            self.__precond_op = precond_op
        elif precond_op in ["Strang", "TChan", "RChan", "KK2"]:
            if input_type == "power_spectrum":
                cov = np.fft.ifft(input).real[:size]
            else:
                cov = input[:size]

            if precond_op == "Strang":
                temp_size = int(np.floor(cov.size / 2))
                if cov.size % 2 == 0:
                    new_cov = np.concatenate(
                        [cov[:temp_size], cov[1 : temp_size + 1][::-1]]
                    )
                else:
                    new_cov = np.concatenate(
                        [cov[: temp_size + 1], cov[1 : temp_size + 1][::-1]]
                    )
            elif precond_op == "TChan":
                new_cov = np.empty_like(cov)
                new_cov[0] = cov[0]
                n = cov.size
                for idx in range(1, n):
                    new_cov[idx] = ((n - idx) * cov[idx] + idx * cov[n - idx]) / n
            elif precond_op == "RChan":
                new_cov = np.roll(np.flip(cov), 1)
                new_cov += cov
                new_cov[0] = cov[0]
            elif precond_op == "KK2":  # Circulant but not symmetric
                new_cov = np.roll(np.flip(cov), 1)
                new_cov[0] = 0
                new_cov = cov - new_cov

            self.__precond_op = InvNoiseCovLO_Circulant(
                size=size,
                input=new_cov,
                input_type="covariance",
                dtype=dtype,
            )
        else:
            MPI_RAISE_EXCEPTION(
                condition=True,
                exception=ValueError,
                message="Invalid preconditioner operator provided!",
            )

        super(InvNoiseCovLO_Toeplitz01, self).__init__(
            nargin=size,
            matvec=self._mult,
            input_type=input_type,
            dtype=dtype,
        )

    @property
    def diag(self) -> np.ndarray:
        factor = 1.0
        return factor * np.ones(self.size, dtype=self.dtype)

    def get_inverse(self):
        return self.__toeplitz_op

    def _mult(self, vec: np.ndarray):
        MPI_RAISE_EXCEPTION(
            condition=(len(vec) != self.shape[0]),
            exception=ValueError,
            message=f"Dimensions of `vec` is not compatible with the dimensions of this `InvNoiseCovLO_Toeplitz` instance.\nShape of `InvNoiseCovLO_Toeplitz` instance: {self.shape}\nShape of `vec`: {vec.shape}",
        )

        if vec.dtype != self.dtype:
            if MPI_UTILS.rank == 0:
                warnings.warn(
                    f"dtype of `vec` will be changed to {self.dtype}",
                    TypeChangeWarning,
                )
            vec = vec.astype(dtype=self.dtype, copy=False)

        prod, _ = cg(
            A=self.__toeplitz_op,
            b=vec,
            rtol=self.__precond_rtol,
            atol=self.__precond_atol,
            maxiter=self.__precond_maxiter,
            M=self.__precond_op,
            callback=self.__precond_callback,
            parallel=False,
        )

        return prod
