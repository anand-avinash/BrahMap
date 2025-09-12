import numpy as np
import warnings
from typing import List, Union, Literal, Callable

from ..base import TypeChangeWarning
from ..base import LinearOperator, NoiseCovLinearOperator, InvNoiseCovLinearOperator
from ..math import DTypeFloat, cg
from ..mpi import MPI_RAISE_EXCEPTION
from ..core import InvNoiseCovLO_Circulant

from brahmap import MPI_UTILS


class NoiseCovLO_Toeplitz01(NoiseCovLinearOperator):
    """Linear operator for Toeplitz noise covariance

    The input covariance array must be at least of the size n. The input power
    spectrum array must be of the size 2n-2 or 2n-1.

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
    """Linear operator for the inverse of Toeplitz noise covariance

    The input covariance array must be at least of the size n. The input power
    spectrum array must be of the size 2n-2 or 2n-1.

    Parameters
    ----------
    size : int
        _description_
    input : Union[np.ndarray, List]
        _description_
    input_type : Literal["covariance", "power_spectrum"], optional
        _description_, by default "power_spectrum"
    precond_op : Union[ LinearOperator, Literal[None, "Strang", "TChan", "RChan", "KK2"] ], optional
        _description_, by default None
    precond_maxiter : int, optional
        _description_, by default 50
    precond_atol : float, optional
        _description_, by default 1.0e-10
    precond_callback : Callable, optional
        _description_, by default None
    dtype : DTypeFloat, optional
        _description_, by default np.float64
    """

    def __init__(
        self,
        size: int,
        input: Union[np.ndarray, List],
        input_type: Literal["covariance", "power_spectrum"] = "power_spectrum",
        precond_op: Union[
            LinearOperator, Literal[None, "Strang", "TChan", "RChan", "KK2"]
        ] = None,
        precond_maxiter: int = 50,
        precond_atol: float = 1.0e-10,
        precond_callback: Callable = None,
        dtype: DTypeFloat = np.float64,
    ):
        self.__toeplitz_op = NoiseCovLO_Toeplitz01(
            size=size,
            input=input,
            input_type=input_type,
            dtype=dtype,
        )

        self.precond_atol = precond_atol
        self.precond_maxiter = precond_maxiter
        self.precond_callback = precond_callback

        self.__previous_num_iterations = 0

        super(InvNoiseCovLO_Toeplitz01, self).__init__(
            nargin=size,
            matvec=self._mult,
            input_type=input_type,
            dtype=dtype,
        )

        if precond_op is None:
            self.precond_op = None
        elif isinstance(precond_op, LinearOperator) or isinstance(
            precond_op, np.ndarray
        ):
            self.precond_op = precond_op
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

            self.precond_op = InvNoiseCovLO_Circulant(
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

    @property
    def precond_op(self):
        return self.__precond_op

    @precond_op.setter
    def precond_op(self, operator: LinearOperator):
        if operator is not None:
            MPI_RAISE_EXCEPTION(
                condition=(self.shape != operator.shape),
                exception=ValueError,
                message=f"The shape of the input operator {operator.shape} is not compatible with the shape of inverse Toeplitz operator {self.shape}",
            )
        self.__precond_op = operator

    @property
    def diag(self) -> np.ndarray:
        try:
            diag_arr = getattr(self, "__diag")
        except AttributeError:
            factor = 1.0
            diag_arr = factor * np.ones(self.size, dtype=self.dtype)
        return diag_arr

    @diag.setter
    def diag(self, diag: np.ndarray):
        self.__diag = diag

    @property
    def previous_num_iterations(self) -> int:
        return self.__previous_num_iterations

    def get_inverse(self):
        return self.__toeplitz_op

    def __callback(self, x, r, norm_residual):
        self.__previous_num_iterations += 1
        if self.precond_callback is not None:
            self.precond_callback(x, r, norm_residual)

    def _mult(self, vec: np.ndarray):
        MPI_RAISE_EXCEPTION(
            condition=(len(vec) != self.shape[0]),
            exception=ValueError,
            message=f"Dimensions of `vec` is not compatible with the dimensions of this `InvNoiseCovLO_Toeplitz` instance.\nShape of `InvNoiseCovLO_Toeplitz` instance: {self.shape}\nShape of `vec`: {vec.shape}",
        )

        self.__previous_num_iterations = 0

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
            atol=self.precond_atol,
            maxiter=self.precond_maxiter,
            M=self.precond_op,
            callback=self.__callback,
            parallel=False,
        )

        return prod
