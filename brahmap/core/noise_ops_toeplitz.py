import numpy as np
import numpy.typing as npt
import scipy.fft
from typing import Literal, Callable, cast

from ..base import LinearOperator, NoiseCovLinearOperator, InvNoiseCovLinearOperator
from ..math import DTypeFloat, cg
from ..mpi import MPI_UTILS
from .noise_ops_circulant import InvNoiseCovLO_Circulant


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
        input: npt.ArrayLike,
        input_type: Literal["covariance", "power_spectrum"] = "power_spectrum",
        dtype: DTypeFloat = np.float64,
    ) -> None:
        input = np.asarray(a=input, dtype=dtype)

        if input.ndim != 1:
            raise ValueError("The `input` array must be a 1-d vector")

        if input_type == "covariance":
            if size > input.shape[0]:
                raise ValueError(
                    "The input noise covariance array must be at least of the size of the linear operator"
                )
            covariance = input[:size]
        elif input_type == "power_spectrum":
            if (2 * size - 1 != input.shape[0]) and (2 * size - 2 != input.shape[0]):
                raise ValueError(
                    "The input power spectrum array must be of the size 2n-2 or 2n-1, where n is the size of the linear operator"
                )
            covariance = scipy.fft.ifft(
                input,
                workers=MPI_UTILS.nthreads_per_process,
            )[:size]
            covariance = covariance.real.astype(  # type: ignore
                dtype=dtype,
                copy=False,
            )

        self.__diag_factor = covariance[0]
        self.__input = np.concatenate([covariance, np.roll(covariance[::-1], 1)])
        self.__input = scipy.fft.rfft(  # type: ignore
            self.__input,
            workers=MPI_UTILS.nthreads_per_process,
        )

        del covariance

        super().__init__(
            nargin=size,
            matvec=self._mult,
            input_type=input_type,
            dtype=dtype,
        )

    @property
    def diag(self) -> npt.NDArray[np.number]:
        return self.__diag_factor * np.ones(self.size, dtype=self.dtype)

    def get_inverse(self) -> "InvNoiseCovLO_Toeplitz01":
        covariance = scipy.fft.irfft(
            self.__input,
            n=2 * self.size,
            workers=MPI_UTILS.nthreads_per_process,
        )[: self.size]
        covariance = covariance.astype(dtype=self.dtype)  # type: ignore
        inv_noise_cov = InvNoiseCovLO_Toeplitz01(
            size=self.size,
            input=covariance,
            input_type="covariance",
            dtype=cast(DTypeFloat, self.dtype),
        )
        return inv_noise_cov

    def _mult(self, vec: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
        prod = np.pad(vec, pad_width=((0, self.size)), mode="constant")

        prod = scipy.fft.rfft(
            prod,
            workers=MPI_UTILS.nthreads_per_process,
        )
        prod = prod * self.__input
        prod = scipy.fft.irfft(
            prod,
            n=2 * self.size,
            workers=MPI_UTILS.nthreads_per_process,
        )[: self.size]

        return prod.astype(dtype=self.dtype, copy=False)  # type: ignore


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
        input: npt.ArrayLike,
        input_type: Literal["covariance", "power_spectrum"] = "power_spectrum",
        precond_op: LinearOperator
        | Literal[None, "Strang", "TChan", "RChan", "KK2"] = None,
        precond_maxiter: int = 50,
        precond_atol: float = 1.0e-10,
        precond_callback: Callable | None = None,
        dtype: DTypeFloat = np.float64,
    ) -> None:
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

        super().__init__(
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
                cov = scipy.fft.ifft(
                    input,
                    workers=MPI_UTILS.nthreads_per_process,
                )[:size]
                cov = cov.real.astype(dtype=dtype, copy=False)  # type: ignore
            else:
                cov = np.asarray(input, dtype=dtype)[:size]

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
            raise ValueError("Invalid preconditioner operator provided!")

    @property
    def precond_op(self) -> LinearOperator | None:
        return self.__precond_op

    @precond_op.setter
    def precond_op(self, operator: LinearOperator | None) -> None:
        if operator is not None:
            if self.shape != operator.shape:
                raise ValueError(
                    f"The shape of the input operator {operator.shape} is not compatible with the shape of inverse Toeplitz operator {self.shape}"
                )
        self.__precond_op = operator

    @property
    def diag(self) -> npt.NDArray[np.number]:
        try:
            diag_arr = getattr(self, "__diag")
        except AttributeError:
            factor = 1.0
            diag_arr = factor * np.ones(self.size, dtype=self.dtype)
        return diag_arr

    @diag.setter
    def diag(self, diag: npt.NDArray[np.number]) -> None:
        self.__diag = diag

    @property
    def previous_num_iterations(self) -> int:
        return self.__previous_num_iterations

    def get_inverse(self) -> "NoiseCovLO_Toeplitz01":  # type: ignore
        return self.__toeplitz_op

    def __callback(self, x, r, norm_residual) -> None:
        self.__previous_num_iterations += 1
        if self.precond_callback is not None:
            self.precond_callback(x, r, norm_residual)

    def _mult(self, vec: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
        self.__previous_num_iterations = 0

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
