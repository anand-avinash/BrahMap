import numpy as np

from ..base import LinearOperator

from ..mpi import MPI_RAISE_EXCEPTION


class NoiseCovLinearOperator(LinearOperator):
    def __init__(
        self, nargin, matvec, input_type="covariance", dtype=np.float64, **kwargs
    ):
        MPI_RAISE_EXCEPTION(
            condition=(input_type not in ["covariance", "power_spectrum"]),
            exception=ValueError,
            message="Please provide only one of `covariance` or `power_spectrum`",
        )

        super(NoiseCovLinearOperator, self).__init__(
            nargin=nargin,
            nargout=nargin,
            matvec=matvec,
            symmetric=True,
            dtype=dtype,
            **kwargs,
        )

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
        self, nargin, matvec, input_type="covariance", dtype=np.float64, **kwargs
    ):
        super(InvNoiseCovLinearOperator, self).__init__(
            nargin, matvec, input_type, dtype, **kwargs
        )
