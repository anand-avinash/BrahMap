import numpy as np
from typing import List, Union

from ..base import LinearOperator, BlockDiagonalLinearOperator

from ..math import DTypeFloat

from ..mpi import MPI_RAISE_EXCEPTION


class NoiseCovLinearOperator(LinearOperator):
    def __init__(
        self,
        nargin: int,
        matvec: int,
        input_type: str = "covariance",
        dtype: DTypeFloat = np.float64,
        **kwargs,
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

        self.size = nargin

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
            message="Please subclass to implement `get_inverse()`",
        )


class InvNoiseCovLinearOperator(NoiseCovLinearOperator):
    def __init__(
        self,
        nargin: int,
        matvec: int,
        input_type: str = "covariance",
        dtype: DTypeFloat = np.float64,
        **kwargs,
    ):
        super(InvNoiseCovLinearOperator, self).__init__(
            nargin,
            matvec,
            input_type,
            dtype,
            **kwargs,
        )


class BlockDiagNoiseCovLinearOperator(BlockDiagonalLinearOperator):
    def __init__(
        self,
        operator,
        block_size: Union[np.ndarray, List],
        block_input: List[Union[np.ndarray, List]],
        input_type: str = "power_spectrum",
        dtype: DTypeFloat = np.float64,
        **kwargs,
    ):
        block_size = np.asarray(block_size, dtype=int)

        MPI_RAISE_EXCEPTION(
            condition=(block_size.shape[0] != len(block_input)),
            exception=ValueError,
            message="The number of blocks listed in `block_size` is different"
            "from the number of blocks listed in `block_input`",
        )

        self.dtype = dtype
        self.__block_size = block_size
        self.size = sum(self.block_size)
        self.__build_blocks(
            operator=operator,
            block_input=block_input,
            input_type=input_type,
            dtype=dtype,
        )

        super(BlockDiagNoiseCovLinearOperator, self).__init__(
            blocks=self.block_list,
            **kwargs,
        )

    @property
    def diag(self) -> np.ndarray:
        return self.__diag

    @property
    def block_size(self) -> np.ndarray:
        return self.__block_size

    @property
    def block_list(self) -> List:
        return self.__block_list

    def get_inverse(self):
        MPI_RAISE_EXCEPTION(
            condition=True,
            exception=NotImplementedError,
            message="Please subclass to implement `get_inverse()`",
        )

    def __build_blocks(self, operator, block_input, input_type, dtype):
        self.__block_list = []
        self.__diag = np.empty(self.size, dtype=dtype)
        start_idx = 0
        for idx, input in enumerate(block_input):
            block_op = operator(
                size=self.block_size[idx],
                input=input,
                input_type=input_type,
                dtype=dtype,
            )
            self.__block_list.append(block_op)
            end_idx = start_idx + self.block_size[idx]
            self.__diag[start_idx:end_idx] = block_op.diag
            start_idx = end_idx


BlockDiagNoiseCovLO = BlockDiagNoiseCovLinearOperator
BlockDiagInvNoiseCovLO = BlockDiagNoiseCovLinearOperator
