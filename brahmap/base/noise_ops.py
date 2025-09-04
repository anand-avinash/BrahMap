import numpy as np
from typing import Literal, List, Any

from ..base import LinearOperator, BlockDiagonalLinearOperator

from ..math import DTypeFloat

from ..mpi import MPI_RAISE_EXCEPTION


class NoiseCovLinearOperator(LinearOperator):
    """Base class for noise covariance operators

    Parameters
    ----------
    nargin : int
        _description_
    matvec : int
        _description_
    input_type : Literal["covariance", "power_spectrum"], optional
        _description_, by default "covariance"
    dtype : DTypeFloat, optional
        _description_, by default np.float64
    **kwargs: Any
        _description_
    """

    def __init__(
        self,
        nargin: int,
        matvec: int,
        input_type: Literal["covariance", "power_spectrum"] = "covariance",
        dtype: DTypeFloat = np.float64,
        **kwargs: Any,
    ):
        MPI_RAISE_EXCEPTION(
            condition=(input_type not in ["covariance", "power_spectrum"]),
            exception=ValueError,
            message="Please provide only one of `covariance` or `power_spectrum`",
        )

        self.__size = nargin

        super(NoiseCovLinearOperator, self).__init__(
            nargin=nargin,
            nargout=nargin,
            matvec=matvec,
            symmetric=True,
            dtype=dtype,
            **kwargs,
        )

    @property
    def size(self) -> int:
        return self.__size

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
    """Base class for inverse noise covariance operators

    Parameters
    ----------
    nargin : int
        _description_
    matvec : int
        _description_
    input_type : Literal["covariance", "power_spectrum"], optional
        _description_, by default "covariance"
    dtype : DTypeFloat, optional
        _description_, by default np.float64
    **kwargs: Any
        _description_
    """

    def __init__(
        self,
        nargin: int,
        matvec: int,
        input_type: Literal["covariance", "power_spectrum"] = "covariance",
        dtype: DTypeFloat = np.float64,
        **kwargs: Any,
    ):
        super(InvNoiseCovLinearOperator, self).__init__(
            nargin,
            matvec,
            input_type,
            dtype,
            **kwargs,
        )


class BaseBlockDiagNoiseCovLinearOperator(BlockDiagonalLinearOperator):
    """Base class for block-diagonal noise covariance operator

    Parameters
    ----------
    block_list : List[NoiseCovLinearOperator]
        _description_
    **kwargs: Any
        _description_
    """

    def __init__(
        self,
        block_list: List[NoiseCovLinearOperator],
        **kwargs: Any,
    ):
        super(BaseBlockDiagNoiseCovLinearOperator, self).__init__(block_list, **kwargs)

        MPI_RAISE_EXCEPTION(
            condition=(not self.symmetric),
            exception=ValueError,
            message="The noise (inv-)covariance operators must be symmetric",
        )

    @property
    def size(self) -> int:
        return sum(self.col_size)

    @property
    def diag(self) -> np.ndarray:
        diag = np.concatenate(
            [block.diag for block in self.block_list],
            axis=None,
        )
        return diag

    def get_inverse(self) -> "BaseBlockDiagInvNoiseCovLinearOperator":
        inverse_list = [block.get_inverse() for block in self.block_list]
        return BaseBlockDiagInvNoiseCovLinearOperator(block_list=inverse_list)


class BaseBlockDiagInvNoiseCovLinearOperator(BaseBlockDiagNoiseCovLinearOperator):
    """Base class for block-diagonal inverse noise covariance operator

    Parameters
    ----------
    block_list : List[InvNoiseCovLinearOperator]
        _description_
    **kwargs: Any
        _description_
    """

    def __init__(
        self,
        block_list: List[InvNoiseCovLinearOperator],
        **kwargs: Any,
    ):
        super(BaseBlockDiagInvNoiseCovLinearOperator, self).__init__(
            block_list, **kwargs
        )

    def get_inverse(self) -> "BaseBlockDiagNoiseCovLinearOperator":
        inverse_list = [block.get_inverse() for block in self.block_list]
        return BaseBlockDiagNoiseCovLinearOperator(block_list=inverse_list)
