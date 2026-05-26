import numpy as np
import numpy.typing as npt
from typing import Literal, List, Any, Callable, cast, Union

from ..base import LinearOperator, BlockDiagonalLinearOperator

from ..math import DTypeFloat


class NoiseCovLinearOperator(LinearOperator):
    """Base class for noise covariance operators

    Parameters
    ----------
    nargin : int
        The number of rows/columns of the operator
    matvec : Callable
        A function that defines the matrix-vector product $x \\mapsto N(x)=Nx$
    input_type : Literal['covariance', 'power_spectrum'], optional
        Specifies whether the input is a covariance array or a power spectrum array,
         by default `"covariance"`
    dtype : DTypeFloat, optional
        The data type of the operator, by default `np.float64`
    **kwargs : Any
        Extra keyword arguments

    Attributes
    ----------
    size : int
        The dimension i.e. the number of rows/columns of the operator
    diag : npt.NDArray[np.number]
        An array containing the diagonal of the operator
    """

    def __init__(
        self,
        nargin: int,
        matvec: Callable,
        input_type: Literal["covariance", "power_spectrum"] = "covariance",
        dtype: DTypeFloat = np.float64,
        **kwargs: Any,
    ) -> None:
        if input_type not in ["covariance", "power_spectrum"]:
            raise ValueError(
                "Please provide only one of `covariance` or `power_spectrum`"
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
        """The dimension i.e. the number of rows/columns of the operator

        Returns
        -------
        int
            The size of the operator
        """
        return self.__size

    @property
    def diag(self) -> npt.NDArray[np.number]:  # type: ignore
        """The diagonal of the operator.

        Returns
        -------
        npt.NDArray[np.number]
            An array containing the diagonal of the operator
        """
        raise NotImplementedError("Please subclass to implement `diag`")

    def get_inverse(self) -> "InvNoiseCovLinearOperator":  # type: ignore
        """Returns the inverse of the operator.

        Returns
        -------
        InvNoiseCovLinearOperator
            The inverse noise covariance operator
        """
        raise NotImplementedError("Please subclass to implement `get_inverse()`")


class InvNoiseCovLinearOperator(NoiseCovLinearOperator):
    """Base class for inverse noise covariance operators

    Parameters
    ----------
    nargin : int
        The number of rows/columns of the operator
    matvec : Callable
        A function that defines the inverse matrix-vector product
        $x \\mapsto N^{-1}(x)=N^{-1}x$
    input_type : Literal['covariance', 'power_spectrum'], optional
        Specifies whether the input is a covariance array or a power spectrum array,
        by default `"covariance"`
    dtype : DTypeFloat, optional
        The data type of the operator, by default `np.float64`
    **kwargs : Any
        Extra keyword arguments
    """

    def __init__(
        self,
        nargin: int,
        matvec: Callable,
        input_type: Literal["covariance", "power_spectrum"] = "covariance",
        dtype: DTypeFloat = np.float64,
        **kwargs: Any,
    ) -> None:
        super(InvNoiseCovLinearOperator, self).__init__(
            nargin,
            matvec,
            input_type,
            dtype,
            **kwargs,
        )


class BaseBlockDiagNoiseCovLinearOperator(BlockDiagonalLinearOperator):
    """Base class for block-diagonal noise covariance operator.

    Parameters
    ----------
    block_list : List[NoiseCovLinearOperator]
        A list of linear operators representing the individual diagonal blocks
    **kwargs : Any
        Extra keyword arguments

    Attributes
    ----------
    size : int
        An array containing the number of rows/columns for each block
    diag : npt.NDArray[np.number]
        An array containing the diagonal of the operator
    """

    def __init__(
        self,
        block_list: List[NoiseCovLinearOperator],
        **kwargs: Any,
    ):
        super(BaseBlockDiagNoiseCovLinearOperator, self).__init__(
            cast(List[LinearOperator], block_list), **kwargs
        )

        if not self.symmetric:
            raise ValueError("The noise (inv-)covariance operators must be symmetric")

    @property
    def size(self) -> int:
        """Array containing the number of rows/columns for each block

        Returns
        -------
        int
            Array containing the number of rows/columns for each block
        """
        return sum(self.col_size)

    @property
    def diag(self) -> npt.NDArray[np.number]:
        """Array containing the diagonal of the operator

        Returns
        -------
        npt.NDArray[np.number]
            Array containing the diagonal of the operator
        """
        diag = np.concatenate(
            [cast(NoiseCovLinearOperator, block).diag for block in self.block_list],
            axis=None,
        )
        return diag

    def get_inverse(self) -> "BaseBlockDiagInvNoiseCovLinearOperator":
        """Returns the inverse block-diagonal covariance operator.

        Returns
        -------
        BaseBlockDiagInvNoiseCovLinearOperator
            The inverse block-diagonal covariance operator
        """
        inverse_list = [
            cast(NoiseCovLinearOperator, block).get_inverse()
            for block in self.block_list
        ]
        return BaseBlockDiagInvNoiseCovLinearOperator(
            block_list=inverse_list,
        )


class BaseBlockDiagInvNoiseCovLinearOperator(BaseBlockDiagNoiseCovLinearOperator):
    """Base class for block-diagonal inverse noise covariance operator.

    Parameters
    ----------
    block_list : List[InvNoiseCovLinearOperator]
        A list of linear operators representing the individual diagonal blocks
    **kwargs : Any
        Extra keyword arguments
    """

    def __init__(
        self,
        block_list: List[InvNoiseCovLinearOperator],
        **kwargs: Any,
    ) -> None:
        super(BaseBlockDiagInvNoiseCovLinearOperator, self).__init__(
            cast(List[NoiseCovLinearOperator], block_list), **kwargs
        )

    def get_inverse(self) -> "BaseBlockDiagNoiseCovLinearOperator":  # type: ignore
        """Returns the block-diagonal covariance operator.

        Returns
        -------
        BaseBlockDiagNoiseCovLinearOperator
            The block-diagonal covariance operator
        """
        inverse_list = [
            cast(InvNoiseCovLinearOperator, block).get_inverse()
            for block in self.block_list
        ]
        return BaseBlockDiagNoiseCovLinearOperator(
            block_list=cast(List[NoiseCovLinearOperator], inverse_list),
        )


DTypeNoiseCov = Union[
    NoiseCovLinearOperator,
    InvNoiseCovLinearOperator,
    BaseBlockDiagNoiseCovLinearOperator,
    BaseBlockDiagInvNoiseCovLinearOperator,
]
