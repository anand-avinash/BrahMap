import numpy as np
import numpy.typing as npt
from typing import List, Literal, Dict, Any

from ..base import (
    LinearOperator,
    BaseBlockDiagNoiseCovLinearOperator,
    BaseBlockDiagInvNoiseCovLinearOperator,  # noqa
)
from ..math import DTypeFloat


class BlockDiagNoiseCovLO(BaseBlockDiagNoiseCovLinearOperator):
    """A linear operator representing a block-diagonal noise covariance matrix $N$.

    Parameters
    ----------
    operator : type
        The base operator for the diagonal blocks
    block_size : npt.NDArray[np.number] | List
        A list defining the sizes of each diagonal block
    block_input : List | Dict
        A list defining the input data for each diagonal block
    input_type : Literal["covariance", "power_spectrum"], optional
        Specifies whether the `input` is a covariance array or a power
        spectrum array, by default `"power_spectrum"`
    dtype : DTypeFloat, optional
        The data type of the operator, by default `np.float64`
    extra_kwargs : Dict[str, Any], optional
        Additional keyword arguments passed to the underlying routines, by default `{}`
    """

    def __init__(
        self,
        operator: type[LinearOperator],
        block_size: npt.NDArray[np.number] | List,
        block_input: List | Dict,
        input_type: Literal["covariance", "power_spectrum"] = "power_spectrum",
        dtype: DTypeFloat = np.float64,
        extra_kwargs: Dict[str, Any] = {},
    ):
        if isinstance(block_input, list):
            if len(block_size) != len(block_input):
                raise ValueError(
                    "The number of blocks listed in `block_size` is different"
                    " from the number of blocks provided in `block_input`"
                )

            block_list = self.__build_blocks_from_list(
                operator=operator,
                block_size=block_size,
                block_input=block_input,
                input_type=input_type,
                dtype=dtype,
                extra_kwargs=extra_kwargs,
            )

        elif isinstance(block_input, dict):
            block_list = self.__build_blocks_from_dict(
                operator=operator,
                block_size=block_size,
                block_input=block_input,
                input_type=input_type,
                dtype=dtype,
                extra_kwargs=extra_kwargs,
            )

        else:
            raise ValueError(
                "`block_input` must be either a list of arrays or list"
                " OR a dictionary that maps operator size to an array or a list"
            )

        super().__init__(
            block_list=block_list,  # type: ignore
        )

    def __build_blocks_from_list(
        self,
        operator,
        block_input: List,
        block_size: npt.NDArray[np.number] | List,
        input_type,
        dtype,
        extra_kwargs,
    ):
        block_list = []
        for idx, input in enumerate(block_input):
            block_op = operator(
                size=block_size[idx],
                input=input,
                input_type=input_type,
                dtype=dtype,
                **extra_kwargs,
            )
            block_list.append(block_op)

        return block_list

    def __build_blocks_from_dict(
        self,
        operator,
        block_input: Dict,
        block_size: npt.NDArray[np.number] | List,
        input_type,
        dtype,
        extra_kwargs,
    ):
        op_dict = {}
        for shape in block_input.keys():
            op_dict[shape] = operator(
                size=shape,
                input=block_input[shape],
                input_type=input_type,
                dtype=dtype,
                **extra_kwargs,
            )

        block_list = []
        for shape in block_size:
            if shape in op_dict.keys():
                block_list.append(op_dict[shape])
            else:
                raise ValueError(
                    f"Operator for shape {shape} is missing from the input dictionary"
                )

        return block_list


class BlockDiagInvNoiseCovLO(BlockDiagNoiseCovLO):
    """A linear operator representing the inverse of a block-diagonal
    noise covariance matrix $N^{-1}$.

    Parameters
    ----------
    operator : type
        The base operator for the diagonal blocks
    block_size : npt.NDArray[np.number] | List
        A list defining the sizes of each diagonal block
    block_input : List | Dict
        A list defining the input data for each diagonal block
    input_type : Literal["covariance", "power_spectrum"], optional
        Specifies whether the `input` is a covariance array or a power
        spectrum array, by default `"power_spectrum"`
    dtype : DTypeFloat, optional
        The data type of the operator, by default `np.float64`
    extra_kwargs : Dict[str, Any], optional
        Additional keyword arguments passed to the underlying routines, by default `{}`
    """

    def __init__(
        self,
        operator: type[LinearOperator],
        block_size: npt.NDArray[np.number] | List,
        block_input: List | Dict,
        input_type: Literal["covariance", "power_spectrum"] = "power_spectrum",
        dtype: DTypeFloat = np.float64,
        extra_kwargs: Dict[str, Any] = {},
    ):
        super(BlockDiagInvNoiseCovLO, self).__init__(
            operator,
            block_size,
            block_input,
            input_type,
            dtype,
            extra_kwargs,
        )
