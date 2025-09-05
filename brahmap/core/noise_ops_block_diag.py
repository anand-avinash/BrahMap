import numpy as np
from typing import List, Union, Literal, Dict, Any

from ..base import (
    BaseBlockDiagNoiseCovLinearOperator,
    BaseBlockDiagInvNoiseCovLinearOperator,  # noqa
)
from ..math import DTypeFloat
from ..mpi import MPI_RAISE_EXCEPTION


class BlockDiagNoiseCovLO(BaseBlockDiagNoiseCovLinearOperator):
    """Linear operator for block-diagonal noise covariance

    Parameters
    ----------
    operator : _type_
        _description_
    block_size : Union[np.ndarray, List]
        _description_
    block_input : Union[List, Dict]
        _description_
    input_type : Literal["covariance", "power_spectrum"], optional
        _description_, by default "power_spectrum"
    dtype : DTypeFloat, optional
        _description_, by default np.float64
    extra_kwargs : Dict[str, Any], optional
        _description_, by default {}
    """

    def __init__(
        self,
        operator,
        block_size: Union[np.ndarray, List],
        block_input: Union[List, Dict],
        input_type: Literal["covariance", "power_spectrum"] = "power_spectrum",
        dtype: DTypeFloat = np.float64,
        extra_kwargs: Dict[str, Any] = {},
    ):
        if isinstance(block_input, list):
            MPI_RAISE_EXCEPTION(
                condition=(len(block_size) != len(block_input)),
                exception=ValueError,
                message="The number of blocks listed in `block_size` is different"
                " from the number of blocks provided in `block_input`",
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
            MPI_RAISE_EXCEPTION(
                condition=True,
                exception=ValueError,
                message="`block_input` must be either a list of arrays or list"
                " OR a dictionary that maps operator size to an array or a list",
            )

        super(BlockDiagNoiseCovLO, self).__init__(
            block_list=block_list,
        )

    def __build_blocks_from_list(
        self,
        operator,
        block_input: List,
        block_size: Union[np.ndarray, List],
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
        block_size: Union[np.ndarray, List],
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
                MPI_RAISE_EXCEPTION(
                    condition=True,
                    exception=ValueError,
                    message=f"Operator for shape {shape} is missing from the input dictionary",
                )

        return block_list


class BlockDiagInvNoiseCovLO(BlockDiagNoiseCovLO):
    """Linear operator for block-diagonal inverse noise covariance

    Parameters
    ----------
    operator : _type_
        _description_
    block_size : Union[np.ndarray, List]
        _description_
    block_input : Union[List, Dict]
        _description_
    input_type : Literal["covariance", "power_spectrum"], optional
        _description_, by default "power_spectrum"
    dtype : DTypeFloat, optional
        _description_, by default np.float64
    extra_kwargs : Dict[str, Any], optional
        _description_, by default {}
    """

    def __init__(
        self,
        operator,
        block_size: Union[np.ndarray, List],
        block_input: Union[List, Dict],
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
