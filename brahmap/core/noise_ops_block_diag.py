import numpy as np
from typing import List, Union, Literal, Dict, Any

from ..base import (
    BaseBlockDiagNoiseCovLinearOperator,
    BaseBlockDiagInvNoiseCovLinearOperator,  # noqa
)
from ..math import DTypeFloat
from ..mpi import MPI_RAISE_EXCEPTION


class BlockDiagNoiseCovLO(BaseBlockDiagNoiseCovLinearOperator):
    def __init__(
        self,
        operator,
        block_size: Union[np.ndarray, List],
        block_input: List[Union[np.ndarray, List]],
        input_type: Literal["covariance", "power_spectrum"] = "power_spectrum",
        dtype: DTypeFloat = np.float64,
        extra_kwargs: Dict[str, Any] = {},
    ):
        MPI_RAISE_EXCEPTION(
            condition=(len(block_size) != len(block_input)),
            exception=ValueError,
            message="The number of blocks listed in `block_size` is different"
            "from the number of blocks listed in `block_input`",
        )

        block_list = self.__build_blocks(
            operator=operator,
            block_size=block_size,
            block_input=block_input,
            input_type=input_type,
            dtype=dtype,
            extra_kwargs=extra_kwargs,
        )

        super(BlockDiagNoiseCovLO, self).__init__(
            block_list=block_list,
        )

    def __build_blocks(
        self,
        operator,
        block_input,
        block_size,
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


class BlockDiagInvNoiseCovLO(BlockDiagNoiseCovLO):
    def __init__(
        self,
        operator,
        block_size: Union[np.ndarray, List],
        block_input: List[Union[np.ndarray, List]],
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
