# Original code:
#
# Copyright (c) 2008-2013, Dominique Orban <dominique.orban@gerad.ca>
# All rights reserved.
#
# Copyright (c) 2013-2014, Ghislain Vaillant <ghisvail@gmail.com>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 1. Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# 3. Neither the name of the linop developers nor the names of any contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#
#
# Modified version:
#
# Copyright (c) 2023-present, Avinash Anand <avinash.anand@roma2.infn.it>
# and Giuseppe Puglisi
#
# This file is part of BrahMap.
#
# Licensed under the MIT License. See the <LICENSE.txt> file for details.

import numpy as np
import itertools
import warnings
from typing import List
from functools import reduce, partial

from ..base import BaseLinearOperator, LinearOperator
from ..base import null_log
from ..utilities import ShapeError, TypeChangeWarning
from ..mpi import MPI_RAISE_EXCEPTION

from brahmap import MPI_UTILS


class BlockLinearOperator(LinearOperator):
    """
    A linear operator defined by blocks. Each block must be a linear operator.

    `blocks` should be a list of lists describing the blocks row-wise.
    If there is only one block row, it should be specified as
    `[[b1, b2, ..., bn]]`, not as `[b1, b2, ..., bn]`.

    If the overall linear operator is symmetric, only its upper triangle
    need be specified, e.g., `[[A,B,C], [D,E], [F]]`, and the blocks on the
    diagonal must be square and symmetric.

    """

    def __init__(self, blocks, symmetric=False, **kwargs):
        # If building a symmetric operator, fill in the blanks.
        # They're just references to existing objects.
        try:
            for block_row in blocks:
                for block_col in block_row:
                    __ = block_col.shape
        except (TypeError, AttributeError):
            raise ValueError("blocks should be a nested list of operators")

        if symmetric:
            nrow = len(blocks)
            ncol = len(blocks[0])
            if nrow != ncol:
                raise ShapeError("Inconsistent shape.")

            for block_row in blocks:
                if not block_row[0].symmetric:
                    raise ValueError("Blocks on diagonal must be symmetric.")

            self._blocks = blocks[:]
            for i in range(1, nrow):
                for j in range(i - 1, -1, -1):
                    self._blocks[i].insert(0, self._blocks[j][i].T)

        else:
            self._blocks = blocks

        log = kwargs.get("logger", null_log)
        log.debug("Building new BlockLinearOperator")

        nargins = [[blk.shape[-1] for blk in row] for row in self._blocks]
        log.debug("nargins = " + repr(nargins))
        nargins_by_row = [nargin[0] for nargin in nargins]
        if min(nargins_by_row) != max(nargins_by_row):
            raise ShapeError("Inconsistent block shapes")

        nargouts = [[blk.shape[0] for blk in row] for row in self._blocks]
        log.debug("nargouts = " + repr(nargouts))
        for row in nargouts:
            if min(row) != max(row):
                raise ShapeError("Inconsistent block shapes")

        nargin = sum(nargins[0])
        nargout = sum([out[0] for out in nargouts])

        # Create blocks of transpose operator.
        blocksT = list(map(lambda *row: [blk.T for blk in row], *self._blocks))

        def blk_matvec(x, blks):
            nargins = [[blk.shape[-1] for blk in blkrow] for blkrow in blks]
            nargouts = [[blk.shape[0] for blk in blkrow] for blkrow in blks]
            nargin = sum(nargins[0])
            nargout = sum([out[0] for out in nargouts])
            nx = len(x)
            self.logger.debug("Multiplying with a vector of size %d" % nx)
            self.logger.debug("nargin=%d, nargout=%d" % (nargin, nargout))
            if nx != nargin:
                raise ShapeError("Multiplying with vector of wrong shape.")

            result_type = np.result_type(self.dtype, x.dtype)
            y = np.zeros(nargout, dtype=result_type)

            nblk_row = len(blks)
            nblk_col = len(blks[0])

            row_start = col_start = 0
            for row in range(nblk_row):
                row_end = row_start + nargouts[row][0]
                yout = y[row_start:row_end]
                for col in range(nblk_col):
                    col_end = col_start + nargins[0][col]
                    xin = x[col_start:col_end]
                    B = blks[row][col]
                    yout[:] += B * xin
                    col_start = col_end
                row_start = row_end
                col_start = 0

            return y

        flat_blocks = list(itertools.chain(*blocks))
        blk_dtypes = [blk.dtype for blk in flat_blocks]
        op_dtype = np.result_type(*blk_dtypes)

        super(BlockLinearOperator, self).__init__(
            nargin,
            nargout,
            symmetric=symmetric,
            matvec=lambda x: blk_matvec(x, self._blocks),
            rmatvec=lambda x: blk_matvec(x, blocksT),
            dtype=op_dtype,
            **kwargs,
        )

        self.H._blocks = blocksT

    @property
    def blocks(self):
        """The list of blocks defining the block operator."""
        return self._blocks

    def __getitem__(self, indices):
        blks = np.matrix(self._blocks, dtype=object)[indices]
        # If indexing narrowed it down to a single block, return it.
        if isinstance(blks, BaseLinearOperator):
            return blks
        # Otherwise, we have a matrix of blocks.
        return BlockLinearOperator(blks.tolist(), symmetric=False)

    def __contains__(self, op):
        flat_blocks = list(itertools.chain(*self.blocks))
        return op in flat_blocks

    def __iter__(self):
        for block in self._blocks:
            yield block


class BlockDiagonalLinearOperator(LinearOperator):
    def __init__(self, block_list, **kwargs):
        try:
            for block in block_list:
                __, __ = block.shape
        except (TypeError, AttributeError):
            MPI_RAISE_EXCEPTION(
                condition=True,
                exception=ValueError,
                message="The `block_list` must be a flat list of linear" "operators",
            )

        self.__row_size = np.asarray(
            [block.shape[0] for block in block_list], dtype=int
        )
        self.__col_size = np.asarray(
            [block.shape[-1] for block in block_list], dtype=int
        )

        nargin = sum(self.__col_size)
        nargout = sum(self.__row_size)
        symmetric = reduce(
            lambda x, y: x and y, [block.symmetric for block in block_list]
        )
        dtype = np.result_type(*[block.dtype for block in block_list])

        self.__block_list = block_list

        # transpose operator
        blocks_list_transposed = [block.T for block in block_list]

        matvec = partial(
            self._mult,
            block_list=self.block_list,
            dtype=dtype,
        )
        rmatvec = partial(
            self._mult,
            block_list=blocks_list_transposed,
            dtype=dtype,
        )

        super(BlockDiagonalLinearOperator, self).__init__(
            nargin=nargin,
            nargout=nargout,
            symmetric=symmetric,
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=dtype,
            **kwargs,
        )

    @property
    def block_list(self) -> List:
        return self.__block_list

    @property
    def num_blocks(self) -> int:
        return len(self.block_list)

    @property
    def row_size(self) -> np.ndarray:
        return self.__row_size

    @property
    def col_size(self) -> np.ndarray:
        return self.__col_size

    def __getitem__(self, idx):
        block_range = self.block_list[idx]
        if isinstance(idx, slice):
            return BlockDiagonalLinearOperator(
                block_list=block_range,
            )
        else:
            return block_range

    def _mult(self, vec: np.ndarray, block_list: List, dtype) -> np.ndarray:
        nrows = sum([block.shape[0] for block in block_list])
        ncols = sum([block.shape[1] for block in block_list])
        MPI_RAISE_EXCEPTION(
            condition=(len(vec) != ncols),
            exception=ValueError,
            message=f"Dimensions of `vec` is not compatible with the dimensions of this `BlockDiagonalLinearOperator` instance.\nShape of `BlockDiagonalLinearOperator` instance: ({nrows, ncols})\nShape of `vec`: {vec.shape}",
        )

        if vec.dtype != dtype:
            if MPI_UTILS.rank == 0:
                warnings.warn(
                    f"dtype of `vec` will be changed to {dtype}",
                    TypeChangeWarning,
                )
            vec = vec.astype(dtype=dtype, copy=False)

        prod = np.zeros(nrows, dtype=dtype)

        start_row_idx = 0
        start_col_idx = 0
        for idx, block in enumerate(block_list):
            end_row_idx = start_row_idx + block.shape[0]
            end_col_idx = start_col_idx + block.shape[1]

            prod[start_row_idx:end_row_idx] = block * vec[start_col_idx:end_col_idx]

            start_row_idx = end_row_idx
            start_col_idx = end_col_idx

        return prod


class BlockPreconditioner(BlockLinearOperator):
    """An alias for ``BlockLinearOperator``.

    Holds an additional ``solve`` method equivalent to ``__mul__``.

    """

    def solve(self, x):
        """An alias to __call__."""
        return self.__call__(x)


class BlockDiagonalPreconditioner(BlockDiagonalLinearOperator):
    """
    An alias for ``BlockDiagonalLinearOperator``.

    Holds an additional ``solve`` method equivalent to ``__mul__``.

    """

    def solve(self, x):
        """An alias to __call__."""
        return self.__call__(x)


class BlockHorizontalLinearOperator(BlockLinearOperator):
    """
    A block horizontal linear operator.

    Each block must be a linear operator.
    The blocks must be specified as one list, e.g., `[A, B, C]`.

    """

    def __init__(self, blocks, **kwargs):
        try:
            for block in blocks:
                __ = block.shape
        except (TypeError, AttributeError):
            raise ValueError("blocks should be a flattened list of operators")

        blocks = [[blk for blk in blocks]]

        super(BlockHorizontalLinearOperator, self).__init__(
            blocks=blocks, symmetric=False, **kwargs
        )


class BlockVerticalLinearOperator(BlockLinearOperator):
    """
    A block vertical linear operator.

    Each block must be a linear operator.
    The blocks must be specified as one list, e.g., `[A, B, C]`.

    """

    def __init__(self, blocks, **kwargs):
        try:
            for block in blocks:
                __ = block.shape
        except (TypeError, AttributeError):
            raise ValueError("blocks should be a flattened list of operators")

        blocks = [[blk] for blk in blocks]

        super(BlockVerticalLinearOperator, self).__init__(
            blocks=blocks, symmetric=False, **kwargs
        )


# some shorter aliases
BlockOperator = BlockLinearOperator
BlockDiagonalOperator = BlockDiagonalLinearOperator
BlockHorizontalOperator = BlockHorizontalLinearOperator
BlockVerticalOperator = BlockVerticalLinearOperator
