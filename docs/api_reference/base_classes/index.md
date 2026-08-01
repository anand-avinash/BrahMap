# Base Operators

<!-- markdownlint-disable MD013 -->

## `linop` sub-module — basic linear operators

| Class / Function | Description |
| :--- | :--- |
| [`BaseLinearOperator`](BaseLinearOperator.md) | Abstract base class for all linear operators in the library. |
| [`LinearOperator`](LinearOperator.md) | Class representing a general linear operator defined by matrix-vector multiplication routines. |
| [`IdentityOperator`](IdentityOperator.md) | Linear operator representing an identity matrix. |
| [`DiagonalOperator`](DiagonalOperator.md) | Linear operator representing a diagonal matrix. |
| [`MatrixLinearOperator`](MatrixLinearOperator.md) | Linear operator wrapper around a standard 2D NumPy array. |
| [`ZeroOperator`](ZeroOperator.md) | Linear operator that maps any vector to a zero vector. |
| [`InverseLO`](InverseLO.md) | Linear operator representing the inverse of another linear operator. |
| [`ReducedLinearOperator`](ReducedLinearOperator.md) | Linear operator wrapper that restricts inputs and outputs to a subset of indices. |
| [`SymmetricallyReducedLinearOperator`](SymmetricallyReducedLinearOperator.md) | Reduced linear operator where the row and column restrictions are symmetric. |
| [`aslinearoperator`](aslinearoperator.md) | Helper function to cast a 2D array, matrix, or callable into a `LinearOperator`. |

## `blkop` sub-module — block-linear operators

| Class | Description |
| :--- | :--- |
| [`BlockLinearOperator`](BlockLinearOperator.md) | General block-structured linear operator composed of smaller sub-operators. |
| [`BlockDiagonalLinearOperator`](BlockDiagonalLinearOperator.md) | Block-linear operator with non-zero sub-operators only along the diagonal. |
| [`BlockHorizontalLinearOperator`](BlockHorizontalLinearOperator.md) | Block-linear operator arranged horizontally. |
| [`BlockVerticalLinearOperator`](BlockVerticalLinearOperator.md) | Block-linear operator arranged vertically. |

## Base noise covariance (and inverse) operators

| Class | Description |
| :--- | :--- |
| [`NoiseCovLinearOperator`](NoiseCovLinearOperator.md) | Base class for noise covariance operators. |
| [`InvNoiseCovLinearOperator`](InvNoiseCovLinearOperator.md) | Base class for inverse noise covariance operators. |
| [`BaseBlockDiagNoiseCovLinearOperator`](BaseBlockDiagNoiseCovLinearOperator.md) | Base class for block-diagonal noise covariance operators. |
| [`BaseBlockDiagInvNoiseCovLinearOperator`](BaseBlockDiagInvNoiseCovLinearOperator.md) | Base class for inverse block-diagonal noise covariance operators. |

## Base class for processing the time samples

| Class | Description |
| :--- | :--- |
| [`BaseProcessTimeSamples`](BaseProcessTimeSamples.md) | Abstract base class for processing pointing information and pre-computing map-making weights. |

<!-- markdownlint-enable MD013 -->
