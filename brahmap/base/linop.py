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


from typing import Callable, Tuple, Any, cast
import numbers
import numpy as np
import numpy.typing as npt
import logging

from .misc import ShapeError


# Default (null) logger.
null_log = logging.getLogger("linop")
null_log.setLevel(logging.WARNING)
null_log.addHandler(logging.NullHandler())


class BaseLinearOperator(object):
    """Base class for defining the common interface shared by all linear
    operators within the BrahMap framework.

    The linear operators abstract the large matrix operations into
    matrix-free functional mappings.

    A linear operator is a linear mapping $x \\mapsto A(x)$ such that
    the size of the input vector $x$ is `nargin` and the size of the
    output vector is `nargout`. The operator $A$ acts equivalently to a
    dense matrix of shape `(nargout, nargin)`, but computes products
    analytically or dynamically to maintain high performance and low
    memory footprints.

    Parameters
    ----------
    nargin : int
        Size of the input vector $x$, i.e. the number of columns of the operator
    nargout : int
        Size of the output vector $A(x)$, i.e. the number of rows of the operator
    symmetric : bool, optional
        A parameter to specify whether the linear operator is symmetric, by
        default `False`
    dtype : npt.DTypeLike, optional
        Data type of the linear operator, by default `np.float64`
    **kwargs : Any
        Extra keywords arguments

    Attributes
    ----------
    dtype : np.dtype
        The data type of the operator
    nargin : int
        Size of the input vector $x$, i.e. the number of columns of the operator
    nargout : int
        Size of the output vector $A(x)$, i.e. the number of rows of the operator
    symmetric : bool
        Indicates whether the operator is symmetric or not
    shape : tuple[int, int]
        A tuple `(nargout, nargin)` representing the shape of the operator
    nMatvec : int
        The number of matrix-vector multiplications computed so far
    logger : logging.Logger
        The logger instance for this operator
    """

    # A logger may be attached to the linear operator via the `logger` keyword
    # argument.

    def __init__(
        self,
        nargin: int,
        nargout: int,
        symmetric: bool = False,
        dtype: npt.DTypeLike = np.float64,
        **kwargs,
    ) -> None:
        self.__nargin = nargin
        self.__nargout = nargout
        self.__symmetric = symmetric
        self.__shape = (nargout, nargin)
        self.dtype = dtype
        self._nMatvec = 0

        # Log activity.
        self.logger = kwargs.get("logger", null_log)
        self.logger.info("New linear operator with shape " + str(self.shape))
        return

    @property
    def nargin(self) -> int:
        """Size of the input vector $x$, i.e. the number of columns of the operator

        Returns
        -------
        int
            The number of input columns
        """
        return self.__nargin

    @property
    def nargout(self) -> int:
        """Size of the output vector $A(x)$, i.e. the number of rows of the operator

        Returns
        -------
        int
            The number of output rows
        """
        return self.__nargout

    @property
    def symmetric(self) -> bool:
        """Indicates whether the operator is symmetric or not

        Returns
        -------
        bool
            `True` if symmetric, `False` otherwise
        """
        return self.__symmetric

    @property
    def shape(self) -> Tuple[int, int]:
        """A tuple `(nargout, nargin)` representing the shape of the operator

        Returns
        -------
        tuple[int, int]
            A tuple `(nrows, ncols)`
        """
        return self.__shape

    @property
    def dtype(self) -> npt.DTypeLike:
        """The data type of the operator.

        Returns
        -------
        npt.DTypeLike
            The NumPy data type of the operator
        """
        return self.__dtype

    @dtype.setter
    def dtype(self, dtype) -> None:
        self.__dtype = dtype

    @property
    def nMatvec(self) -> int:
        """The number of matrix-vector multiplications computed so far

        Returns
        -------
        int
            The number of matrix-vector multiplications performed
        """
        return self._nMatvec

    def reset_counters(self) -> None:
        """Resets matrix-vector product counter to zero."""
        self._nMatvec = 0

    def dot(self, x) -> npt.NDArray[np.number]:
        """Numpy-like dot() method.

        Parameters
        ----------
        x : Any
            The input vector or object to multiply with.
        Returns
        -------
        npt.NDArray[np.number]
            The result of the dot product.
        """
        return self.__mul__(x)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # An alias for __mul__.
        return self.__mul__(*args, **kwargs)

    def __mul__(self, x: Any) -> Any:
        raise NotImplementedError(
            f"{self.__class__.__name__}: Please subclass to implement __mul__."
        )

    def __repr__(self) -> str:
        if self.symmetric:
            s = "Symmetric"
        else:
            s = "Asymmetric"
        s += " <" + self.__class__.__name__ + ">"
        s += " of type %s" % self.dtype
        s += " with shape (%d,%d)" % (self.nargout, self.nargin)
        return s


class LinearOperator(BaseLinearOperator):
    """
    A concrete linear operator constructed from functional mappings.

    This class provides the generic foundation for a linear operator for
    the matrix-vector multiplication operation. It requires a function
    `matvec` for the operation $x \\mapsto A(x)=Ax$, and an optional
    transposed-matrix-vector function `rmatvec` for the operation
    $x \\mapsto A(x)=A^T x$.

    For symmetric operators (like $P^T P$ or block-diagonal
    preconditioners), `rmatvec` is ignored.

    All other keyword arguments are passed directly to the superclass.

    Parameters
    ----------
    nargin : int
        Size of the input vector $x$, i.e. the number of columns of the operator
    nargout : int
        Size of the output vector $A(x)$, i.e. the number of rows of the operator
    matvec : Callable
        A function that defines the matrix-vector product $x \\mapsto A(x)=Ax$
    rmatvec : Optional[Callable], optional
        A function that defines the transposed-matrix-vector product
        $x \\mapsto A(x)=A^T x$, by default `None`
    **kwargs : Any
        Extra keywords arguments

    Attributes
    ----------
    dtype : np.dtype
        The data type of the operator
    nargin : int
        Size of the input vector $x$, i.e. the number of columns of the operator
    nargout : int
        Size of the output vector $A(x)$, i.e. the number of rows of the operator
    symmetric : bool
        Indicates whether the operator is symmetric or not
    shape : tuple[int, int]
        A tuple `(nargout, nargin)` representing the shape of the operator
    nMatvec : int
        The number of matrix-vector multiplications computed so far
    T : LinearOperator
        The transpose of this linear operator
    H : LinearOperator
        The Hermitian adjoint of this linear operator
    logger : logging.Logger
        The logger instance for this operator
    """

    def __init__(
        self,
        nargin: int,
        nargout: int,
        matvec: Callable,
        rmatvec: Callable | None = None,
        **kwargs,
    ) -> None:
        super(LinearOperator, self).__init__(
            nargin,
            nargout,
            **kwargs,
        )
        adjoint_of = kwargs.get("adjoint_of", None) or kwargs.get("transpose_of", None)
        rmatvec = rmatvec or kwargs.get("matvec_transp", None)

        self.__matvec = matvec

        self.__H: LinearOperator | None = None

        if self.symmetric:
            self.__H = self
        else:
            if adjoint_of is None:
                if rmatvec is not None:
                    # Create 'pointer' to transpose operator.
                    self.__H = LinearOperator(
                        nargout,
                        nargin,
                        matvec=rmatvec,
                        rmatvec=matvec,
                        adjoint_of=self,
                        **kwargs,
                    )
                else:
                    self.__H = None
            else:
                # Use operator supplied as transpose operator.
                if isinstance(adjoint_of, LinearOperator):
                    self.__H = adjoint_of
                else:
                    msg = (
                        "kwarg adjoint_of / transpose_of must be of type"
                        " LinearOperator."
                    )
                    msg += " Got " + str(adjoint_of.__class__)
                    msg = f"{self.__class__.__name__}: " + msg
                    raise ValueError(msg)

    @property
    def T(self) -> "LinearOperator":
        """The transpose operator

        Returns
        -------
        LinearOperator
            The transpose of this linear operator
        """
        return cast(LinearOperator, self.__H)

    @property
    def H(self) -> "LinearOperator":
        """The adjoint operator

        Returns
        -------
        LinearOperator
            The Hermitian adjoint of this linear operator
        """
        return cast(LinearOperator, self.__H)

    def matvec(self, x) -> npt.NDArray[np.number]:
        """
        Matrix-vector multiplication method.

        The `matvec` method encapsulates the `matvec`
        routine specified at construct time, to ensure the
        consistency of the input and output arrays with the
        operator's shape.

        Parameters
        ----------
        x : npt.NDArray[np.number]
            The input vector $x$ to be multiplied by the operator

        Returns
        -------
        npt.NDArray[np.number]
            The result of the matrix-vector multiplication $A(x)$
        """
        x = np.asanyarray(x, dtype=self.dtype)
        M, N = self.shape

        # check input data consistency
        N = int(N)
        try:
            x = x.reshape(N)
        except ValueError:
            msg = (
                f"The size of the input array is incompatible with the "
                f"dimensions required by the operator\n"
                f"size of the input array: {x.size}\n"
                f"shape of the operator: {self.shape}"
            )
            msg = f"{self.__class__.__name__}: " + msg
            raise ValueError(msg)

        y = self.__matvec(x)

        # check output data consistency
        M = int(M)
        try:
            y = y.reshape(M)
        except ValueError:
            msg = (
                f"The size of the output array is incompatible with the "
                f"dimensions required by the operator\n"
                f"size of the output array: {y.size}\n"
                f"shape of the operator: {self.shape}"
            )
            msg = f"{self.__class__.__name__}: " + msg
            raise ValueError(msg)

        return y

    def to_array(self) -> npt.NDArray[np.number]:
        """Returns the dense form of the linear operator as a 2D NumPy array.

        !!! Warning

            This method first allocates a NumPy array of shape `self.shape`
            and data-type `self.dtype`, and then fills them with numbers. As
            such, for a large linear operator, it can occupy an enormous
            amount of memory and crash your system. Don't use it unless you
            understand the risk!

        Returns
        -------
        npt.NDArray[np.number]
            The dense 2D array representation of the linear operator
        """
        n, m = self.shape
        H = np.empty((n, m), dtype=self.dtype)
        ej = np.zeros(m, dtype=self.dtype)
        for j in range(m):
            ej[j] = 1.0
            H[:, j] = self * ej
            ej[j] = 0.0
        return H

    def __mul_scalar(self, x) -> "LinearOperator":
        # Product between a linear operator and a scalar
        result_type = np.result_type(self.dtype, type(x))

        if x != 0:

            def matvec(y):
                return x * (self(y))

            def rmatvec(y):
                return x * (self.H(y))

            return LinearOperator(
                self.nargin,
                self.nargout,
                symmetric=self.symmetric,
                matvec=matvec,
                rmatvec=rmatvec,
                dtype=result_type,
            )
        else:
            return ZeroOperator(self.nargin, self.nargout, dtype=result_type)

    def __mul_linop(self, op) -> "LinearOperator":
        # Product between two linear operators
        if self.nargin != op.nargout:
            msg = (
                "Cannot multiply the two operators together\n"
                f"shape of the first operator: {self.shape}\n"
                f"shape of the second operator: {op.shape}"
            )
            msg = f"{self.__class__.__name__}: " + msg
            raise ShapeError(msg)

        def matvec(x):
            return self(op(x))

        def rmatvec(x):
            return op.T(self.H(x))

        result_type = np.result_type(self.dtype, op.dtype)

        return LinearOperator(
            op.nargin,
            self.nargout,
            symmetric=False,  # Generally.
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=result_type,
        )

    def __mul_vector(self, x) -> npt.NDArray[np.number]:
        # Product between a linear operator and a vector
        self._nMatvec += 1
        result_type = np.result_type(self.dtype, x.dtype)
        return self.matvec(x).astype(result_type, copy=False)

    def __mul__(self, x) -> "LinearOperator | npt.NDArray[np.number]":
        # Returns a linear operator if x is a scalar or a linear operator
        # Returns a vector if x is an array
        if isinstance(x, numbers.Number):
            return self.__mul_scalar(x)
        elif isinstance(x, BaseLinearOperator):
            return self.__mul_linop(x)
        elif isinstance(x, np.ndarray):
            return self.__mul_vector(x)
        else:
            raise ValueError(
                f"{self.__class__.__name__}: Invalid multiplier! Cannot multiply"
            )

    def __rmul__(self, x) -> "LinearOperator | npt.NDArray[np.number]":
        if np.isscalar(x):
            return self.__mul__(x)
        raise ValueError(
            f"{self.__class__.__name__}: Invalid operation! Cannot multiply"
        )

    def __add__(self, other) -> "LinearOperator":
        if not isinstance(other, BaseLinearOperator):
            raise ValueError(
                f"{self.__class__.__name__}: Invalid operation! Cannot add"
            )
        if self.shape != other.shape:
            msg = (
                "Cannot add the two operators together\n"
                f"shape of the first operator: {self.shape}\n"
                f"shape of the second operator: {other.shape}"
            )
            msg = f"{self.__class__.__name__}: " + msg
            raise ShapeError(msg)

        other_op = cast(LinearOperator, other)

        def matvec(x):
            return self(x) + other(x)

        def rmatvec(x):
            return self.H(x) + other_op.T(x)

        result_type = np.result_type(self.dtype, other.dtype)

        return LinearOperator(
            self.nargin,
            self.nargout,
            symmetric=self.symmetric and other.symmetric,
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=result_type,
        )

    def __neg__(self) -> "LinearOperator":
        return self * (-1)  # type: ignore

    def __sub__(self, other) -> "LinearOperator":
        if not isinstance(other, BaseLinearOperator):
            raise ValueError(
                f"{self.__class__.__name__}: Invalid operation! Cannot subtract"
            )
        if self.shape != other.shape:
            msg = (
                "Cannot subtract one operator from the other\n"
                f"shape of the first operator: {self.shape}\n"
                f"shape of the second operator: {other.shape}"
            )
            msg = f"{self.__class__.__name__}: " + msg
            raise ShapeError(msg)

        other_op = cast(LinearOperator, other)

        def matvec(x):
            return self(x) - other(x)

        def rmatvec(x):
            return self.H(x) - other_op.T(x)

        result_type = np.result_type(self.dtype, other.dtype)

        return LinearOperator(
            self.nargin,
            self.nargout,
            symmetric=self.symmetric and other.symmetric,
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=result_type,
        )

    def __truediv__(self, other) -> "LinearOperator":
        if isinstance(other, (numbers.Number, np.number)):
            return self * (1.0 / cast(Any, other))
        else:
            raise ValueError(
                f"{self.__class__.__name__}: Invalid operation! Cannot divide"
            )

    def __pow__(self, other) -> "LinearOperator":
        if not isinstance(other, int):
            raise ValueError(
                f"{self.__class__.__name__}: Can only raise to integer power"
            )
        if other < 0:
            raise ValueError(
                f"{self.__class__.__name__}: Can only raise to nonnegative power"
            )
        if self.nargin != self.nargout:
            raise ShapeError(
                f"{self.__class__.__name__}: Can only raise square operators to a power"
            )
        if other == 0:
            return IdentityOperator(self.nargin)
        if other == 1:
            return self
        return self * self ** (other - 1)  # type: ignore


class IdentityOperator(LinearOperator):
    """A linear operator representing an identity mapping of size `nargin`.

    This operator is often used as a placeholder where the output vector
    exactly matches the input vector.

    Parameters
    ----------
    nargin : int
        Size of the input vector i.e. the number of rows/columns of the operator
    **kwargs: Any
        Extra keywords arguments.

    """

    def __init__(self, nargin: int, **kwargs: Any) -> None:
        if "symmetric" in kwargs:
            kwargs.pop("symmetric")
        if "matvec" in kwargs:
            kwargs.pop("matvec")

        super(IdentityOperator, self).__init__(
            nargin, nargin, symmetric=True, matvec=lambda x: x, **kwargs
        )


class DiagonalOperator(LinearOperator):
    """A linear operator representing a diagonal matrix.

    Parameters
    ----------
    diag : npt.NDArray[np.number]
        The diagonal elements of the linear operator or matrix
    **kwargs: Any
        Extra keyword arguments

    """

    def __init__(self, diag: npt.NDArray[np.number], **kwargs: Any) -> None:
        if "symmetric" in kwargs:
            kwargs.pop("symmetric")
        if "matvec" in kwargs:
            kwargs.pop("matvec")
        if "dtype" in kwargs:
            kwargs.pop("dtype")

        self.diag = np.asarray(diag)
        if self.diag.ndim != 1:
            msg = "diag array must be 1-d"
            msg = f"{self.__class__.__name__}: " + msg
            raise ValueError(msg)

        super(DiagonalOperator, self).__init__(
            self.diag.shape[0],
            self.diag.shape[0],
            symmetric=True,
            matvec=lambda x: self.diag * x,
            dtype=self.diag.dtype,
            **kwargs,
        )


class MatrixLinearOperator(LinearOperator):
    """A linear operator wrapping a dense or sparse 2D NumPy/SciPy matrix.

    While BrahMap typically relies on matrix-free operations for large-scale
    data, `MatrixLinearOperator` allows standard explicitly constructed matrices
    (e.g., small covariance blocks or low-resolution masks) to seamlessly
    interact with the iterative solvers and operator algebra in the framework.
    The operator's dtype is the same as the specified `matrix` argument.

    Parameters
    ----------
    matrix : npt.NDArray[np.number]
        A dense 2D matrix to be wrapped as a LinearOperator
    **kwargs: Any
        Extra keyword arguments

    """

    def __init__(self, matrix: npt.NDArray[np.number], **kwargs: Any) -> None:
        if "symmetric" in kwargs:
            kwargs.pop("symmetric")
        if "matvec" in kwargs:
            kwargs.pop("matvec")
        if "dtype" in kwargs:
            kwargs.pop("dtype")

        if not hasattr(matrix, "shape"):
            matrix = np.asanyarray(matrix)

        if matrix.ndim != 2:
            msg = "matrix must be 2-d (shape can be [M, N], [M, 1] or [1, N])"
            msg = f"{self.__class__.__name__}: " + msg
            raise ValueError(msg)

        matvec = matrix.dot
        iscomplex = np.iscomplexobj(matrix)

        if matrix.shape[0] == matrix.shape[1]:
            symmetric = np.all(matrix == matrix.conj().T)
        else:
            symmetric = False

        if not symmetric:
            rmatvec = matrix.conj().T.dot if iscomplex else matrix.T.dot
        else:
            rmatvec = None

        super(MatrixLinearOperator, self).__init__(
            matrix.shape[1],
            matrix.shape[0],
            symmetric=symmetric,
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=matrix.dtype,
            **kwargs,
        )


class ZeroOperator(LinearOperator):
    """A linear operator representing a zero-operator of shape `(nargout, nargin)`.

    This operator always maps the input vector to a vector of zeros.

    Parameters
    ----------
    nargin : int
        Size of the input vector i.e. the number of columns of the operator
    nargout : int
        Size of the output vector i.e. the number of rows of the operator
    **kwargs: Any
        Extra keyword arguments

    """

    def __init__(self, nargin: int, nargout: int, **kwargs: Any) -> None:
        if "matvec" in kwargs:
            kwargs.pop("matvec")
        if "rmatvec" in kwargs:
            kwargs.pop("rmatvec")

        def matvec(x):
            if x.shape != (nargin,):
                msg = "Input has shape " + str(x.shape)
                msg += " instead of (%d,)" % self.nargin
                msg = f"{self.__class__.__name__}: " + msg
                raise ValueError(msg)
            return np.zeros(nargout)

        def rmatvec(x):
            if x.shape != (nargout,):
                msg = "Input has shape " + str(x.shape)
                msg += " instead of (%d,)" % self.nargout
                msg = f"{self.__class__.__name__}: " + msg
                raise ValueError(msg)
            return np.zeros(nargin)

        super(ZeroOperator, self).__init__(
            nargin, nargout, matvec=matvec, rmatvec=rmatvec, **kwargs
        )


class InverseLO(LinearOperator):
    """Constructs the inverse of a linear operator $A$, represented as another linear
    operator.

    This class wraps the inversion operator, applying an iterative
    solver provided with the `method` argument, whenever the inverse
    operator is multiplied with a vector.

    Parameters
    ----------
    A : LinearOperator
        The primary linear operator or matrix
    method : Callable
        The solver method to use (e.g., `cg`, `pcg`)
    preconditioner : LinearOperator | None, optional
        An optional preconditioner operator to accelerate convergence, by default
        `None`
    """

    def __init__(
        self,
        A: LinearOperator,
        method: Callable,
        preconditioner: LinearOperator | None = None,
    ) -> None:
        super(InverseLO, self).__init__(
            nargin=A.shape[0], nargout=A.shape[1], matvec=self.mult, symmetric=True
        )
        self.A = A
        self.__method = method
        self.__preconditioner = preconditioner
        self.__converged = None

    def mult(self, x: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
        """Computes $y = A^{-1}x$ by solving the linear system $Ay = x$ for $y$.

        This method uses the iterative solver routine (e.g., `scipy.sparse.linalg.cg`)
        specified during initialization as `method`.

        Parameters
        ----------
        x : npt.NDArray[np.number]
            The input vector $x$ to be multiplied by the inverse operator

        Returns
        -------
        npt.NDArray[np.number]
            The computed solution vector $y$
        """

        if self.method is None:
            raise ValueError(
                f"{self.__class__.__name__}: InverseLO solver method is not specified."
            )
        y, info = self.method(self.A, x, M=self.preconditioner)
        self.isconverged(info)
        return y

    def isconverged(self, info: int) -> None:
        """Stores the convergence information depending on the exit status of the
        solver.

        Parameters
        ----------
        info : int
            The output status code of the solver method
        """
        self.__converged = info

    @property
    def method(self) -> Callable:
        """The solver method used to compute the inverse of $A$."""
        return self.__method

    @property
    def converged(self) -> int | None:
        """Provides the solver convergence information.

        - `0` : Successful exit
        - `>0` : Convergence to tolerance not achieved, number of iterations
        - `<0` : Illegal input or breakdown
        """
        return self.__converged

    @property
    def preconditioner(self) -> LinearOperator | None:
        """The preconditioner linear operator for the iterative solver."""
        return self.__preconditioner


def ReducedLinearOperator(
    op: LinearOperator, row_indices, col_indices
) -> LinearOperator:
    """
    Restricts a non-symmetric linear operator to a subset of its rows and columns.

    This operation can be useful in masking out unobserved or unwanted
    dimensions by projecting them out of the functional space.

    Parameters
    ----------
    op : LinearOperator
        The original linear operator to be reduced
    row_indices : Sequence[int]
        Indices to restrict the output vector
    col_indices : Sequence[int]
        Indices to restrict the input vector

    Returns
    -------
    LinearOperator
        A new reduced linear operator of shape `(len(row_indices), len(col_indices))`
    """

    nargin, nargout = len(col_indices), len(row_indices)
    m, n = op.shape  # Shape of non-reduced operator.

    def matvec(x):
        z = np.zeros(n, dtype=x.dtype)
        z[col_indices] = x[:]
        y = op * z
        return y[row_indices]  # type: ignore

    def rmatvec(x):
        z = np.zeros(m, dtype=x.dtype)
        z[row_indices] = x[:]
        y = op.H * z
        return y[col_indices]  # type: ignore

    return LinearOperator(
        nargin, nargout, matvec=matvec, symmetric=False, rmatvec=rmatvec
    )


def SymmetricallyReducedLinearOperator(op: LinearOperator, indices):
    """
    Symmetrically restricts a linear operator to a subset of its dimensions.

    This operation is similar to [`ReducedLinearOperator`][..ReducedLinearOperator]
    but it restricts both of the dimensions equally.

    Parameters
    ----------
    op : LinearOperator
        The original linear operator to be reduced
    indices : Sequence[int]
        Indices to restrict both the input and output vectors

    Returns
    -------
    LinearOperator
        A new symmetrically reduced linear operator of shape
        `(len(indices), len(indices))`
    """

    nargin = len(indices)
    m, n = op.shape  # Shape of non-reduced operator.

    def matvec(x):
        z = np.zeros(n, dtype=x.dtype)
        z[indices] = x[:]
        y = op * z
        return y[indices]  # type: ignore

    def rmatvec(x):
        z = np.zeros(m, dtype=x.dtype)
        z[indices] = x[:]
        y = op * z
        return y[indices]  # type: ignore

    return LinearOperator(
        nargin, nargin, matvec=matvec, symmetric=op.symmetric, rmatvec=rmatvec
    )


def aslinearoperator(A) -> LinearOperator:
    """Converts a standard matrix or duck-typed object into a BrahMap `LinearOperator`.

    This function safely coerces various matrix-like objects -- such as
    SciPy sparse matrices, dense NumPy arrays, or custom objects with
    `.shape` and `.matvec` attributes -- into the framework's native
    `LinearOperator` type, ensuring they can participate in algebraic
    expressions (addition, composition) with core map-making operators.

    Parameters
    ----------
    A : Any
        An object that can be interpreted as a linear operator. 'A' may be any of the
        following types:

        - `linop.LinearOperator`
        - `scipy.LinearOperator`
        - `ndarray`
        - `matrix`
        - sparse matrix (e.g. csr_matrix, lil_matrix, etc.)
        - any object with .shape and .matvec attributes

    Returns
    -------
    LinearOperator
        The standard `LinearOperator` wrapping the input `A`
    """
    if isinstance(A, LinearOperator):
        return A

    try:
        import numpy as np

        if isinstance(A, np.ndarray) or isinstance(A, np.matrix):
            return MatrixLinearOperator(A)
    except ImportError:
        pass

    try:
        import scipy.sparse as ssp

        if ssp.isspmatrix(A):
            return MatrixLinearOperator(A)
    except ImportError:
        pass

    if hasattr(A, "shape"):
        nargout, nargin = A.shape
        matvec = None
        rmatvec = None
        dtype = None
        symmetric = False
        if hasattr(A, "matvec"):
            matvec = A.matvec
            if hasattr(A, "rmatvec"):
                rmatvec = A.rmatvec
            elif hasattr(A, "matvec_transp"):
                rmatvec = A.matvec_transp
            if hasattr(A, "dtype"):
                dtype = A.dtype
            if hasattr(A, "symmetric"):
                symmetric = A.symmetric
        elif hasattr(A, "__mul__"):

            def matvec(x):
                return A * x

            if hasattr(A, "__rmul__"):

                def rmatvec(x):
                    return x * A

            if hasattr(A, "dtype"):
                dtype = A.dtype
            try:
                symmetric = A.isSymmetric()
            except Exception:
                symmetric = False
        if matvec is None:
            raise TypeError("unsupported object type: missing matvec or __mul__")
        return LinearOperator(
            nargin,
            nargout,
            symmetric=symmetric,
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=dtype,
        )
    else:
        raise TypeError("unsupported object type")


# some shorter aliases
MatrixOperator = MatrixLinearOperator
aslinop = aslinearoperator
