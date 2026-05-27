from numpy import _typing

from .unary_functions import (
    sin,
    cos,
    tan,
    arcsin,
    arccos,
    arctan,
    exp,
    exp2,
    log,
    log2,
    sqrt,
    cbrt,
)

from .linalg_tools import multiply_array

from .linalg import parallel_norm, cg

DTypeFloat = _typing._DTypeLikeFloat
"""Type-hint for the `dtype` of floating-point numbers"""

DTypeInt = _typing._DTypeLikeInt
"""Type-hint for the `dtype` of signed integers"""

DTypeUInit = _typing._DTypeLikeUInt
"""Type-hint for the `dtype` of unsigned integers"""

DTypeBool = _typing._DTypeLikeBool
"""Type-hint for the `dtype` of bools"""

__all__ = [
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "exp",
    "exp2",
    "log",
    "log2",
    "sqrt",
    "cbrt",
    "multiply_array",
    "parallel_norm",
    "cg",
    "DTypeFloat",
    "DTypeInt",
    "DTypeUInit",
    "DTypeBool",
]
