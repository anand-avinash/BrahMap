"""
This module contains the utilities functions strictly related to the computation, and input/output.

"""

from importlib.util import find_spec

from .tools import (
    bash_colors,
    modify_numpy_context,
    profile_run,
    output_profile,
)

if find_spec("matplotlib") is not None:
    from .visualizations import plot_LinearOperator

    __all__ = [
        "plot_LinearOperator",
    ]
else:
    __all__ = []


__all__ = __all__ + [
    # tools.py
    "bash_colors",
    "modify_numpy_context",
    "profile_run",
    "output_profile",
]
