"""
This module contains the utilities functions strictly related to the computation, and input/output.

"""

from .tools import (
    bash_colors,
    modify_numpy_context,
    TypeChangeWarning,
    LowerTypeCastWarning,
    filter_warnings,
    ShapeError,
    profile_run,
    output_profile,
)

from .visualizations import plot_LinearOperator


__all__ = [
    # tools.py
    "bash_colors",
    "modify_numpy_context",
    "TypeChangeWarning",
    "LowerTypeCastWarning",
    "filter_warnings",
    "ShapeError",
    "profile_run",
    "output_profile",
    # visualizations.py
    "plot_LinearOperator",
]
