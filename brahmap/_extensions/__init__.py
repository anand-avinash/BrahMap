"""
This module imports only the extension submodules
"""

from . import BlkDiagPrecondLO_tools
from . import compute_weights
from . import PointingLO_tools
from . import repixelize

__all__ = [
    "BlkDiagPrecondLO_tools",
    "compute_weights",
    "PointingLO_tools",
    "repixelize",
]

# stub generation with nanobind
# python -m nanobind.stubgen -m brahmap._extensions.compute_weights -m brahmap._extensions.repixelize -m brahmap._extensions.PointingLO_tools -m brahmap._extensions.BlkDiagPrecondLO_tools -m brahmap.math.linalg_tools -m brahmap.math.unary_functions
