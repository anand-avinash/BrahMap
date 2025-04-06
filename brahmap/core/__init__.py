from .process_time_samples import SolverType, ProcessTimeSamples

from .linearoperators import (
    PointingLO,
    BlockDiagonalPreconditionerLO,
)

from .noise_operators import ToeplitzLO, InvNoiseCovLO_Uncorrelated, BlockLO

# Imports for type hinting
from ..base import DiagonalOperator
from typing import Union

DTypeNoiseCov = Union[
    DiagonalOperator,
    ToeplitzLO,
    InvNoiseCovLO_Uncorrelated,
    BlockLO,
]

from .GLS import (  # noqa: E402
    GLSParameters,
    GLSResult,
    separate_map_vectors,
    compute_GLS_maps_from_PTS,
    compute_GLS_maps,
)


__all__ = [
    # process_time_samples.py
    "SolverType",
    "ProcessTimeSamples",
    # linearoperators.py
    "PointingLO",
    "BlockDiagonalPreconditionerLO",
    # noise_operators.py
    "ToeplitzLO",
    "InvNoiseCovLO_Uncorrelated",
    "BlockLO",
    # GLS.py
    "GLSParameters",
    "GLSResult",
    "separate_map_vectors",
    "compute_GLS_maps_from_PTS",
    "compute_GLS_maps",
    # NoiseCov dtype
    "DTypeNoiseCov",
]
