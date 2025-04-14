from .process_time_samples import SolverType, ProcessTimeSamples

from .linearoperators import (
    PointingLO,
    BlockDiagonalPreconditionerLO,
)

from .noise_ops_diagonal import (
    ToeplitzLO,
    BlockLO,
    InvNoiseCovLO_Diagonal,
    NoiseCovLO_Diagonal,
)

# Imports for type hinting
from ..base import DiagonalOperator
from typing import Union

DTypeNoiseCov = Union[
    DiagonalOperator,
    InvNoiseCovLO_Diagonal,
    ToeplitzLO,
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
    "BlockLO",
    "InvNoiseCovLO_Diagonal",
    "NoiseCovLO_Diagonal",
    # GLS.py
    "GLSParameters",
    "GLSResult",
    "separate_map_vectors",
    "compute_GLS_maps_from_PTS",
    "compute_GLS_maps",
    # NoiseCov dtype
    "DTypeNoiseCov",
]
