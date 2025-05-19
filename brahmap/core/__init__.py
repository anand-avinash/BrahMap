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

from .noise_ops_circulant import (
    NoiseCovLO_Circulant,
    InvNoiseCovLO_Circulant,
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
    # noise_ops_diagonal.py
    "ToeplitzLO",
    "BlockLO",
    "InvNoiseCovLO_Diagonal",
    "NoiseCovLO_Diagonal",
    # noise_ops_circulant.py
    "NoiseCovLO_Circulant",
    "InvNoiseCovLO_Circulant",
    # GLS.py
    "GLSParameters",
    "GLSResult",
    "separate_map_vectors",
    "compute_GLS_maps_from_PTS",
    "compute_GLS_maps",
    # NoiseCov dtype
    "DTypeNoiseCov",
]
