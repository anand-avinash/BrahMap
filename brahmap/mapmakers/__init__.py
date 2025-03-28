from .GLS import (
    GLSParameters,
    GLSResult,
    separate_map_vectors,
    compute_GLS_maps_from_PTS,
    compute_GLS_maps,
)

from importlib.util import find_spec

# suggestion taken from: <https://docs.astral.sh/ruff/rules/unused-import/>
if find_spec("litebird_sim") is not None:
    from .lbsim_mapmakers import (
        LBSimGLSParameters,
        LBSimGLSResult,
        LBSim_InvNoiseCovLO_UnCorr,
        LBSimProcessTimeSamples,
        LBSim_compute_GLS_maps,
    )

    __all__ = [
        "LBSimGLSParameters",
        "LBSimGLSResult",
        "LBSim_InvNoiseCovLO_UnCorr",
        "LBSimProcessTimeSamples",
        "LBSim_compute_GLS_maps",
    ]
else:
    __all__ = []

__all__ = __all__ + [
    "GLSParameters",
    "GLSResult",
    "separate_map_vectors",
    "compute_GLS_maps_from_PTS",
    "compute_GLS_maps",
]
