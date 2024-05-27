from .GLS import GLSParameters, GLSResult, compute_GLS_maps

from importlib.util import find_spec

# suggestion taken from: <https://docs.astral.sh/ruff/rules/unused-import/>
if find_spec("litebird_sim") is not None:
    from .lbsim_mapmakers import (
        LBSimGLSParameters,
        LBSimGLSResult,
        LBSim_InvNoiseCovLO_UnCorr,
        LBSim_compute_GLS_maps_from_obs,
        LBSim_compute_GLS_maps,
    )

    __all__ = [
        "LBSimGLSParameters",
        "LBSimGLSResult",
        "LBSim_InvNoiseCovLO_UnCorr",
        "LBSim_compute_GLS_maps_from_obs",
        "LBSim_compute_GLS_maps",
    ]
else:
    __all__ = []

__all__ = __all__ + ["GLSParameters", "GLSResult", "compute_GLS_maps"]
