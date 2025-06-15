import mpi4py
import atexit

from importlib.util import find_spec

mpi4py.rc.initialize = False

from mpi4py import MPI  # noqa: E402

if MPI.Is_initialized() is False:
    MPI.Init_thread(required=MPI.THREAD_FUNNELED)


from .mpi import MPI_UTILS, Finalize, MPI_RAISE_EXCEPTION  # noqa: E402

from . import base, _extensions, core, utilities, math  # noqa: E402

from .core import (  # noqa: E402
    SolverType,
    ProcessTimeSamples,
    PointingLO,
    BlockDiagonalPreconditionerLO,
    NoiseCovLO_Diagonal,
    InvNoiseCovLO_Diagonal,
    NoiseCovLO_Circulant,
    InvNoiseCovLO_Circulant,
    NoiseCovLO_Toeplitz01,
    InvNoiseCovLO_Toeplitz01,
    BlockDiagNoiseCovLO,
    BlockDiagInvNoiseCovLO,
    GLSParameters,
    GLSResult,
    separate_map_vectors,
    compute_GLS_maps_from_PTS,
    compute_GLS_maps,
)

from .utilities import (  # noqa: E402
    TypeChangeWarning,
    LowerTypeCastWarning,
    modify_numpy_context,
    ShapeError,
)

if find_spec("litebird_sim") is not None:
    from . import lbsim
    from .lbsim import (
        LBSimProcessTimeSamples,
        LBSim_InvNoiseCovLO_UnCorr,
        LBSimGLSParameters,
        LBSimGLSResult,
        LBSim_compute_GLS_maps,
    )

    __all__ = [
        "lbsim",
        "LBSimProcessTimeSamples",
        "LBSim_InvNoiseCovLO_UnCorr",
        "LBSimGLSParameters",
        "LBSimGLSResult",
        "LBSim_compute_GLS_maps",
    ]
else:
    __all__ = []


__all__ = __all__ + [
    # ./mpi.py
    "MPI_UTILS",
    "Finalize",
    "MPI_RAISE_EXCEPTION",
    # ./base/
    "base",
    # ./_extensions/
    "_extensions",
    # ./core/
    "core",
    "SolverType",
    "ProcessTimeSamples",
    "PointingLO",
    "BlockDiagonalPreconditionerLO",
    "NoiseCovLO_Diagonal",
    "InvNoiseCovLO_Diagonal",
    "NoiseCovLO_Circulant",
    "InvNoiseCovLO_Circulant",
    "NoiseCovLO_Toeplitz01",
    "InvNoiseCovLO_Toeplitz01",
    "BlockDiagNoiseCovLO",
    "BlockDiagInvNoiseCovLO",
    "GLSParameters",
    "GLSResult",
    "separate_map_vectors",
    "compute_GLS_maps_from_PTS",
    "compute_GLS_maps",
    # ./utilities/
    "utilities",
    "TypeChangeWarning",
    "LowerTypeCastWarning",
    "modify_numpy_context",
    "ShapeError",
    # ./math/
    "math",
]

atexit.register(Finalize)
