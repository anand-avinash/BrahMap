import mpi4py
import atexit

from importlib.util import find_spec

mpi4py.rc.initialize = False

from mpi4py import MPI  # noqa: E402

if MPI.Is_initialized() is False:
    MPI.Init_thread(required=MPI.THREAD_FUNNELED)


from .mpi import MPI_UTILS, Finalize, MPI_RAISE_EXCEPTION  # noqa: E402

from . import linop, _extensions, interfaces, utilities, mapmakers, math  # noqa: E402

from .utilities import SolverType, ProcessTimeSamples  # noqa: E402

from .interfaces import (  # noqa: E402
    PointingLO,
    InvNoiseCovLO_Uncorrelated,
    BlockDiagonalPreconditionerLO,
)

from .mapmakers import (  # noqa: E402
    GLSParameters,
    separate_map_vectors,
    compute_GLS_maps_from_PTS,
    compute_GLS_maps,
)

if find_spec("litebird_sim") is not None:
    from .mapmakers import (
        LBSimGLSParameters,
        LBSim_InvNoiseCovLO_UnCorr,
        LBSimProcessTimeSamples,
        LBSim_compute_GLS_maps,
    )

    __all__ = [
        "LBSimGLSParameters",
        "LBSim_InvNoiseCovLO_UnCorr",
        "LBSimProcessTimeSamples",
        "LBSim_compute_GLS_maps",
    ]
else:
    __all__ = []


__all__ = __all__ + [
    "MPI_UTILS",
    "Finalize",
    "MPI_RAISE_EXCEPTION",
    "linop",
    "_extensions",
    "interfaces",
    "utilities",
    "mapmakers",
    "math",
    "SolverType",
    "ProcessTimeSamples",
    "PointingLO",
    "InvNoiseCovLO_Uncorrelated",
    "BlockDiagonalPreconditionerLO",
    "GLSParameters",
    "separate_map_vectors",
    "compute_GLS_maps_from_PTS",
    "compute_GLS_maps",
]

atexit.register(Finalize)
