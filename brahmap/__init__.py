import mpi4py

mpi4py.rc.initialize = False

from mpi4py import MPI  # noqa: E402

if MPI.Is_initialized() is False:
    MPI.Init_thread(required=MPI.THREAD_FUNNELED)

from . import interfaces, utilities, linop, mapmakers, _extensions  # noqa: E402

from .utilities import Initialize, MPI_RAISE_EXCEPTION  # noqa: E402

bMPI = None

__all__ = [
    "interfaces",
    "utilities",
    "linop",
    "mapmakers",
    "_extensions",
    "Initialize",
    "MPI_RAISE_EXCEPTION",
]
