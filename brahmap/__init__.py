import mpi4py
import atexit

mpi4py.rc.initialize = False

from mpi4py import MPI  # noqa: E402

if MPI.Is_initialized() is False:
    MPI.Init_thread(required=MPI.THREAD_FUNNELED)

bMPI = None

from .mpi import Initialize, Finalize, MPI_RAISE_EXCEPTION  # noqa: E402

from . import linop, _extensions, interfaces, utilities, mapmakers  # noqa: E402

__all__ = [
    "bMPI",
    "Initialize",
    "Finalize",
    "MPI_RAISE_EXCEPTION",
    "linop",
    "_extensions",
    "interfaces",
    "utilities",
    "mapmakers",
]

atexit.register(Finalize)
