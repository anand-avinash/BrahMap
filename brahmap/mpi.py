import os
import sys

from mpi4py import MPI
from mpi4py.MPI import Intracomm

import brahmap


class _MPI(object):
    def __init__(
        self,
        comm: Intracomm,
    ) -> None:
        self.update_communicator(comm=comm)

    def update_communicator(self, comm: Intracomm) -> None:
        self.__comm = comm
        self.__size = comm.size
        self.__rank = comm.rank

    @property
    def comm(self) -> Intracomm:
        return self.__comm

    @property
    def size(self) -> int:
        return self.__size

    @property
    def rank(self) -> int:
        return self.__rank

    @property
    def nthreads_per_process(self) -> int:
        value = int(os.environ.get("OMP_NUM_THREADS", 1))
        return value


MPI_UTILS: _MPI = _MPI(comm=MPI.COMM_WORLD)


def Finalize() -> None:
    """A function to be called at the end of execution. Once registered with `atexit`, it will be called automatically at the end. The user doesn't need to call this function explicitly."""
    try:
        MPI.Finalize()
    except Exception as e:
        if brahmap.MPI_UTILS.rank == 0:
            print(f"Caught an exception during MPI finalization: {e}")


sys_excepthook = sys.excepthook


# If errors during a parallel run are not handled properly, they can lead to a deadlock
# as discussed here:
# <https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html#exceptions-and-deadlocks>
# The following exception hook taken from <https://stackoverflow.com/a/34313363>
# solves the problem by flushing stderr before calling `Abort(1)` on global
# communicator effectively aborting the MPI execution environment
def mpi_excepthook(exctype, value, traceback):
    """Ensure the rank that crashes prints its traceback, then kill all processes."""

    sys.stderr.write(
        f"\n*** Exception raised by MPI rank {brahmap.MPI_UTILS.rank} ***\n"
    )
    sys_excepthook(exctype, value, traceback)

    sys.stderr.flush()

    # Force the MPI runtime to abort in order to prevent deadlocks
    MPI.COMM_WORLD.Abort(1)


# Override the default Python exception handler
sys.excepthook = mpi_excepthook
