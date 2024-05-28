import os
import brahmap

import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.finalize = False

from mpi4py import MPI  # noqa: E402

if MPI.Is_initialized() is False:
    MPI.Init_thread(required=MPI.THREAD_FUNNELED)


def Initialize(communicator=None, raise_exception_per_process: bool = True):
    if brahmap.bMPI is None:
        brahmap.bMPI = _MPI(
            comm=communicator, raise_exception_per_process=raise_exception_per_process
        )


class _MPI(object):
    def __init__(self, comm, raise_exception_per_process: bool) -> None:
        if comm is None:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm
        self.size = self.comm.size
        self.rank = self.comm.rank
        self.raise_exception_per_process = raise_exception_per_process

        if "OMP_NUM_THREADS" in os.environ:
            self.nthreads_per_process = os.environ.get("OMP_NUM_THREADS")
        else:
            self.nthreads_per_process = 1


def MPI_RAISE_EXCEPTION(
    condition,
    exception,
    message,
):
    """Will raise `exception` with `message` if the `condition` is false.

    Args:
        condition (_type_): The condition to be evaluated
        exception (_type_): The exception to throw
        message (_type_): The message to pass to the `Exception`

    Raises:
        exception: _description_
        exception: _description_
    """

    if brahmap.bMPI.raise_exception_per_process:
        if condition is False:
            error_str = f"Exception raised by MPI rank {brahmap.bMPI.rank}\n"
            raise exception(error_str + message)
    else:
        exception_count = brahmap.bMPI.comm.reduce(condition, MPI.SUM, 0)

        if brahmap.bMPI.rank == 0:
            error_str = f"Exception raised by {brahmap.bMPI.comm.size - exception_count} MPI process(es)\n"
            raise exception(error_str + message)
