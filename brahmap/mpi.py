import os
from mpi4py import MPI

import brahmap


def Initialize(communicator=None, raise_exception_per_process: bool = True):
    if brahmap.bMPI is None:
        brahmap.bMPI = _MPI(
            comm=communicator, raise_exception_per_process=raise_exception_per_process
        )


def Finalize():
    """A function to be called at the end of execution. Once registered with `atexit`, it will be called automatically at the end. The user doesn't need to call this function explicitly."""
    try:
        MPI.Finalize()
    except Exception as e:
        if brahmap.bMPI.rank == 0:
            print(f"Caught an exception during MPI finalization: {e}")


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
    """Will raise `exception` with `message` if the `condition` is `True`.

    Args:
        condition (_type_): The condition to be evaluated
        exception (_type_): The exception to throw
        message (_type_): The message to pass to the `Exception`

    Raises:
        exception: _description_
        exception: _description_
    """

    if brahmap.bMPI.raise_exception_per_process:
        if condition is True:
            error_str = f"Exception raised by MPI rank {brahmap.bMPI.rank}\n"
            raise exception(error_str + message)
    else:
        exception_count = brahmap.bMPI.comm.reduce(condition, MPI.SUM, 0)

        if exception_count > 0 and brahmap.bMPI.rank == 0:
            error_str = f"Exception raised by {int(exception_count)} MPI process(es)\n"
            raise exception(error_str + message)
