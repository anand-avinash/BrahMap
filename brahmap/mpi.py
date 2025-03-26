import os

from mpi4py import MPI
from mpi4py.MPI import Intracomm

import brahmap


class _MPI(object):
    def __init__(
        self,
        comm: Intracomm,
        raise_exception_per_process: bool,
    ) -> None:
        self.update_communicator(comm=comm)
        self.raise_exception_per_process = raise_exception_per_process

    def update_communicator(self, comm: Intracomm) -> None:
        self.__comm = comm
        self.__size = comm.size
        self.__rank = comm.rank

    @property
    def comm(self):
        return self.__comm

    @property
    def size(self):
        return self.__size

    @property
    def rank(self):
        return self.__rank

    @property
    def nthreads_per_process(self):
        if "OMP_NUM_THREADS" in os.environ:
            value = os.environ.get("OMP_NUM_THREADS")
        else:
            value = 1
        return value


MPI_UTILS: _MPI = _MPI(comm=MPI.COMM_WORLD, raise_exception_per_process=True)


def Finalize() -> None:
    """A function to be called at the end of execution. Once registered with `atexit`, it will be called automatically at the end. The user doesn't need to call this function explicitly."""
    try:
        MPI.Finalize()
    except Exception as e:
        if brahmap.MPI_UTILS.rank == 0:
            print(f"Caught an exception during MPI finalization: {e}")


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

    if brahmap.MPI_UTILS.raise_exception_per_process:
        if condition is True:
            error_str = f"Exception raised by MPI rank {brahmap.MPI_UTILS.rank}\n"
            raise exception(error_str + message)
    else:
        exception_count = brahmap.MPI_UTILS.comm.reduce(condition, MPI.SUM, 0)

        if exception_count > 0 and brahmap.MPI_UTILS.rank == 0:
            error_str = f"Exception raised by {int(exception_count)} MPI process(es)\n"
            raise exception(error_str + message)
