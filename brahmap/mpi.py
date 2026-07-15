import os
import sys
from typing import cast

from mpi4py import MPI
from mpi4py.MPI import Intracomm, Comm

import numpy as np
import numpy.typing as npt


class SharedMemoryManager(object):
    def __init__(
        self,
        base_comm: Intracomm,
        nproc_reduce: int = 1,
        node_root: int = 0,
    ) -> None:
        self._base_comm = base_comm
        self._node_comm: Intracomm = cast(
            Intracomm, self._base_comm.Split_type(MPI.COMM_TYPE_SHARED)
        )
        self._node_rank = self._node_comm.rank
        self._node_size = self._node_comm.size
        self._nproc_reduce = min(max(nproc_reduce, 1), self._node_size)
        self._node_root = node_root

        # node root communicator: a communicator that contains the
        # root/master/leaders of all node communicators
        root_color = 0 if self._node_rank == self._node_root else MPI.UNDEFINED
        self._node_root_comm: Comm | None = self._base_comm.Split(
            root_color, self._base_comm.rank
        )

        # This is block grouping, so `--map-by core` option would be most optimal
        tree_grp_color = self._node_rank // self._nproc_reduce
        self._tree_grp_comm = self._node_comm.Split(
            color=tree_grp_color, key=self._node_rank
        )
        self._tree_grp_rank = self._tree_grp_comm.rank
        self._tree_grp_size = self._tree_grp_comm.size
        self._tree_grp_root = 0
        tree_root_color = (
            0 if self._tree_grp_rank == self._tree_grp_root else MPI.UNDEFINED
        )
        self._tree_grp_root_comm = self._node_comm.Split(
            tree_root_color, self._tree_grp_comm.rank
        )

        # List of MPI shared memory windows
        self._list_windows: dict[Intracomm, list[MPI.Win]] = {}
        self._list_arrays: dict[Intracomm, list[npt.NDArray]] = {}

    @property
    def base_comm(self) -> Intracomm:
        return self._base_comm

    @property
    def node_comm(self) -> Intracomm:
        return self._node_comm

    @property
    def node_rank(self) -> int:
        return self._node_rank

    @property
    def node_size(self) -> int:
        return self._node_size

    @property
    def nproc_reduce(self) -> int:
        return self._nproc_reduce

    @property
    def node_root(self) -> int:
        return self._node_root

    @property
    def node_root_comm(self) -> Comm | None:
        return self._node_root_comm

    @property
    def tree_grp_comm(self) -> Comm:
        return self._tree_grp_comm

    @property
    def tree_grp_rank(self) -> int:
        return self._tree_grp_rank

    @property
    def tree_grp_size(self) -> int:
        return self._tree_grp_size

    @property
    def tree_grp_root(self) -> int:
        return self._tree_grp_root

    @property
    def tree_grp_root_comm(self) -> Comm | None:
        return self._tree_grp_root_comm

    @property
    def list_arrays(self) -> dict:
        return self._list_arrays

    def alloc_shared_array(
        self,
        size: int,
        dtype: npt.DTypeLike,
        comm: Intracomm,
        comm_root: int = 0,
    ) -> tuple[npt.NDArray, MPI.Win]:
        dtype = np.dtype(dtype)
        dtype_bytes = dtype.itemsize
        arr_bytes = size * dtype_bytes if comm.rank == comm_root else 0

        win = MPI.Win.Allocate_shared(
            arr_bytes,
            dtype_bytes,
            comm=comm,
        )
        buf, _ = win.Shared_query(rank=comm_root)
        # np.ndarray provides the view, it doesn't owns the memory
        array = np.ndarray(shape=size, dtype=dtype, buffer=buf)

        if comm not in self._list_windows:
            self._list_windows[comm] = []
        if comm not in self._list_arrays:
            self._list_arrays[comm] = []

        self._list_windows[comm].append(win)
        self._list_arrays[comm].append(array)
        return array, win

    def free_shared_arrays(self) -> None:
        # np.ndarray() simply provides the view, it doesn't transfer the
        # memory ownership. The buffer is owned by the window, so freeing
        # the window frees the buffer as well.

        for comm, wins in self._list_windows.items():
            for win in wins:
                win.Free()
        self._list_windows = {}
        self._list_arrays = {}

    def free_shared_arrays_comm(self, comm: Intracomm) -> None:
        if comm in self._list_windows:
            for win in self._list_windows[comm]:
                win.Free()
            del self._list_windows[comm]
        if comm in self._list_arrays:
            del self._list_arrays[comm]


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
    """A cleanup function to be called at the end of execution.

    Once registered with `atexit`, it will be called automatically at the end.
    The user doesn't need to call this function explicitly.
    """
    try:
        MPI.Finalize()
    except Exception as e:
        if MPI_UTILS.rank == 0:
            print(f"Caught an exception during MPI finalization: {e}")


sys_excepthook = sys.excepthook


# If errors during a parallel run are not handled properly, they can lead to a deadlock
# as discussed here:
# <https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html#exceptions-and-deadlocks>
# The following exception hook taken from <https://stackoverflow.com/a/34313363>
# solves the problem by flushing stderr before calling `Abort(1)` on global
# communicator effectively aborting the MPI execution environment
def mpi_excepthook(exctype, value, traceback):
    """Ensures the rank that crashes prints its traceback, then kills all
    MPI processes upon encountering an exception."""

    sys.stderr.write(f"\n*** Exception raised by MPI rank {MPI_UTILS.rank} ***\n")
    sys_excepthook(exctype, value, traceback)

    sys.stderr.flush()

    # Force the MPI runtime to abort in order to prevent deadlocks
    MPI.COMM_WORLD.Abort(1)


# Override the default Python exception handler
sys.excepthook = mpi_excepthook
