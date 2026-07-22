import os
import sys
from typing import cast

from mpi4py import MPI
from mpi4py.MPI import Intracomm, Comm

import numpy as np
import numpy.typing as npt


class SharedMemoryManager(object):
    """Manages MPI shared-memory communicators, window allocations, and tree
    group reductions.

    This manager splits a base MPI communicator into a node-level
    shared-memory communicator and allocates MPI window-backed shared NumPy
    arrays. It also splits the node-level communicator into a tree group
    sub-communicators to orchestrate sequential accumulations and group-wise
    reductions within each node.

    Parameters
    ----------
    base_comm : Intracomm
        The base MPI communicator (typically `MPI.COMM_WORLD`)
    nproc_reduce : int, optional
        The size of each local tree group sub-communicator split from the
        node-level communicator. This group size determines how the
        node-local processes are partitioned into smaller sub-communicator
        chunks. It is typically used by calling containers to orchestrate
        sequential shared-memory accumulations within each sub-communicator
        and group-wise reductions across group roots. By default `1`
    node_root : int, optional
        The designated root rank within the node-level shared memory
        communicator. By default `0`

    Attributes
    ----------
    base_comm : Intracomm
        The base MPI communicator
    node_comm : Intracomm
        The node-level shared-memory MPI communicator
    node_rank : int
        The process rank within the node-level communicator
    node_size : int
        The total number of processes on the current node
    nproc_reduce : int
        The size of each local tree group sub-communicator split from the
        node-level communicator
    node_root : int
        The root rank on the current node communicator
    node_root_comm : Comm or None
        Communicator containing only the root ranks of each node, used for
        inter-node communication
    tree_grp_comm : Intracomm
        Sub-communicator group for local serialized accumulations
    tree_grp_rank : int
        The rank within the local tree group communicator
    tree_grp_size : int
        The size of the local tree group communicator
    tree_grp_root : int
        The designated root rank of the tree group (usually 0)
    tree_grp_root_comm : Intracomm
        Communicator containing only the tree group roots on this node, used
        for intra-node aggregation
    list_windows : dict[int, list[MPI.Win]]
        Tracks allocated shared-memory MPI windows mapped by communicator handle
    list_arrays : dict[int, list[npt.NDArray]]
        Tracks allocated shared-memory NumPy array views mapped by
        communicator handle
    """

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
        self._list_windows: dict[int, list[MPI.Win]] = {}
        self._list_arrays: dict[int, list[npt.NDArray]] = {}

    @property
    def base_comm(self) -> Intracomm:
        """The base global MPI communicator

        Returns
        -------
        MPI.Intracomm
            The base MPI communicator
        """
        return self._base_comm

    @property
    def node_comm(self) -> Intracomm:
        """The node-level shared-memory MPI communicator

        Returns
        -------
        MPI.Intracomm
            The node-level MPI communicator
        """
        return self._node_comm

    @property
    def node_rank(self) -> int:
        """The process rank within the node-level communicator

        Returns
        -------
        int
            Node-local rank corresponding to the node-level communicator
        """
        return self._node_rank

    @property
    def node_size(self) -> int:
        """The total number of processes on the current node

        Returns
        -------
        int
            Size of this node-level communicator
        """
        return self._node_size

    @property
    def nproc_reduce(self) -> int:
        """The size of each local tree group sub-communicator split from the
        node-level communicator

        Returns
        -------
        int
            Size of the local tree group
        """
        return self._nproc_reduce

    @property
    def node_root(self) -> int:
        """The root rank on the current node

        Returns
        -------
        int
            Root rank of the node-level communicator
        """
        return self._node_root

    @property
    def node_root_comm(self) -> Comm | None:
        """The communicator containing only the root ranks of each
        node-level communicator, or None

        Returns
        -------
        MPI.Comm
            The node root communicator
        """
        return self._node_root_comm

    @property
    def tree_grp_comm(self) -> Comm:
        """The sub-communicator group for local tree-like serialized
        accumulations

        Returns
        -------
        MPI.Comm
            The local tree group communicator
        """
        return self._tree_grp_comm

    @property
    def tree_grp_rank(self) -> int:
        """The rank within the local tree group communicator

        Returns
        -------
        int
            Local tree group rank
        """
        return self._tree_grp_rank

    @property
    def tree_grp_size(self) -> int:
        """The size of the local tree group communicator

        Returns
        -------
        int
            Local tree group size
        """
        return self._tree_grp_size

    @property
    def tree_grp_root(self) -> int:
        """The designated root rank of the tree group

        Returns
        -------
        int
            Local tree group root
        """
        return self._tree_grp_root

    @property
    def tree_grp_root_comm(self) -> Comm | None:
        """The communicator containing only the tree group roots on this
        node, or None

        Returns
        -------
        MPI.Comm
            The tree group root communicator
        """
        return self._tree_grp_root_comm

    @property
    def list_windows(self) -> dict:
        """The dictionary mapping communicators to list of allocated
        shared-memory windows for that communicator

        Returns
        -------
        dict
            The dictionary mapping communicators to list of allocated
            shared-memory windows for that communicator
        """
        return self._list_windows

    @property
    def list_arrays(self) -> dict:
        """The dictionary mapping communicators to list of allocated
        shared-memory arrays for that communicator

        Returns
        -------
        dict
            The dictionary mapping communicators to list of allocated
            shared-memory arrays for that communicator
        """
        return self._list_arrays

    def alloc_shared_array_comm(
        self,
        size: int,
        dtype: npt.DTypeLike,
        comm: Intracomm,
        comm_root: int = 0,
    ) -> tuple[npt.NDArray, MPI.Win]:
        """Allocates a shared-memory MPI window-backed 1D NumPy array for a
        communicator.

        Parameters
        ----------
        size : int
            The size of the array
        dtype : npt.DTypeLike
            The data type of the array
        comm : Intracomm
            The MPI communicator over which the shared memory window is
            allocated
        comm_root : int, optional
            The root rank in `comm` that allocates the actual memory buffer.
            By default `0`

        Returns
        -------
        tuple[npt.NDArray, MPI.Win]
            A tuple containing the shared NumPy array view and the backing
            MPI window object
        """
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

        handle = comm.handle
        if handle not in self._list_windows:
            self._list_windows[handle] = []
        if handle not in self._list_arrays:
            self._list_arrays[handle] = []

        self._list_windows[handle].append(win)
        self._list_arrays[handle].append(array)
        return array, win

    def alloc_shared_array_node(
        self,
        size: int,
        dtype: npt.DTypeLike,
    ) -> tuple[npt.NDArray, MPI.Win]:
        """Allocates a shared-memory MPI window-backed 1D NumPy array for the
        node-level communicator

        Parameters
        ----------
        size : int
            The size of the array
        dtype : npt.DTypeLike
            The data type of the array

        Returns
        -------
        tuple[npt.NDArray, MPI.Win]
            A tuple containing the shared NumPy array view and the backing
            MPI window object
        """
        return self.alloc_shared_array_comm(
            size=size,
            dtype=dtype,
            comm=self.node_comm,
            comm_root=self.node_root,
        )

    def free_shared_arrays_all(self) -> None:
        """Frees all allocated shared-memory MPI windows and clears manager
        state.

        Since the window owns the actual memory buffer, freeing the windows
        also deallocates the underlying buffers of all tracked shared-memory
        arrays.

        Returns
        -------
        None
        """
        # np.ndarray() simply provides the view, it doesn't transfer the
        # memory ownership. The buffer is owned by the window, so freeing
        # the window frees the buffer as well.

        for comm, wins in self._list_windows.items():
            for win in wins:
                win.Free()
        self._list_windows = {}
        self._list_arrays = {}

    def free_shared_arrays_comm(self, comm: Intracomm) -> None:
        """Frees all shared-memory MPI windows allocated for a specific
        communicator.

        Parameters
        ----------
        comm : Intracomm
            The MPI communicator whose shared memory windows should be freed.

        Returns
        -------
        None
        """
        handle = comm.handle
        if handle in self._list_windows:
            for win in self._list_windows[handle]:
                win.Free()
            del self._list_windows[handle]
        if handle in self._list_arrays:
            del self._list_arrays[handle]

    def free_shared_array(self, comm: Intracomm, win: MPI.Win) -> None:
        """Frees a specific shared-memory MPI window and removes its associated
        array view and window from the manager's tracking lists.

        Parameters
        ----------
        comm : Intracomm
            The MPI communicator over which the shared memory window was
            allocated
        win : MPI.Win
            The MPI window object to be freed

        Returns
        -------
        None
        """
        handle = comm.handle
        if handle in self._list_windows and win in self._list_windows[handle]:
            idx = self._list_windows[handle].index(win)
            win.Free()
            self._list_windows[handle].pop(idx)
            self._list_arrays[handle].pop(idx)
            if not self._list_windows[handle]:
                del self._list_windows[handle]
            if not self._list_arrays[handle]:
                del self._list_arrays[handle]


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
