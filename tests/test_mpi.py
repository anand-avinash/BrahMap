import pytest
import numpy as np
from mpi4py import MPI
import brahmap
from brahmap.mpi import SharedMemoryManager


class TestMPIUtils:
    def test_mpi_utils(self, monkeypatch):
        from brahmap.mpi import _MPI

        # start with a global MPI communicator. An explicit communicator
        # update is required to avoid global state pollution
        brahmap.MPI_UTILS.update_communicator(MPI.COMM_WORLD)

        # Test the global MPI_UTILS properties
        assert brahmap.MPI_UTILS.comm == MPI.COMM_WORLD
        assert brahmap.MPI_UTILS.size == MPI.COMM_WORLD.size
        assert brahmap.MPI_UTILS.rank == MPI.COMM_WORLD.rank

        # Test creating a new _MPI instance
        custom_mpi = _MPI(comm=MPI.COMM_SELF)
        assert custom_mpi.comm == MPI.COMM_SELF
        assert custom_mpi.size == MPI.COMM_SELF.size
        assert custom_mpi.rank == MPI.COMM_SELF.rank

        # Test update_communicator
        custom_mpi.update_communicator(MPI.COMM_WORLD)
        assert custom_mpi.comm == MPI.COMM_WORLD
        assert custom_mpi.size == MPI.COMM_WORLD.size
        assert custom_mpi.rank == MPI.COMM_WORLD.rank

        # Test nthreads_per_process
        monkeypatch.setenv("OMP_NUM_THREADS", "4")
        assert custom_mpi.nthreads_per_process == 4

        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        assert custom_mpi.nthreads_per_process == 1


class TestSharedMemoryManager:
    def test_shared_memory_manager_properties(self):
        comm = brahmap.MPI_UTILS.comm
        nproc_reduce = 2
        node_root = 0 if comm.size == 1 else 1

        mgr = SharedMemoryManager(
            base_comm=comm,
            nproc_reduce=nproc_reduce,
            node_root=node_root,
        )

        # Test basic properties
        assert mgr.base_comm == comm
        assert mgr.nproc_reduce == min(max(nproc_reduce, 1), mgr.node_size)
        assert mgr.node_root == node_root

        # Verify comm properties
        assert isinstance(mgr.node_comm, MPI.Intracomm)
        assert 0 <= mgr.node_rank < mgr.node_size
        assert mgr.node_size > 0

        if mgr.node_root_comm is not None and mgr.node_root_comm != MPI.COMM_NULL:
            assert isinstance(mgr.node_root_comm, MPI.Intracomm)
            assert 0 <= mgr.node_root_comm.rank < mgr.node_root_comm.size

        assert isinstance(mgr.tree_grp_comm, MPI.Intracomm)
        assert 0 <= mgr.tree_grp_rank < mgr.tree_grp_size

        if (
            mgr.tree_grp_root_comm is not None
            and mgr.tree_grp_root_comm != MPI.COMM_NULL
        ):
            assert isinstance(mgr.tree_grp_root_comm, MPI.Intracomm)
            assert 0 <= mgr.tree_grp_root_comm.rank < mgr.tree_grp_root_comm.size

        mgr.free_shared_arrays_all()

    @pytest.mark.parametrize(
        "dtype", [np.int32, np.int64, np.float32, np.float64, bool]
    )
    def test_shared_memory_manager_allocations(self, dtype):
        comm = brahmap.MPI_UTILS.comm
        mgr = SharedMemoryManager(base_comm=comm, nproc_reduce=1)

        size = 100
        array, win = mgr.alloc_shared_array_node(size, dtype)

        assert isinstance(array, np.ndarray)
        assert isinstance(win, MPI.Win)
        assert array.shape == (size,)
        assert array.dtype == np.dtype(dtype)

        # Test that array actually acts as shared memory:
        # 1. Initialize to 0 on all ranks
        array[:] = 0
        mgr.node_comm.Barrier()

        # 2. Rank 0 writes a value
        if mgr.node_rank == 0:
            array[10] = 42
        mgr.node_comm.Barrier()

        # 3. All ranks should see the updated value
        if dtype is bool:
            assert array[10]
        else:
            assert array[10] == 42

        # Verify tracking
        handle = mgr.node_comm.handle
        assert handle in mgr._list_windows
        assert handle in mgr._list_arrays
        assert win in mgr._list_windows[handle]

        # Test freeing single shared array
        mgr.free_shared_array(mgr.node_comm, win)
        assert handle not in mgr._list_windows
        assert handle not in mgr._list_arrays

    def test_shared_memory_manager_free_methods(self):
        comm = brahmap.MPI_UTILS.comm
        mgr = SharedMemoryManager(base_comm=comm, nproc_reduce=1)

        arr1, win1 = mgr.alloc_shared_array_node(10, np.float64)
        arr2, win2 = mgr.alloc_shared_array_node(20, np.int32)

        handle = mgr.node_comm.handle
        assert len(mgr._list_windows[handle]) == 2
        assert len(mgr._list_arrays[handle]) == 2

        # Test free_shared_array for one of the multiple arrays
        mgr.free_shared_array(mgr.node_comm, win1)
        # Verify win1 is removed, but win2 remains
        assert handle in mgr._list_windows
        assert handle in mgr._list_arrays
        assert len(mgr._list_windows[handle]) == 1
        assert len(mgr._list_arrays[handle]) == 1
        assert win2 in mgr._list_windows[handle]
        assert win1 not in mgr._list_windows[handle]

        # Free the remaining one using free_shared_array
        mgr.free_shared_array(mgr.node_comm, win2)
        assert handle not in mgr._list_windows
        assert handle not in mgr._list_arrays

        # Re-allocate and test free all
        arr3, win3 = mgr.alloc_shared_array_node(10, np.float64)
        assert handle in mgr._list_windows
        mgr.free_shared_arrays_all()
        assert mgr._list_windows == {}
        assert mgr._list_arrays == {}
