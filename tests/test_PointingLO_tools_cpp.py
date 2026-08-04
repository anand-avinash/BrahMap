############################ TEST DESCRIPTION ############################
#
# Test defined here are related to the functions defined in the extension
# module `PointingLO_tools`. All the tests defined here simply test if the
# computations defined the cpp functions produce the same results as their
# python analog.
#
# - class `TestPointingLOTools_I`:
#
#   -   `test_I`: tests the computations of `PointingLO_tools.PLO_mult_I()`
# and `PointingLO_tools.PLO_rmult_I()`
#
# - Same as above, but for QU and IQU
#
###########################################################################

import pytest
import numpy as np

import brahmap
from brahmap._extensions import PointingLO_tools

import py_ProcessTimeSamples as hpts
import py_PointingLO_tools as hplo_tools


TOLERANCES = {
    np.float32: {"rtol": 1.5e-4, "atol": 1.0e-5},
    np.float64: {"rtol": 1.5e-5, "atol": 1.0e-10},
}


class TestPointingLOTools_I:
    def test_I(self, setup_scan):
        initint, initfloat = setup_scan

        tol = TOLERANCES[initfloat.dtype]
        rtol, atol = tol["rtol"], tol["atol"]
        solver_type = hpts.SolverType.I

        PTS = hpts.ProcessTimeSamples(
            npix=initint.npix,
            pointings=initint.pointings,
            pointings_flag=initint.pointings_flag,
            solver_type=solver_type,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        nrows = initint.nsamples
        ncols = PTS.new_npix * PTS.solver_type

        cpp_mult_prod = np.zeros(nrows, dtype=initfloat.dtype)
        vec = np.resize(initfloat.vec, ncols)

        PointingLO_tools.PLO_mult_I(
            nrows,
            PTS.pointings,
            PTS.pointings_flag,
            vec,
            cpp_mult_prod,
        )
        py_mult_prod = hplo_tools.PLO_mult_I(
            nrows,
            PTS.pointings,
            PTS.pointings_flag,
            vec,
        )

        cpp_rmult_prod = np.zeros(ncols, dtype=initfloat.dtype)
        rvec = initfloat.rvec

        PointingLO_tools.PLO_rmult_I(
            PTS.new_npix,
            nrows,
            PTS.pointings,
            PTS.pointings_flag,
            rvec,
            cpp_rmult_prod,
            brahmap.MPI_UTILS.comm,
        )
        py_rmult_prod = hplo_tools.PLO_rmult_I(
            nrows,
            ncols,
            PTS.pointings,
            PTS.pointings_flag,
            rvec,
            brahmap.MPI_UTILS.comm,
        )

        np.testing.assert_allclose(
            cpp_mult_prod,
            py_mult_prod,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            cpp_rmult_prod,
            py_rmult_prod,
            rtol=rtol,
            atol=atol,
        )


class TestPointingLOTools_QU:
    def test_QU(self, setup_scan):
        initint, initfloat = setup_scan

        tol = TOLERANCES[initfloat.dtype]
        rtol, atol = tol["rtol"], tol["atol"]
        solver_type = hpts.SolverType.QU

        PTS = hpts.ProcessTimeSamples(
            npix=initint.npix,
            pointings=initint.pointings,
            pointings_flag=initint.pointings_flag,
            solver_type=solver_type,
            pol_angles=initfloat.pol_angles,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        nrows = initint.nsamples
        ncols = PTS.new_npix * PTS.solver_type

        cpp_mult_prod = np.zeros(nrows, dtype=initfloat.dtype)
        vec = np.resize(initfloat.vec, ncols)

        PointingLO_tools.PLO_mult_QU(
            nrows,
            PTS.pointings,
            PTS.pointings_flag,
            PTS.sin2phi,
            PTS.cos2phi,
            vec,
            cpp_mult_prod,
        )
        py_mult_prod = hplo_tools.PLO_mult_QU(
            nrows,
            PTS.pointings,
            PTS.pointings_flag,
            PTS.sin2phi,
            PTS.cos2phi,
            vec,
        )

        cpp_rmult_prod = np.zeros(ncols, dtype=initfloat.dtype)
        rvec = initfloat.rvec

        PointingLO_tools.PLO_rmult_QU(
            PTS.new_npix,
            nrows,
            PTS.pointings,
            PTS.pointings_flag,
            PTS.sin2phi,
            PTS.cos2phi,
            rvec,
            cpp_rmult_prod,
            brahmap.MPI_UTILS.comm,
        )
        py_rmult_prod = hplo_tools.PLO_rmult_QU(
            nrows,
            ncols,
            PTS.pointings,
            PTS.pointings_flag,
            PTS.sin2phi,
            PTS.cos2phi,
            rvec,
            brahmap.MPI_UTILS.comm,
        )

        np.testing.assert_allclose(
            cpp_mult_prod,
            py_mult_prod,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            cpp_rmult_prod,
            py_rmult_prod,
            rtol=rtol,
            atol=atol,
        )


class TestPointingLOTools_IQU:
    def test_IQU(self, setup_scan):
        initint, initfloat = setup_scan

        tol = TOLERANCES[initfloat.dtype]
        rtol, atol = tol["rtol"], tol["atol"]
        solver_type = hpts.SolverType.IQU

        PTS = hpts.ProcessTimeSamples(
            npix=initint.npix,
            pointings=initint.pointings,
            pointings_flag=initint.pointings_flag,
            solver_type=solver_type,
            pol_angles=initfloat.pol_angles,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        nrows = initint.nsamples
        ncols = PTS.new_npix * PTS.solver_type

        cpp_mult_prod = np.zeros(nrows, dtype=initfloat.dtype)
        vec = np.resize(initfloat.vec, ncols)

        PointingLO_tools.PLO_mult_IQU(
            nrows,
            PTS.pointings,
            PTS.pointings_flag,
            PTS.sin2phi,
            PTS.cos2phi,
            vec,
            cpp_mult_prod,
        )
        py_mult_prod = hplo_tools.PLO_mult_IQU(
            nrows,
            PTS.pointings,
            PTS.pointings_flag,
            PTS.sin2phi,
            PTS.cos2phi,
            vec,
        )

        cpp_rmult_prod = np.zeros(ncols, dtype=initfloat.dtype)
        rvec = initfloat.rvec

        PointingLO_tools.PLO_rmult_IQU(
            PTS.new_npix,
            nrows,
            PTS.pointings,
            PTS.pointings_flag,
            PTS.sin2phi,
            PTS.cos2phi,
            rvec,
            cpp_rmult_prod,
            brahmap.MPI_UTILS.comm,
        )
        py_rmult_prod = hplo_tools.PLO_rmult_IQU(
            nrows,
            ncols,
            PTS.pointings,
            PTS.pointings_flag,
            PTS.sin2phi,
            PTS.cos2phi,
            rvec,
            brahmap.MPI_UTILS.comm,
        )

        np.testing.assert_allclose(
            cpp_mult_prod,
            py_mult_prod,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            cpp_rmult_prod,
            py_rmult_prod,
            rtol=rtol,
            atol=atol,
        )


class TestPointingLOTools_ShMem:
    def _shmem_test(self, setup_scan, solver_type, shmem_func, py_func):
        initint, initfloat = setup_scan

        tol = TOLERANCES[initfloat.dtype]
        rtol, atol = tol["rtol"], tol["atol"]

        shm_PTS = brahmap.core.SharedMemProcessTimeSamples(
            npix=initint.npix,
            pointings=initint.pointings,
            pointings_flag=initint.pointings_flag,
            solver_type=solver_type,
            pol_angles=initfloat.pol_angles if solver_type > 1 else None,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
            nproc_reduce=2,
        )

        mgr = shm_PTS.shared_mem_manager
        ncols = shm_PTS.new_npix * shm_PTS.solver_type

        node_prod, win_node_prod = mgr.alloc_shared_zeros_node(
            ncols,
            initfloat.dtype,
        )

        if mgr.tree_grp_size == 1:
            grp_prod = node_prod
            win_grp_prod = win_node_prod
        else:
            grp_prod, win_grp_prod = mgr.alloc_shared_zeros_comm(
                ncols,
                initfloat.dtype,
                comm=mgr.tree_grp_comm,
                comm_root=0,
            )
            mgr.fence_comm_all(mgr.tree_grp_comm)

        mgr.fence_comm_all(mgr.node_comm)

        rvec = initfloat.rvec.astype(initfloat.dtype)

        if solver_type == brahmap.core.SolverType.I:
            shmem_func(
                shm_PTS.new_npix,
                initint.nsamples,
                shm_PTS.pointings,
                shm_PTS.pointings_flag,
                rvec,
                grp_prod,
                win_grp_prod,
                node_prod,
                win_node_prod,
                mgr.node_root,
                mgr.grp_reduce,
                mgr.tree_grp_comm,
                mgr.tree_grp_root_comm,
                mgr.node_comm,
                mgr.node_root_comm,
            )
        else:
            shmem_func(
                shm_PTS.new_npix,
                initint.nsamples,
                shm_PTS.pointings,
                shm_PTS.pointings_flag,
                shm_PTS.sin2phi,
                shm_PTS.cos2phi,
                rvec,
                grp_prod,
                win_grp_prod,
                node_prod,
                win_node_prod,
                mgr.node_root,
                mgr.grp_reduce,
                mgr.tree_grp_comm,
                mgr.tree_grp_root_comm,
                mgr.node_comm,
                mgr.node_root_comm,
            )

        if solver_type == brahmap.core.SolverType.I:
            py_rmult_prod = py_func(
                initint.nsamples,
                ncols,
                shm_PTS.pointings,
                shm_PTS.pointings_flag,
                rvec,
                brahmap.MPI_UTILS.comm,
            )
        else:
            py_rmult_prod = py_func(
                initint.nsamples,
                ncols,
                shm_PTS.pointings,
                shm_PTS.pointings_flag,
                shm_PTS.sin2phi,
                shm_PTS.cos2phi,
                rvec,
                brahmap.MPI_UTILS.comm,
            )

        np.testing.assert_allclose(
            node_prod,
            py_rmult_prod,
            rtol=rtol,
            atol=atol,
        )

        shm_PTS.free_shmem_arrays()

    def test_I_shmem(self, setup_scan):
        self._shmem_test(
            setup_scan,
            brahmap.core.SolverType.I,
            PointingLO_tools.shmem_PLO_rmult_I,
            hplo_tools.PLO_rmult_I,
        )

    def test_QU_shmem(self, setup_scan):
        self._shmem_test(
            setup_scan,
            brahmap.core.SolverType.QU,
            PointingLO_tools.shmem_PLO_rmult_QU,
            hplo_tools.PLO_rmult_QU,
        )

    def test_IQU_shmem(self, setup_scan):
        self._shmem_test(
            setup_scan,
            brahmap.core.SolverType.IQU,
            PointingLO_tools.shmem_PLO_rmult_IQU,
            hplo_tools.PLO_rmult_IQU,
        )


if __name__ == "__main__":
    pytest.main([f"{__file__}::TestPointingLOTools_I::test_I", "-v", "-s"])
    pytest.main([f"{__file__}::TestPointingLOTools_QU::test_QU", "-v", "-s"])
    pytest.main([f"{__file__}::TestPointingLOTools_IQU::test_IQU", "-v", "-s"])
    pytest.main([f"{__file__}::TestPointingLOTools_ShMem::test_I_shmem", "-v", "-s"])
    pytest.main([f"{__file__}::TestPointingLOTools_ShMem::test_QU_shmem", "-v", "-s"])
    pytest.main([f"{__file__}::TestPointingLOTools_ShMem::test_IQU_shmem", "-v", "-s"])
