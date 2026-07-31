############################ TEST DESCRIPTION ############################
#
# Test defined here are related to the `BlockDiagonalPreconditionerLO` of BrahMap.
# Analogous to this class, in the test suite, we have defined another version of `BlockDiagonalPreconditionerLO` based on only the python routines.
#
# - class `TestBlkDiagPrecondLO_I_Cpp`:
#
#   -   `test_I_cpp`: tests whether `mult` and `rmult` method overloads of
# the two versions of `BlkDiagPrecondLO_tools.BDPLO_mult_I()` produce the
# same result
#
# - Same as above, but for QU and IQU
#
# - class `TestBlkDiagPrecondLO_I`:
#
#   -   `test_I`: The matrix view of the operator
# `brahmap.interfaces.BlockDiagonalPreconditionerLO` is a block matrix.
# In this test, we first compute the matrix view of the operator and then
# compare the elements of each block (corresponding to a given pixel) with
# their explicit computations
#
# - Same as above, but for QU and IQU
#
###########################################################################

import pytest
import numpy as np

import brahmap

import py_BlkDiagPrecondLO as bdplo
import py_ProcessTimeSamples as hpts


TOLERANCES_1 = {
    np.float32: {"rtol": 1.5e-4, "atol": 1.0e-5},
    np.float64: {"rtol": 1.5e-5, "atol": 1.0e-10},
}

TOLERANCES_2 = {
    np.float32: {"rtol": 1.5e-3, "atol": 1.0e-5},
    np.float64: {"rtol": 1.5e-5, "atol": 1.0e-10},
}


class TestBlkDiagPrecondLO_Cpp:
    def _cpp_test(self, setup_scan, solver_type):
        initint, initfloat = setup_scan

        tol = TOLERANCES_1[initfloat.dtype]
        rtol, atol = tol["rtol"], tol["atol"]

        PTS = hpts.ProcessTimeSamples(
            npix=initint.npix,
            pointings=initint.pointings,
            pointings_flag=initint.pointings_flag,
            solver_type=solver_type,
            pol_angles=initfloat.pol_angles if solver_type > 1 else None,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )
        BDP_cpp = brahmap.core.BlockDiagonalPreconditionerLO(PTS)
        BDP_py = bdplo.BlockDiagonalPreconditionerLO(PTS)

        vec = np.random.random(PTS.new_npix * PTS.solver_type).astype(
            dtype=initfloat.dtype, copy=False
        )

        cpp_prod = BDP_cpp * vec
        py_prod = BDP_py * vec

        np.testing.assert_allclose(cpp_prod, py_prod, rtol=rtol, atol=atol)

    def test_I_cpp(self, setup_scan):
        self._cpp_test(setup_scan, hpts.SolverType.I)

    def test_QU_cpp(self, setup_scan):
        self._cpp_test(setup_scan, hpts.SolverType.QU)

    def test_IQU_cpp(self, setup_scan):
        self._cpp_test(setup_scan, hpts.SolverType.IQU)


class TestBlkDiagPrecondLO:
    def test_I(self, setup_scan):
        initint, initfloat = setup_scan

        tol = TOLERANCES_1[initfloat.dtype]
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
        BDP = brahmap.core.BlockDiagonalPreconditionerLO(PTS)

        bdp_array = BDP.to_array()
        diag_inv_count = np.diag(1.0 / PTS.weighted_counts)

        np.testing.assert_allclose(bdp_array, diag_inv_count, rtol=rtol, atol=atol)

    def test_QU(self, setup_scan):
        initint, initfloat = setup_scan

        tol = TOLERANCES_2[initfloat.dtype]
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
        BDP = brahmap.core.BlockDiagonalPreconditionerLO(PTS)

        bdp_matrix = BDP.to_array()

        bdp_test_matrix = np.zeros(
            (PTS.new_npix * PTS.solver_type, PTS.new_npix * PTS.solver_type),
            dtype=initfloat.dtype,
        )

        for idx in range(PTS.new_npix):
            block_matrix = np.zeros((2, 2), dtype=initfloat.dtype)
            block_matrix[0, 0] = PTS.weighted_cos_sq[idx]
            block_matrix[0, 1] = PTS.weighted_sincos[idx]
            block_matrix[1, 0] = PTS.weighted_sincos[idx]
            block_matrix[1, 1] = PTS.weighted_sin_sq[idx]
            block_inv = np.linalg.inv(block_matrix)

            bdp_test_matrix[
                idx * 2 : (idx + 1) * 2, idx * 2 : (idx + 1) * 2
            ] = block_inv

        np.testing.assert_allclose(bdp_matrix, bdp_test_matrix, rtol=rtol, atol=atol)

    def test_IQU(self, setup_scan):
        initint, initfloat = setup_scan

        tol = TOLERANCES_2[initfloat.dtype]
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
        BDP = brahmap.core.BlockDiagonalPreconditionerLO(PTS)

        bdp_matrix = BDP.to_array()

        bdp_test_matrix = np.zeros(
            (PTS.new_npix * PTS.solver_type, PTS.new_npix * PTS.solver_type),
            dtype=initfloat.dtype,
        )

        for idx in range(PTS.new_npix):
            block_matrix = np.zeros((3, 3), dtype=initfloat.dtype)
            block_matrix[0, 0] = PTS.weighted_counts[idx]
            block_matrix[0, 1] = PTS.weighted_cos[idx]
            block_matrix[0, 2] = PTS.weighted_sin[idx]
            block_matrix[1, 0] = PTS.weighted_cos[idx]
            block_matrix[1, 1] = PTS.weighted_cos_sq[idx]
            block_matrix[1, 2] = PTS.weighted_sincos[idx]
            block_matrix[2, 0] = PTS.weighted_sin[idx]
            block_matrix[2, 1] = PTS.weighted_sincos[idx]
            block_matrix[2, 2] = PTS.weighted_sin_sq[idx]
            block_inv = np.linalg.inv(block_matrix)

            bdp_test_matrix[
                idx * 3 : (idx + 1) * 3, idx * 3 : (idx + 1) * 3
            ] = block_inv

        np.testing.assert_allclose(bdp_matrix, bdp_test_matrix, rtol=rtol, atol=atol)


class TestShMemBlkDiagPrecondLO:
    def _shmem_test(self, setup_scan, solver_type):
        initint, initfloat = setup_scan

        tol = TOLERANCES_1[initfloat.dtype]
        rtol, atol = tol["rtol"], tol["atol"]

        nproc_reduce = 2

        # Create SharedMemProcessTimeSamples
        shm_PTS = brahmap.core.SharedMemProcessTimeSamples(
            npix=initint.npix,
            pointings=initint.pointings,
            pointings_flag=initint.pointings_flag,
            solver_type=solver_type,
            pol_angles=initfloat.pol_angles if solver_type > 1 else None,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
            nproc_reduce=nproc_reduce,
        )

        # Create standard ProcessTimeSamples
        std_PTS = brahmap.core.ProcessTimeSamples(
            npix=initint.npix,
            pointings=initint.pointings,
            pointings_flag=initint.pointings_flag,
            solver_type=solver_type,
            pol_angles=initfloat.pol_angles if solver_type > 1 else None,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        # Create operators
        shm_P = brahmap.core.BlockDiagonalPreconditionerLO(shm_PTS, return_copy=True)
        shm_P_nocopy = brahmap.core.BlockDiagonalPreconditionerLO(
            shm_PTS, return_copy=False
        )
        std_P = brahmap.core.BlockDiagonalPreconditionerLO(std_PTS)

        vec = None
        if brahmap.MPI_UTILS.rank == 0:
            vec = np.random.random(shm_PTS.new_npix * shm_PTS.solver_type).astype(
                dtype=initfloat.dtype, copy=False
            )
        vec = brahmap.MPI_UTILS.comm.bcast(vec, root=0)

        shm_mult_prod = shm_P * vec
        shm_mult_prod_nocopy = shm_P_nocopy * vec
        std_mult_prod = std_P * vec

        np.testing.assert_allclose(
            shm_mult_prod,
            std_mult_prod,
            rtol=rtol,
            atol=atol,
        )

        np.testing.assert_allclose(
            shm_mult_prod_nocopy,
            std_mult_prod,
            rtol=rtol,
            atol=atol,
        )

        assert shm_mult_prod_nocopy.ctypes.data == shm_P_nocopy._node_prod.ctypes.data

    def test_I(self, setup_scan):
        self._shmem_test(setup_scan, brahmap.core.SolverType.I)

    def test_QU(self, setup_scan):
        self._shmem_test(setup_scan, brahmap.core.SolverType.QU)

    def test_IQU(self, setup_scan):
        self._shmem_test(setup_scan, brahmap.core.SolverType.IQU)


if __name__ == "__main__":
    pytest.main(
        [
            f"{__file__}::TestBlkDiagPrecondLO_Cpp::test_I_cpp",
            "-v",
            "-s",
        ]
    )
    pytest.main(
        [
            f"{__file__}::TestBlkDiagPrecondLO_Cpp::test_QU_cpp",
            "-v",
            "-s",
        ]
    )
    pytest.main(
        [
            f"{__file__}::TestBlkDiagPrecondLO_Cpp::test_IQU_cpp",
            "-v",
            "-s",
        ]
    )

    pytest.main(
        [
            f"{__file__}::TestBlkDiagPrecondLO::test_I",
            "-v",
            "-s",
        ]
    )
    pytest.main(
        [
            f"{__file__}::TestBlkDiagPrecondLO::test_QU",
            "-v",
            "-s",
        ]
    )
    pytest.main(
        [
            f"{__file__}::TestBlkDiagPrecondLO::test_IQU",
            "-v",
            "-s",
        ]
    )
    pytest.main(
        [
            f"{__file__}::TestShMemBlkDiagPrecondLO::test_I",
            "-v",
            "-s",
        ]
    )
    pytest.main(
        [
            f"{__file__}::TestShMemBlkDiagPrecondLO::test_QU",
            "-v",
            "-s",
        ]
    )
    pytest.main(
        [
            f"{__file__}::TestShMemBlkDiagPrecondLO::test_IQU",
            "-v",
            "-s",
        ]
    )
