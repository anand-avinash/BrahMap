############################ TEST DESCRIPTION ############################
#
# Test defined here are related to the `PointingLO` class of BrahMap.
# Analogous to this class, in the test suite, we have defined another version
# of `PointingLO` based on only the python routines.
#
# - class `TestPointingLO_Cpp`:
#
#   -   `test_I_Cpp`: tests whether the `mult` and `rmult` method overloads
# of the the two versions of `PointingLO` produce the same results.
#
# - Same as above, but for QU and IQU
#
# - class `TestPointingLO`:
#
#   -   `test_I`: tests the `mult` and `rmult` method overloads of
# `brahmap.interfaces.PointingLO` against their explicit computations.
#
# - Same as above, but for QU and IQU
#
# Note: For I case `P.T * noise_vector` must be equal to the
# `weighted_counts` vector. For QU case, the resulting vector must have
# `weighted_cos` and `weighted_sin` at alternating position. And for IQU
# case, the resulting vector must have `weighted_counts`, `weighted_cos`
# and `weighted_sin` at alternating positions.
#
###########################################################################

import pytest
import numpy as np
import brahmap

import py_PointingLO as hplo

from mpi4py import MPI


TOLERANCES = {
    np.float32: {"rtol": 1.5e-4, "atol": 1.0e-5},
    np.float64: {"rtol": 1.5e-5, "atol": 1.0e-10},
}


class TestPointingLO_Cpp:
    def test_I_Cpp(self, setup_scan):
        initint, initfloat = setup_scan

        tol = TOLERANCES[initfloat.dtype]
        rtol, atol = tol["rtol"], tol["atol"]
        solver_type = brahmap.core.SolverType.I

        PTS = brahmap.core.ProcessTimeSamples(
            npix=initint.npix,
            pointings=initint.pointings,
            pointings_flag=initint.pointings_flag,
            solver_type=solver_type,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        P_cpp = brahmap.core.PointingLO(PTS)
        P_py = hplo.PointingLO(PTS)

        ncols = PTS.new_npix * PTS.solver_type

        vec = np.resize(initfloat.vec, ncols)
        cpp_mult_prod = P_cpp * vec
        py_mult_prod = P_py * vec

        rvec = initfloat.rvec
        cpp_rmult_prod = P_cpp.T * rvec
        py_rmult_prod = P_py.T * rvec

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

    def test_QU_Cpp(self, setup_scan):
        initint, initfloat = setup_scan

        tol = TOLERANCES[initfloat.dtype]
        rtol, atol = tol["rtol"], tol["atol"]
        solver_type = brahmap.core.SolverType.QU

        PTS = brahmap.core.ProcessTimeSamples(
            npix=initint.npix,
            pointings=initint.pointings,
            pointings_flag=initint.pointings_flag,
            solver_type=solver_type,
            pol_angles=initfloat.pol_angles,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        P_cpp = brahmap.core.PointingLO(PTS)
        P_py = hplo.PointingLO(PTS)

        ncols = PTS.new_npix * PTS.solver_type

        vec = np.resize(initfloat.vec, ncols)
        cpp_mult_prod = P_cpp * vec
        py_mult_prod = P_py * vec

        rvec = initfloat.rvec
        cpp_rmult_prod = P_cpp.T * rvec
        py_rmult_prod = P_py.T * rvec

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

    def test_IQU_Cpp(self, setup_scan):
        initint, initfloat = setup_scan

        tol = TOLERANCES[initfloat.dtype]
        rtol, atol = tol["rtol"], tol["atol"]
        solver_type = brahmap.core.SolverType.IQU

        PTS = brahmap.core.ProcessTimeSamples(
            npix=initint.npix,
            pointings=initint.pointings,
            pointings_flag=initint.pointings_flag,
            solver_type=solver_type,
            pol_angles=initfloat.pol_angles,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        P_cpp = brahmap.core.PointingLO(PTS)
        P_py = hplo.PointingLO(PTS)

        ncols = PTS.new_npix * PTS.solver_type

        vec = np.resize(initfloat.vec, ncols)
        cpp_mult_prod = P_cpp * vec
        py_mult_prod = P_py * vec

        rvec = initfloat.rvec
        cpp_rmult_prod = P_cpp.T * rvec
        py_rmult_prod = P_py.T * rvec

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


class TestPointingLO:
    def test_I(self, setup_scan):
        initint, initfloat = setup_scan

        tol = TOLERANCES[initfloat.dtype]
        rtol, atol = tol["rtol"], tol["atol"]
        solver_type = brahmap.core.SolverType.I

        PTS = brahmap.core.ProcessTimeSamples(
            npix=initint.npix,
            pointings=initint.pointings,
            pointings_flag=initint.pointings_flag,
            solver_type=solver_type,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        P = brahmap.core.PointingLO(PTS)

        # Test for P.T * <vector>
        weights = P.T * initfloat.noise_weights

        np.testing.assert_allclose(
            PTS.weighted_counts,
            weights,
            rtol=rtol,
            atol=atol,
        )

        # Test for P * <vector>
        ncols = PTS.new_npix * PTS.solver_type
        vec = np.resize(initfloat.vec, ncols)
        signal = P * vec

        signal_test = np.zeros(initint.nsamples, dtype=initfloat.dtype)

        for idx in range(initint.nsamples):
            pixel = PTS.pointings[idx]
            if PTS.pointings_flag[idx]:
                signal_test[idx] += vec[pixel]

        np.testing.assert_allclose(
            signal,
            signal_test,
            rtol=rtol,
            atol=atol,
        )

    def test_QU(self, setup_scan):
        initint, initfloat = setup_scan

        tol = TOLERANCES[initfloat.dtype]
        rtol, atol = tol["rtol"], tol["atol"]
        solver_type = brahmap.core.SolverType.QU

        PTS = brahmap.core.ProcessTimeSamples(
            npix=initint.npix,
            pointings=initint.pointings,
            pointings_flag=initint.pointings_flag,
            solver_type=solver_type,
            pol_angles=initfloat.pol_angles,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        P = brahmap.core.PointingLO(PTS)

        # Test for P.T * <vector>
        weights = P.T * initfloat.noise_weights

        weighted_sin = np.zeros(PTS.new_npix, dtype=initfloat.dtype)
        weighted_cos = np.zeros(PTS.new_npix, dtype=initfloat.dtype)

        for idx in range(initint.nsamples):
            if PTS.pointings_flag[idx]:
                pixel = PTS.pointings[idx]
                weighted_sin[pixel] += PTS.sin2phi[idx] * initfloat.noise_weights[idx]
                weighted_cos[pixel] += PTS.cos2phi[idx] * initfloat.noise_weights[idx]

        brahmap.MPI_UTILS.comm.Allreduce(MPI.IN_PLACE, weighted_sin, MPI.SUM)
        brahmap.MPI_UTILS.comm.Allreduce(MPI.IN_PLACE, weighted_cos, MPI.SUM)

        np.testing.assert_allclose(
            weighted_sin,
            weights[1::2],
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            weighted_cos,
            weights[0::2],
            rtol=rtol,
            atol=atol,
        )

        # Test for P * <vector>
        ncols = PTS.new_npix * PTS.solver_type
        vec = np.resize(initfloat.vec, ncols)
        signal = P * vec

        signal_test = np.zeros(initint.nsamples, dtype=initfloat.dtype)

        for idx in range(initint.nsamples):
            pixel = PTS.pointings[idx]
            if PTS.pointings_flag[idx]:
                signal_test[idx] += (
                    vec[2 * pixel + 0] * PTS.cos2phi[idx]
                    + vec[2 * pixel + 1] * PTS.sin2phi[idx]
                )

        np.testing.assert_allclose(
            signal,
            signal_test,
            rtol=rtol,
            atol=atol,
        )

    def test_IQU(self, setup_scan):
        initint, initfloat = setup_scan

        tol = TOLERANCES[initfloat.dtype]
        rtol, atol = tol["rtol"], tol["atol"]
        solver_type = brahmap.core.SolverType.IQU

        PTS = brahmap.core.ProcessTimeSamples(
            npix=initint.npix,
            pointings=initint.pointings,
            pointings_flag=initint.pointings_flag,
            solver_type=solver_type,
            pol_angles=initfloat.pol_angles,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        P = brahmap.core.PointingLO(PTS)

        # Test for P.T * <vector>
        weights = P.T * initfloat.noise_weights

        np.testing.assert_allclose(
            PTS.weighted_counts,
            weights[0::3],
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            PTS.weighted_cos,
            weights[1::3],
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            PTS.weighted_sin,
            weights[2::3],
            rtol=rtol,
            atol=atol,
        )

        # Test for P * <vector>
        ncols = PTS.new_npix * PTS.solver_type
        vec = np.resize(initfloat.vec, ncols)
        signal = P * vec

        signal_test = np.zeros(initint.nsamples, dtype=initfloat.dtype)

        for idx in range(initint.nsamples):
            pixel = PTS.pointings[idx]
            if PTS.pointings_flag[idx]:
                signal_test[idx] += (
                    vec[3 * pixel + 0] * 1.0
                    + vec[3 * pixel + 1] * PTS.cos2phi[idx]
                    + vec[3 * pixel + 2] * PTS.sin2phi[idx]
                )

        np.testing.assert_allclose(
            signal,
            signal_test,
            rtol=rtol,
            atol=atol,
        )


class TestShMemPointingLO:
    def _shmem_test(self, setup_scan, solver_type):
        initint, initfloat = setup_scan

        tol = TOLERANCES[initfloat.dtype]
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
        shm_P = brahmap.core.PointingLO(shm_PTS)
        std_P = brahmap.core.PointingLO(std_PTS)

        assert hasattr(shm_P, "_PointingLO__shared_mem_mgr")

        # Test for P * <vector>
        ncols = shm_PTS.new_npix * shm_PTS.solver_type
        vec = np.resize(initfloat.vec, ncols).astype(initfloat.dtype)

        shm_mult_prod = shm_P * vec
        std_mult_prod = std_P * vec

        np.testing.assert_allclose(
            shm_mult_prod,
            std_mult_prod,
            rtol=rtol,
            atol=atol,
        )

        # Test for P.T * <vector>
        rvec = initfloat.rvec.astype(initfloat.dtype)

        shm_rmult_prod = shm_P.T * rvec
        std_rmult_prod = std_P.T * rvec

        np.testing.assert_allclose(
            shm_rmult_prod,
            std_rmult_prod,
            rtol=rtol,
            atol=atol,
        )

    def test_I(self, setup_scan):
        self._shmem_test(setup_scan, brahmap.core.SolverType.I)

    def test_QU(self, setup_scan):
        self._shmem_test(setup_scan, brahmap.core.SolverType.QU)

    def test_IQU(self, setup_scan):
        self._shmem_test(setup_scan, brahmap.core.SolverType.IQU)


if __name__ == "__main__":
    pytest.main([f"{__file__}::TestPointingLO_Cpp::test_I_Cpp", "-v", "-s"])
    pytest.main([f"{__file__}::TestPointingLO_Cpp::test_QU_Cpp", "-v", "-s"])
    pytest.main([f"{__file__}::TestPointingLO_Cpp::test_IQU_Cpp", "-v", "-s"])
    pytest.main([f"{__file__}::TestPointingLO::test_I", "-v", "-s"])
    pytest.main([f"{__file__}::TestPointingLO::test_QU", "-v", "-s"])
    pytest.main([f"{__file__}::TestPointingLO::test_IQU", "-v", "-s"])
