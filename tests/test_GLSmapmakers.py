############################ TEST DESCRIPTION ############################
#
# Tests defined here correspond to `compute_GLS_maps()` function.
#
# - class `TestGLSMapMakers_const_maps`:
#   - Scans the constant I, Q and U maps with random pointings and pol_angles.
# Then does the map-making with unit noise covariance using `compute_GLS_maps()`
# function and tests the results against the constant input maps.
#
# - class `TestGLSMapMakers_rand_maps`
#   - Scans the random I, Q and U maps with random pointings and pol_angles.
# Then does the map-making with unit noise covariance using `compute_GLS_maps()`
# function and tests the results against the random input maps.
#
###########################################################################


import pytest
import numpy as np

import brahmap


TOLERANCES = {
    np.float32: {"rtol": 1.5e-3, "atol": 1.0e-5},
    np.float64: {"rtol": 1.5e-5, "atol": 1.0e-10},
}


class TestGLSMapMakers_const_maps:
    def test_GLSMapMakers_I_const_map(self, setup_scan):
        initint, initfloat = setup_scan

        tol = TOLERANCES[initfloat.dtype]
        rtol, atol = tol["rtol"], tol["atol"]
        solver_type = brahmap.SolverType.I

        tod = np.zeros(initint.nsamples, dtype=initfloat.dtype)

        # scan the sky
        for idx, pointings in enumerate(initint.pointings):
            tod[idx] += initfloat.const_I_map[pointings]

        GLSparams = brahmap.core.GLSParameters(
            solver_type=solver_type,
            isolver_max_iterations=5,
            return_hit_map=False,
            return_processed_samples=True,
        )

        PTS, GLSresults = brahmap.core.compute_GLS_maps(
            npix=initint.npix,
            pointings=initint.pointings,
            time_ordered_data=tod,
            pointings_flag=initint.pointings_flag,
            dtype_float=initfloat.dtype,
            gls_parameters=GLSparams,
            update_pointings_inplace=False,
        )

        np.testing.assert_equal(GLSresults.convergence_status, True)
        # np.testing.assert_equal(GLSresults.num_iterations, 1)

        input_I_map = np.ma.MaskedArray(
            data=initfloat.const_I_map,
            dtype=initfloat.dtype,
            mask=~PTS.pixel_flag,
            fill_value=-1.6375e30,
        )

        np.testing.assert_allclose(
            GLSresults.GLS_maps[0],
            input_I_map,
            rtol=rtol,
            atol=atol,
        )

    def test_GLSMapMakers_QU_const_map(self, setup_scan):
        initint, initfloat = setup_scan

        tol = TOLERANCES[initfloat.dtype]
        rtol, atol = tol["rtol"], tol["atol"]
        solver_type = brahmap.SolverType.QU

        tod = np.zeros(initint.nsamples, dtype=initfloat.dtype)

        sin2phi = np.empty(initint.nsamples, dtype=initfloat.dtype)
        cos2phi = np.empty(initint.nsamples, dtype=initfloat.dtype)
        brahmap.math.sin(initint.nsamples, 2.0 * initfloat.pol_angles, sin2phi)
        brahmap.math.cos(initint.nsamples, 2.0 * initfloat.pol_angles, cos2phi)

        # scan the sky
        for idx, pointings in enumerate(initint.pointings):
            tod[idx] += initfloat.const_Q_map[pointings] * cos2phi[idx]
            tod[idx] += initfloat.const_U_map[pointings] * sin2phi[idx]

        GLSparams = brahmap.core.GLSParameters(
            solver_type=solver_type,
            isolver_max_iterations=5,
            return_hit_map=False,
            return_processed_samples=True,
        )

        PTS, GLSresults = brahmap.core.compute_GLS_maps(
            npix=initint.npix,
            pointings=initint.pointings,
            time_ordered_data=tod,
            pointings_flag=initint.pointings_flag,
            pol_angles=initfloat.pol_angles,
            dtype_float=initfloat.dtype,
            gls_parameters=GLSparams,
            update_pointings_inplace=False,
        )

        np.testing.assert_equal(GLSresults.convergence_status, True)
        # np.testing.assert_equal(GLSresults.num_iterations, 1)

        input_Q_map = np.ma.MaskedArray(
            data=initfloat.const_Q_map,
            dtype=initfloat.dtype,
            mask=~PTS.pixel_flag,
            fill_value=-1.6375e30,
        )
        input_U_map = np.ma.MaskedArray(
            data=initfloat.const_U_map,
            dtype=initfloat.dtype,
            mask=~PTS.pixel_flag,
            fill_value=-1.6375e30,
        )

        np.testing.assert_allclose(
            GLSresults.GLS_maps[0],
            input_Q_map,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            GLSresults.GLS_maps[1],
            input_U_map,
            rtol=rtol,
            atol=atol,
        )

    def test_GLSMapMakers_IQU_const_map(self, setup_scan):
        initint, initfloat = setup_scan

        tol = TOLERANCES[initfloat.dtype]
        rtol, atol = tol["rtol"], tol["atol"]
        solver_type = brahmap.SolverType.IQU

        tod = np.zeros(initint.nsamples, dtype=initfloat.dtype)

        sin2phi = np.empty(initint.nsamples, dtype=initfloat.dtype)
        cos2phi = np.empty(initint.nsamples, dtype=initfloat.dtype)
        brahmap.math.sin(initint.nsamples, 2.0 * initfloat.pol_angles, sin2phi)
        brahmap.math.cos(initint.nsamples, 2.0 * initfloat.pol_angles, cos2phi)

        # scan the sky
        for idx, pointings in enumerate(initint.pointings):
            tod[idx] += initfloat.const_I_map[pointings]
            tod[idx] += initfloat.const_Q_map[pointings] * cos2phi[idx]
            tod[idx] += initfloat.const_U_map[pointings] * sin2phi[idx]

        GLSparams = brahmap.core.GLSParameters(
            solver_type=solver_type,
            isolver_max_iterations=5,
            return_hit_map=False,
            return_processed_samples=True,
        )

        PTS, GLSresults = brahmap.core.compute_GLS_maps(
            npix=initint.npix,
            pointings=initint.pointings,
            time_ordered_data=tod,
            pointings_flag=initint.pointings_flag,
            pol_angles=initfloat.pol_angles,
            dtype_float=initfloat.dtype,
            gls_parameters=GLSparams,
            update_pointings_inplace=False,
        )

        np.testing.assert_equal(GLSresults.convergence_status, True)
        # np.testing.assert_equal(GLSresults.num_iterations, 1)

        input_I_map = np.ma.MaskedArray(
            data=initfloat.const_I_map,
            dtype=initfloat.dtype,
            mask=~PTS.pixel_flag,
            fill_value=-1.6375e30,
        )
        input_Q_map = np.ma.MaskedArray(
            data=initfloat.const_Q_map,
            dtype=initfloat.dtype,
            mask=~PTS.pixel_flag,
            fill_value=-1.6375e30,
        )
        input_U_map = np.ma.MaskedArray(
            data=initfloat.const_U_map,
            dtype=initfloat.dtype,
            mask=~PTS.pixel_flag,
            fill_value=-1.6375e30,
        )

        np.testing.assert_allclose(
            GLSresults.GLS_maps[0],
            input_I_map,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            GLSresults.GLS_maps[1],
            input_Q_map,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            GLSresults.GLS_maps[2],
            input_U_map,
            rtol=rtol,
            atol=atol,
        )


class TestGLSMapMakers_rand_maps:
    def test_GLSMapMakers_I_rand_map(self, setup_scan):
        initint, initfloat = setup_scan

        tol = TOLERANCES[initfloat.dtype]
        rtol, atol = tol["rtol"], tol["atol"]
        solver_type = brahmap.SolverType.I

        tod = np.zeros(initint.nsamples, dtype=initfloat.dtype)

        # scan the sky
        for idx, pointings in enumerate(initint.pointings):
            tod[idx] += initfloat.rand_I_map[pointings]

        GLSparams = brahmap.core.GLSParameters(
            solver_type=solver_type,
            isolver_max_iterations=5,
            return_hit_map=False,
            return_processed_samples=True,
        )

        PTS, GLSresults = brahmap.core.compute_GLS_maps(
            npix=initint.npix,
            pointings=initint.pointings,
            time_ordered_data=tod,
            pointings_flag=initint.pointings_flag,
            dtype_float=initfloat.dtype,
            gls_parameters=GLSparams,
            update_pointings_inplace=False,
        )

        np.testing.assert_equal(GLSresults.convergence_status, True)
        # np.testing.assert_equal(GLSresults.num_iterations, 1)

        input_I_map = np.ma.MaskedArray(
            data=initfloat.rand_I_map,
            dtype=initfloat.dtype,
            mask=~PTS.pixel_flag,
            fill_value=-1.6375e30,
        )

        np.testing.assert_allclose(
            GLSresults.GLS_maps[0],
            input_I_map,
            rtol=rtol,
            atol=atol,
        )

    def test_GLSMapMakers_QU_rand_map(self, setup_scan):
        initint, initfloat = setup_scan

        tol = TOLERANCES[initfloat.dtype]
        rtol, atol = tol["rtol"], tol["atol"]
        solver_type = brahmap.SolverType.QU

        tod = np.zeros(initint.nsamples, dtype=initfloat.dtype)

        sin2phi = np.empty(initint.nsamples, dtype=initfloat.dtype)
        cos2phi = np.empty(initint.nsamples, dtype=initfloat.dtype)
        brahmap.math.sin(initint.nsamples, 2.0 * initfloat.pol_angles, sin2phi)
        brahmap.math.cos(initint.nsamples, 2.0 * initfloat.pol_angles, cos2phi)

        # scan the sky
        for idx, pointings in enumerate(initint.pointings):
            tod[idx] += initfloat.rand_Q_map[pointings] * cos2phi[idx]
            tod[idx] += initfloat.rand_U_map[pointings] * sin2phi[idx]

        GLSparams = brahmap.core.GLSParameters(
            solver_type=solver_type,
            isolver_max_iterations=5,
            return_hit_map=False,
            return_processed_samples=True,
        )

        PTS, GLSresults = brahmap.core.compute_GLS_maps(
            npix=initint.npix,
            pointings=initint.pointings,
            time_ordered_data=tod,
            pointings_flag=initint.pointings_flag,
            pol_angles=initfloat.pol_angles,
            dtype_float=initfloat.dtype,
            gls_parameters=GLSparams,
            update_pointings_inplace=False,
        )

        np.testing.assert_equal(GLSresults.convergence_status, True)
        # np.testing.assert_equal(GLSresults.num_iterations, 1)

        input_Q_map = np.ma.MaskedArray(
            data=initfloat.rand_Q_map,
            dtype=initfloat.dtype,
            mask=~PTS.pixel_flag,
            fill_value=-1.6375e30,
        )
        input_U_map = np.ma.MaskedArray(
            data=initfloat.rand_U_map,
            dtype=initfloat.dtype,
            mask=~PTS.pixel_flag,
            fill_value=-1.6375e30,
        )

        np.testing.assert_allclose(
            GLSresults.GLS_maps[0],
            input_Q_map,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            GLSresults.GLS_maps[1],
            input_U_map,
            rtol=rtol,
            atol=atol,
        )

    def test_GLSMapMakers_IQU_rand_map(self, setup_scan):
        initint, initfloat = setup_scan

        tol = TOLERANCES[initfloat.dtype]
        rtol, atol = tol["rtol"], tol["atol"]
        solver_type = brahmap.SolverType.IQU

        tod = np.zeros(initint.nsamples, dtype=initfloat.dtype)

        sin2phi = np.empty(initint.nsamples, dtype=initfloat.dtype)
        cos2phi = np.empty(initint.nsamples, dtype=initfloat.dtype)
        brahmap.math.sin(initint.nsamples, 2.0 * initfloat.pol_angles, sin2phi)
        brahmap.math.cos(initint.nsamples, 2.0 * initfloat.pol_angles, cos2phi)

        # scan the sky
        for idx, pointings in enumerate(initint.pointings):
            tod[idx] += initfloat.rand_I_map[pointings]
            tod[idx] += initfloat.rand_Q_map[pointings] * cos2phi[idx]
            tod[idx] += initfloat.rand_U_map[pointings] * sin2phi[idx]

        GLSparams = brahmap.core.GLSParameters(
            solver_type=solver_type,
            isolver_max_iterations=5,
            return_hit_map=False,
            return_processed_samples=True,
        )

        PTS, GLSresults = brahmap.core.compute_GLS_maps(
            npix=initint.npix,
            pointings=initint.pointings,
            time_ordered_data=tod,
            pointings_flag=initint.pointings_flag,
            pol_angles=initfloat.pol_angles,
            dtype_float=initfloat.dtype,
            gls_parameters=GLSparams,
            update_pointings_inplace=False,
        )

        np.testing.assert_equal(GLSresults.convergence_status, True)
        # np.testing.assert_equal(GLSresults.num_iterations, 1)

        input_I_map = np.ma.MaskedArray(
            data=initfloat.rand_I_map,
            dtype=initfloat.dtype,
            mask=~PTS.pixel_flag,
            fill_value=-1.6375e30,
        )
        input_Q_map = np.ma.MaskedArray(
            data=initfloat.rand_Q_map,
            dtype=initfloat.dtype,
            mask=~PTS.pixel_flag,
            fill_value=-1.6375e30,
        )
        input_U_map = np.ma.MaskedArray(
            data=initfloat.rand_U_map,
            dtype=initfloat.dtype,
            mask=~PTS.pixel_flag,
            fill_value=-1.6375e30,
        )

        np.testing.assert_allclose(
            GLSresults.GLS_maps[0],
            input_I_map,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            GLSresults.GLS_maps[1],
            input_Q_map,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            GLSresults.GLS_maps[2],
            input_U_map,
            rtol=rtol,
            atol=atol,
        )


class TestSharedMemGLSMapMakers:
    def _shmem_test(self, setup_scan, solver_type):
        initint, initfloat = setup_scan

        tol = TOLERANCES[initfloat.dtype]
        rtol, atol = tol["rtol"], tol["atol"]

        tod = np.zeros(initint.nsamples, dtype=initfloat.dtype)

        if solver_type == brahmap.SolverType.I:
            for idx, pointings in enumerate(initint.pointings):
                tod[idx] += initfloat.const_I_map[pointings]
        elif solver_type == brahmap.SolverType.QU:
            sin2phi = np.empty(initint.nsamples, dtype=initfloat.dtype)
            cos2phi = np.empty(initint.nsamples, dtype=initfloat.dtype)
            brahmap.math.sin(initint.nsamples, 2.0 * initfloat.pol_angles, sin2phi)
            brahmap.math.cos(initint.nsamples, 2.0 * initfloat.pol_angles, cos2phi)
            for idx, pointings in enumerate(initint.pointings):
                tod[idx] += initfloat.const_Q_map[pointings] * cos2phi[idx]
                tod[idx] += initfloat.const_U_map[pointings] * sin2phi[idx]
        else:  # IQU
            sin2phi = np.empty(initint.nsamples, dtype=initfloat.dtype)
            cos2phi = np.empty(initint.nsamples, dtype=initfloat.dtype)
            brahmap.math.sin(initint.nsamples, 2.0 * initfloat.pol_angles, sin2phi)
            brahmap.math.cos(initint.nsamples, 2.0 * initfloat.pol_angles, cos2phi)
            for idx, pointings in enumerate(initint.pointings):
                tod[idx] += initfloat.const_I_map[pointings]
                tod[idx] += initfloat.const_Q_map[pointings] * cos2phi[idx]
                tod[idx] += initfloat.const_U_map[pointings] * sin2phi[idx]

        # Standard solver
        GLSparams_std = brahmap.core.GLSParameters(
            solver_type=solver_type,
            isolver_max_iterations=10,
            return_hit_map=False,
            return_processed_samples=False,
        )
        res_std = brahmap.core.compute_GLS_maps(
            npix=initint.npix,
            pointings=initint.pointings,
            time_ordered_data=tod,
            pointings_flag=initint.pointings_flag,
            pol_angles=initfloat.pol_angles if solver_type > 1 else None,
            dtype_float=initfloat.dtype,
            gls_parameters=GLSparams_std,
            update_pointings_inplace=False,
            use_shared_memory=False,
        )

        # Shared memory solver with return_copy=True
        GLSparams_shm_copy = brahmap.core.GLSParameters(
            solver_type=solver_type,
            isolver_max_iterations=10,
            return_hit_map=False,
            return_processed_samples=False,
            shmem_return_copy=True,
        )
        res_shm_copy = brahmap.core.compute_GLS_maps(
            npix=initint.npix,
            pointings=initint.pointings,
            time_ordered_data=tod,
            pointings_flag=initint.pointings_flag,
            pol_angles=initfloat.pol_angles if solver_type > 1 else None,
            dtype_float=initfloat.dtype,
            gls_parameters=GLSparams_shm_copy,
            update_pointings_inplace=False,
            use_shared_memory=True,
            nproc_reduce=2,
        )

        # Shared memory solver with return_copy=False
        GLSparams_shm_nocopy = brahmap.core.GLSParameters(
            solver_type=solver_type,
            isolver_max_iterations=10,
            return_hit_map=False,
            return_processed_samples=False,
            shmem_return_copy=False,
        )
        res_shm_nocopy = brahmap.core.compute_GLS_maps(
            npix=initint.npix,
            pointings=initint.pointings,
            time_ordered_data=tod,
            pointings_flag=initint.pointings_flag,
            pol_angles=initfloat.pol_angles if solver_type > 1 else None,
            dtype_float=initfloat.dtype,
            gls_parameters=GLSparams_shm_nocopy,
            update_pointings_inplace=False,
            use_shared_memory=True,
            nproc_reduce=2,
        )

        # Check equivalence
        np.testing.assert_allclose(
            res_shm_copy.GLS_maps,
            res_std.GLS_maps,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            res_shm_nocopy.GLS_maps,
            res_std.GLS_maps,
            rtol=rtol,
            atol=atol,
        )

    def test_I(self, setup_scan):
        self._shmem_test(setup_scan, brahmap.SolverType.I)

    def test_QU(self, setup_scan):
        self._shmem_test(setup_scan, brahmap.SolverType.QU)

    def test_IQU(self, setup_scan):
        self._shmem_test(setup_scan, brahmap.SolverType.IQU)


if __name__ == "__main__":
    pytest.main(
        [
            f"{__file__}::TestGLSMapMakers_const_maps::test_GLSMapMakers_I_const_map",
            "-v",
            "-s",
        ]
    )
    pytest.main(
        [
            f"{__file__}::TestGLSMapMakers_const_maps::test_GLSMapMakers_QU_const_map",
            "-v",
            "-s",
        ]
    )
    pytest.main(
        [
            f"{__file__}::TestGLSMapMakers_const_maps::test_GLSMapMakers_IQU_const_map",
            "-v",
            "-s",
        ]
    )
    pytest.main(
        [
            f"{__file__}::TestGLSMapMakers_rand_maps::test_GLSMapMakers_I_rand_map",
            "-v",
            "-s",
        ]
    )
    pytest.main(
        [
            f"{__file__}::TestGLSMapMakers_rand_maps::test_GLSMapMakers_QU_rand_map",
            "-v",
            "-s",
        ]
    )
    pytest.main(
        [
            f"{__file__}::TestGLSMapMakers_rand_maps::test_GLSMapMakers_IQU_rand_map",
            "-v",
            "-s",
        ]
    )
    pytest.main(
        [
            f"{__file__}::TestSharedMemGLSMapMakers::test_I",
            "-v",
            "-s",
        ]
    )
    pytest.main(
        [
            f"{__file__}::TestSharedMemGLSMapMakers::test_QU",
            "-v",
            "-s",
        ]
    )
    pytest.main(
        [
            f"{__file__}::TestSharedMemGLSMapMakers::test_IQU",
            "-v",
            "-s",
        ]
    )
