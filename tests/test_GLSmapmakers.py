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
import time
import numpy as np

import brahmap


class InitCommonParams:
    rng = np.random.default_rng(seed=[123345, brahmap.MPI_UTILS.rank])

    # random seed to generate common random map on all the processes
    rand_map_seed = 6454

    npix = 128
    nsamples_global = npix * 6

    div, rem = divmod(nsamples_global, brahmap.MPI_UTILS.size)
    nsamples = div + (brahmap.MPI_UTILS.rank < rem)

    nbad_pixels_global = npix
    div, rem = divmod(nbad_pixels_global, brahmap.MPI_UTILS.size)
    nbad_pixels = div + (brahmap.MPI_UTILS.rank < rem)

    pointings_flag = np.ones(nsamples, dtype=bool)
    bad_samples = rng.integers(low=0, high=nsamples, size=nbad_pixels)
    pointings_flag[bad_samples] = False


class InitIntegerParams(InitCommonParams):
    def __init__(self, dtype_int) -> None:
        super().__init__()

        self.int_rng = np.random.default_rng(seed=[1234345, brahmap.MPI_UTILS.rank])
        self.dtype = dtype_int
        self.pointings = self.int_rng.integers(
            low=0, high=self.npix, size=self.nsamples, dtype=self.dtype
        )


class InitFloatParams(InitCommonParams):
    def __init__(self, dtype_float) -> None:
        super().__init__()

        self.float_rng = np.random.default_rng(seed=[1237345, brahmap.MPI_UTILS.rank])

        self.dtype = dtype_float
        self.pol_angles = self.float_rng.uniform(
            low=-np.pi / 2.0, high=np.pi / 2.0, size=self.nsamples
        ).astype(dtype=self.dtype)
        self.noise_weights = self.float_rng.random(size=self.nsamples, dtype=self.dtype)

        # constant maps
        self.const_I_map = np.ones(self.npix, dtype=self.dtype) * 7.0
        self.const_Q_map = np.ones(self.npix, dtype=self.dtype) * 5.0
        self.const_U_map = np.ones(self.npix, dtype=self.dtype) * 3.0

        # random maps
        rng_map = np.random.default_rng(seed=self.rand_map_seed)
        self.rand_I_map = rng_map.uniform(low=-7.0, high=7.0, size=self.npix).astype(
            dtype=self.dtype
        )
        self.rand_Q_map = rng_map.uniform(low=-5.0, high=5.0, size=self.npix).astype(
            dtype=self.dtype
        )
        self.rand_U_map = rng_map.uniform(low=-3.0, high=3.0, size=self.npix).astype(
            dtype=self.dtype
        )


# Initializing the parameter classes

initint32 = InitIntegerParams(dtype_int=np.int32)
initint64 = InitIntegerParams(dtype_int=np.int64)
initfloat32 = InitFloatParams(dtype_float=np.float32)
initfloat64 = InitFloatParams(dtype_float=np.float64)


# @pytest.mark.skip(
#     reason="Unlike other tests, this one is producing"
#     "different result on each execution. Under investigation!"
# )
@pytest.mark.parametrize(
    "initint, initfloat, rtol",
    [
        (initint32, initfloat32, 1.5e-3),
        (initint64, initfloat32, 1.5e-3),
        (initint32, initfloat64, 1.5e-5),
        (initint64, initfloat64, 1.5e-5),
    ],
)
class TestGLSMapMakers_const_maps(InitCommonParams):
    def test_GLSMapMakers_I_const_map(self, initint, initfloat, rtol):
        time.sleep(1)
        solver_type = brahmap.SolverType.I

        tod = np.zeros(self.nsamples, dtype=initfloat.dtype)

        # scan the sky
        for idx, pointings in enumerate(initint.pointings):
            tod[idx] += initfloat.const_I_map[pointings]

        GLSparams = brahmap.core.GLSParameters(
            solver_type=solver_type,
            preconditioner_max_iterations=5,
            return_hit_map=False,
            return_processed_samples=True,
        )

        PTS, GLSresults = brahmap.core.compute_GLS_maps(
            npix=self.npix,
            pointings=initint.pointings,
            time_ordered_data=tod,
            pointings_flag=self.pointings_flag,
            dtype_float=initfloat.dtype,
            gls_parameters=GLSparams,
            update_pointings_inplace=False,
        )

        np.testing.assert_equal(GLSresults.convergence_status, True)
        np.testing.assert_equal(GLSresults.num_iterations, 1)

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
        )

    def test_GLSMapMakers_QU_const_map(self, initint, initfloat, rtol):
        time.sleep(1)
        solver_type = brahmap.SolverType.QU

        tod = np.zeros(self.nsamples, dtype=initfloat.dtype)

        sin2phi = np.empty(self.nsamples, dtype=initfloat.dtype)
        cos2phi = np.empty(self.nsamples, dtype=initfloat.dtype)
        brahmap.math.sin(self.nsamples, 2.0 * initfloat.pol_angles, sin2phi)
        brahmap.math.cos(self.nsamples, 2.0 * initfloat.pol_angles, cos2phi)

        # scan the sky
        for idx, pointings in enumerate(initint.pointings):
            tod[idx] += initfloat.const_Q_map[pointings] * cos2phi[idx]
            tod[idx] += initfloat.const_U_map[pointings] * sin2phi[idx]

        GLSparams = brahmap.core.GLSParameters(
            solver_type=solver_type,
            preconditioner_max_iterations=5,
            return_hit_map=False,
            return_processed_samples=True,
        )

        PTS, GLSresults = brahmap.core.compute_GLS_maps(
            npix=self.npix,
            pointings=initint.pointings,
            time_ordered_data=tod,
            pointings_flag=self.pointings_flag,
            pol_angles=initfloat.pol_angles,
            dtype_float=initfloat.dtype,
            gls_parameters=GLSparams,
            update_pointings_inplace=False,
        )

        np.testing.assert_equal(GLSresults.convergence_status, True)
        np.testing.assert_equal(GLSresults.num_iterations, 1)

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
        )
        np.testing.assert_allclose(
            GLSresults.GLS_maps[1],
            input_U_map,
            rtol=rtol,
        )

    def test_GLSMapMakers_IQU_const_map(self, initint, initfloat, rtol):
        time.sleep(1)
        solver_type = brahmap.SolverType.IQU

        tod = np.zeros(self.nsamples, dtype=initfloat.dtype)

        sin2phi = np.empty(self.nsamples, dtype=initfloat.dtype)
        cos2phi = np.empty(self.nsamples, dtype=initfloat.dtype)
        brahmap.math.sin(self.nsamples, 2.0 * initfloat.pol_angles, sin2phi)
        brahmap.math.cos(self.nsamples, 2.0 * initfloat.pol_angles, cos2phi)

        # scan the sky
        for idx, pointings in enumerate(initint.pointings):
            tod[idx] += initfloat.const_I_map[pointings]
            tod[idx] += initfloat.const_Q_map[pointings] * cos2phi[idx]
            tod[idx] += initfloat.const_U_map[pointings] * sin2phi[idx]

        GLSparams = brahmap.core.GLSParameters(
            solver_type=solver_type,
            preconditioner_max_iterations=5,
            return_hit_map=False,
            return_processed_samples=True,
        )

        PTS, GLSresults = brahmap.core.compute_GLS_maps(
            npix=self.npix,
            pointings=initint.pointings,
            time_ordered_data=tod,
            pointings_flag=self.pointings_flag,
            pol_angles=initfloat.pol_angles,
            dtype_float=initfloat.dtype,
            gls_parameters=GLSparams,
            update_pointings_inplace=False,
        )

        np.testing.assert_equal(GLSresults.convergence_status, True)
        np.testing.assert_equal(GLSresults.num_iterations, 1)

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
        )
        np.testing.assert_allclose(
            GLSresults.GLS_maps[1],
            input_Q_map,
            rtol=rtol,
        )
        np.testing.assert_allclose(
            GLSresults.GLS_maps[2],
            input_U_map,
            rtol=rtol,
        )


# @pytest.mark.skip(
#     reason="Unlike other tests, this one is producing"
#     "different result on each execution. Under investigation!"
# )
@pytest.mark.parametrize(
    "initint, initfloat, rtol",
    [
        (initint32, initfloat32, 1.5e-3),
        (initint64, initfloat32, 1.5e-3),
        (initint32, initfloat64, 1.5e-5),
        (initint64, initfloat64, 1.5e-5),
    ],
)
class TestGLSMapMakers_rand_maps(InitCommonParams):
    def test_GLSMapMakers_I_rand_map(self, initint, initfloat, rtol):
        solver_type = brahmap.SolverType.I

        tod = np.zeros(self.nsamples, dtype=initfloat.dtype)

        # scan the sky
        for idx, pointings in enumerate(initint.pointings):
            tod[idx] += initfloat.rand_I_map[pointings]

        GLSparams = brahmap.core.GLSParameters(
            solver_type=solver_type,
            preconditioner_max_iterations=5,
            return_hit_map=False,
            return_processed_samples=True,
        )

        PTS, GLSresults = brahmap.core.compute_GLS_maps(
            npix=self.npix,
            pointings=initint.pointings,
            time_ordered_data=tod,
            pointings_flag=self.pointings_flag,
            dtype_float=initfloat.dtype,
            gls_parameters=GLSparams,
            update_pointings_inplace=False,
        )

        np.testing.assert_equal(GLSresults.convergence_status, True)
        np.testing.assert_equal(GLSresults.num_iterations, 1)

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
        )

    def test_GLSMapMakers_QU_rand_map(self, initint, initfloat, rtol):
        solver_type = brahmap.SolverType.QU

        tod = np.zeros(self.nsamples, dtype=initfloat.dtype)

        sin2phi = np.empty(self.nsamples, dtype=initfloat.dtype)
        cos2phi = np.empty(self.nsamples, dtype=initfloat.dtype)
        brahmap.math.sin(self.nsamples, 2.0 * initfloat.pol_angles, sin2phi)
        brahmap.math.cos(self.nsamples, 2.0 * initfloat.pol_angles, cos2phi)

        # scan the sky
        for idx, pointings in enumerate(initint.pointings):
            tod[idx] += initfloat.rand_Q_map[pointings] * cos2phi[idx]
            tod[idx] += initfloat.rand_U_map[pointings] * sin2phi[idx]

        GLSparams = brahmap.core.GLSParameters(
            solver_type=solver_type,
            preconditioner_max_iterations=5,
            return_hit_map=False,
            return_processed_samples=True,
        )

        PTS, GLSresults = brahmap.core.compute_GLS_maps(
            npix=self.npix,
            pointings=initint.pointings,
            time_ordered_data=tod,
            pointings_flag=self.pointings_flag,
            pol_angles=initfloat.pol_angles,
            dtype_float=initfloat.dtype,
            gls_parameters=GLSparams,
            update_pointings_inplace=False,
        )

        np.testing.assert_equal(GLSresults.convergence_status, True)
        np.testing.assert_equal(GLSresults.num_iterations, 1)

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
        )
        np.testing.assert_allclose(
            GLSresults.GLS_maps[1],
            input_U_map,
            rtol=rtol,
        )

    def test_GLSMapMakers_IQU_rand_map(self, initint, initfloat, rtol):
        solver_type = brahmap.SolverType.IQU

        tod = np.zeros(self.nsamples, dtype=initfloat.dtype)

        sin2phi = np.empty(self.nsamples, dtype=initfloat.dtype)
        cos2phi = np.empty(self.nsamples, dtype=initfloat.dtype)
        brahmap.math.sin(self.nsamples, 2.0 * initfloat.pol_angles, sin2phi)
        brahmap.math.cos(self.nsamples, 2.0 * initfloat.pol_angles, cos2phi)

        # scan the sky
        for idx, pointings in enumerate(initint.pointings):
            tod[idx] += initfloat.rand_I_map[pointings]
            tod[idx] += initfloat.rand_Q_map[pointings] * cos2phi[idx]
            tod[idx] += initfloat.rand_U_map[pointings] * sin2phi[idx]

        GLSparams = brahmap.core.GLSParameters(
            solver_type=solver_type,
            preconditioner_max_iterations=5,
            return_hit_map=False,
            return_processed_samples=True,
        )

        PTS, GLSresults = brahmap.core.compute_GLS_maps(
            npix=self.npix,
            pointings=initint.pointings,
            time_ordered_data=tod,
            pointings_flag=self.pointings_flag,
            pol_angles=initfloat.pol_angles,
            dtype_float=initfloat.dtype,
            gls_parameters=GLSparams,
            update_pointings_inplace=False,
        )

        np.testing.assert_equal(GLSresults.convergence_status, True)
        np.testing.assert_equal(GLSresults.num_iterations, 1)

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
        )
        np.testing.assert_allclose(
            GLSresults.GLS_maps[1],
            input_Q_map,
            rtol=rtol,
        )
        np.testing.assert_allclose(
            GLSresults.GLS_maps[2],
            input_U_map,
            rtol=rtol,
        )


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
