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

brahmap.Initialize()


class InitCommonParams:
    np.random.seed(123345 + brahmap.bMPI.rank)

    # random seed to generate same random map on all the processes
    rand_map_seed = 6454

    npix = 128
    nsamples_global = npix * 6

    div, rem = divmod(nsamples_global, brahmap.bMPI.size)
    nsamples = div + (brahmap.bMPI.rank < rem)

    nbad_pixels_global = npix
    div, rem = divmod(nbad_pixels_global, brahmap.bMPI.size)
    nbad_pixels = div + (brahmap.bMPI.rank < rem)

    pointings_flag = np.ones(nsamples, dtype=bool)
    bad_samples = np.random.randint(low=0, high=nsamples, size=nbad_pixels)
    pointings_flag[bad_samples] = False


class InitInt32Params(InitCommonParams):
    def __init__(self) -> None:
        super().__init__()

        self.dtype = np.int32
        self.pointings = np.random.randint(
            low=0, high=self.npix, size=self.nsamples, dtype=self.dtype
        )


class InitInt64Params(InitCommonParams):
    def __init__(self) -> None:
        super().__init__()

        self.dtype = np.int64
        self.pointings = np.random.randint(
            low=0, high=self.npix, size=self.nsamples, dtype=self.dtype
        )


class InitFloat32Params(InitCommonParams):
    def __init__(self) -> None:
        super().__init__()

        self.dtype = np.float32
        self.noise_weights = np.random.random(size=self.nsamples).astype(
            dtype=self.dtype
        )
        self.pol_angles = np.random.uniform(
            low=-np.pi / 2.0, high=np.pi / 2.0, size=self.nsamples
        ).astype(dtype=self.dtype)

        # constant maps
        self.const_I_map = np.ones(self.npix, dtype=self.dtype) * 7.0
        self.const_Q_map = np.ones(self.npix, dtype=self.dtype) * 5.0
        self.const_U_map = np.ones(self.npix, dtype=self.dtype) * 3.0

        # random maps
        np.random.seed(self.rand_map_seed)
        self.rand_I_map = np.random.uniform(low=-7.0, high=7.0, size=self.npix).astype(
            dtype=self.dtype
        )
        self.rand_Q_map = np.random.uniform(low=-5.0, high=5.0, size=self.npix).astype(
            dtype=self.dtype
        )
        self.rand_U_map = np.random.uniform(low=-3.0, high=3.0, size=self.npix).astype(
            dtype=self.dtype
        )


class InitFloat64Params(InitCommonParams):
    def __init__(self) -> None:
        super().__init__()

        self.dtype = np.float64
        self.noise_weights = np.random.random(size=self.nsamples).astype(
            dtype=self.dtype
        )
        self.pol_angles = np.random.uniform(
            low=-np.pi / 2.0, high=np.pi / 2.0, size=self.nsamples
        ).astype(dtype=self.dtype)

        # constant maps
        self.const_I_map = np.ones(self.npix, dtype=self.dtype) * 7.0
        self.const_Q_map = np.ones(self.npix, dtype=self.dtype) * 5.0
        self.const_U_map = np.ones(self.npix, dtype=self.dtype) * 3.0

        # random maps
        np.random.seed(self.rand_map_seed)
        self.rand_I_map = np.random.uniform(low=-7.0, high=7.0, size=self.npix).astype(
            dtype=self.dtype
        )
        self.rand_Q_map = np.random.uniform(low=-5.0, high=5.0, size=self.npix).astype(
            dtype=self.dtype
        )
        self.rand_U_map = np.random.uniform(low=-3.0, high=3.0, size=self.npix).astype(
            dtype=self.dtype
        )


# Initializing the parameter classes
initint32 = InitInt32Params()
initint64 = InitInt64Params()
initfloat32 = InitFloat32Params()
initfloat64 = InitFloat64Params()


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
        solver_type = brahmap.SolverType.I

        tod = np.zeros(self.nsamples, dtype=initfloat.dtype)

        # scan the sky
        for idx, pointings in enumerate(initint.pointings):
            tod[idx] += initfloat.const_I_map[pointings]

        GLSparams = brahmap.mapmakers.GLSParameters(
            solver_type=solver_type,
            preconditioner_max_iterations=5,
            return_hit_map=False,
        )

        PTS, GLSresults = brahmap.mapmakers.compute_GLS_maps(
            npix=self.npix,
            pointings=initint.pointings,
            time_ordered_data=tod,
            pointings_flag=self.pointings_flag,
            dtype_float=initfloat.dtype,
            GLSParameters=GLSparams,
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
        solver_type = brahmap.SolverType.QU

        tod = np.zeros(self.nsamples, dtype=initfloat.dtype)

        # scan the sky
        for idx, pointings in enumerate(initint.pointings):
            tod[idx] += initfloat.const_Q_map[pointings] * np.cos(
                2.0 * initfloat.pol_angles[idx]
            )
            tod[idx] += initfloat.const_U_map[pointings] * np.sin(
                2.0 * initfloat.pol_angles[idx]
            )

        GLSparams = brahmap.mapmakers.GLSParameters(
            solver_type=solver_type,
            preconditioner_max_iterations=5,
            return_hit_map=False,
        )

        PTS, GLSresults = brahmap.mapmakers.compute_GLS_maps(
            npix=self.npix,
            pointings=initint.pointings,
            time_ordered_data=tod,
            pointings_flag=self.pointings_flag,
            pol_angles=initfloat.pol_angles,
            dtype_float=initfloat.dtype,
            GLSParameters=GLSparams,
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
        solver_type = brahmap.SolverType.IQU

        tod = np.zeros(self.nsamples, dtype=initfloat.dtype)

        # scan the sky
        for idx, pointings in enumerate(initint.pointings):
            tod[idx] += initfloat.const_I_map[pointings]
            tod[idx] += initfloat.const_Q_map[pointings] * np.cos(
                2.0 * initfloat.pol_angles[idx]
            )
            tod[idx] += initfloat.const_U_map[pointings] * np.sin(
                2.0 * initfloat.pol_angles[idx]
            )

        GLSparams = brahmap.mapmakers.GLSParameters(
            solver_type=solver_type,
            preconditioner_max_iterations=5,
            return_hit_map=False,
        )

        PTS, GLSresults = brahmap.mapmakers.compute_GLS_maps(
            npix=self.npix,
            pointings=initint.pointings,
            time_ordered_data=tod,
            pointings_flag=self.pointings_flag,
            pol_angles=initfloat.pol_angles,
            dtype_float=initfloat.dtype,
            GLSParameters=GLSparams,
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

        GLSparams = brahmap.mapmakers.GLSParameters(
            solver_type=solver_type,
            preconditioner_max_iterations=5,
            return_hit_map=False,
        )

        PTS, GLSresults = brahmap.mapmakers.compute_GLS_maps(
            npix=self.npix,
            pointings=initint.pointings,
            time_ordered_data=tod,
            pointings_flag=self.pointings_flag,
            dtype_float=initfloat.dtype,
            GLSParameters=GLSparams,
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

        # scan the sky
        for idx, pointings in enumerate(initint.pointings):
            tod[idx] += initfloat.rand_Q_map[pointings] * np.cos(
                2.0 * initfloat.pol_angles[idx]
            )
            tod[idx] += initfloat.rand_U_map[pointings] * np.sin(
                2.0 * initfloat.pol_angles[idx]
            )

        GLSparams = brahmap.mapmakers.GLSParameters(
            solver_type=solver_type,
            preconditioner_max_iterations=5,
            return_hit_map=False,
        )

        PTS, GLSresults = brahmap.mapmakers.compute_GLS_maps(
            npix=self.npix,
            pointings=initint.pointings,
            time_ordered_data=tod,
            pointings_flag=self.pointings_flag,
            pol_angles=initfloat.pol_angles,
            dtype_float=initfloat.dtype,
            GLSParameters=GLSparams,
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

        # scan the sky
        for idx, pointings in enumerate(initint.pointings):
            tod[idx] += initfloat.rand_I_map[pointings]
            tod[idx] += initfloat.rand_Q_map[pointings] * np.cos(
                2.0 * initfloat.pol_angles[idx]
            )
            tod[idx] += initfloat.rand_U_map[pointings] * np.sin(
                2.0 * initfloat.pol_angles[idx]
            )

        GLSparams = brahmap.mapmakers.GLSParameters(
            solver_type=solver_type,
            preconditioner_max_iterations=5,
            return_hit_map=False,
        )

        PTS, GLSresults = brahmap.mapmakers.compute_GLS_maps(
            npix=self.npix,
            pointings=initint.pointings,
            time_ordered_data=tod,
            pointings_flag=self.pointings_flag,
            pol_angles=initfloat.pol_angles,
            dtype_float=initfloat.dtype,
            GLSParameters=GLSparams,
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
