import pytest
import numpy as np
import brahmap
from test_ProcessTimeSamples import (
    InitCommonParams,
    initint32,
    initint64,
    initfloat32,
    initfloat64,
)


@pytest.mark.parametrize(
    "initint, initfloat, nproc_reduce, rtol, atol",
    [
        (initint32, initfloat32, 1, 1.5e-3, 1.0e-5),
        (initint32, initfloat32, 2, 1.5e-3, 1.0e-5),
        (initint64, initfloat64, 1, 1.5e-5, 1.0e-10),
        (initint64, initfloat64, 2, 1.5e-5, 1.0e-10),
    ],
)
class TestSharedMemProcessTimeSamples(InitCommonParams):
    def test_SharedMemProcessTimeSamples_I(
        self, initint, initfloat, nproc_reduce, rtol, atol
    ):
        solver_type = brahmap.core.SolverType.I

        shm_PTS = brahmap.core.SharedMemProcessTimeSamples(
            npix=self.npix,
            pointings=initint.pointings,
            pointings_flag=self.pointings_flag,
            solver_type=solver_type,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
            nproc_reduce=nproc_reduce,
        )

        std_PTS = brahmap.core.ProcessTimeSamples(
            npix=self.npix,
            pointings=initint.pointings,
            pointings_flag=self.pointings_flag,
            solver_type=solver_type,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        np.testing.assert_array_equal(shm_PTS.pointings, std_PTS.pointings)
        np.testing.assert_array_equal(shm_PTS.pointings_flag, std_PTS.pointings_flag)
        np.testing.assert_equal(shm_PTS.new_npix, std_PTS.new_npix)
        np.testing.assert_array_equal(shm_PTS.observed_pixels, std_PTS.observed_pixels)
        np.testing.assert_allclose(
            shm_PTS.weighted_counts,
            std_PTS.weighted_counts,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_array_equal(shm_PTS.pixel_flag, std_PTS.pixel_flag)
        np.testing.assert_array_equal(shm_PTS.old2new_pixel, std_PTS.old2new_pixel)

    def test_SharedMemProcessTimeSamples_QU(
        self, initint, initfloat, nproc_reduce, rtol, atol
    ):
        solver_type = brahmap.core.SolverType.QU

        shm_PTS = brahmap.core.SharedMemProcessTimeSamples(
            npix=self.npix,
            pointings=initint.pointings,
            pointings_flag=self.pointings_flag,
            solver_type=solver_type,
            pol_angles=initfloat.pol_angles,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
            nproc_reduce=nproc_reduce,
        )

        std_PTS = brahmap.core.ProcessTimeSamples(
            npix=self.npix,
            pointings=initint.pointings,
            pointings_flag=self.pointings_flag,
            solver_type=solver_type,
            pol_angles=initfloat.pol_angles,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        np.testing.assert_array_equal(shm_PTS.pointings, std_PTS.pointings)
        np.testing.assert_array_equal(shm_PTS.pointings_flag, std_PTS.pointings_flag)
        np.testing.assert_equal(shm_PTS.new_npix, std_PTS.new_npix)
        np.testing.assert_array_equal(shm_PTS.observed_pixels, std_PTS.observed_pixels)
        np.testing.assert_allclose(
            shm_PTS.sin2phi,
            std_PTS.sin2phi,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            shm_PTS.cos2phi,
            std_PTS.cos2phi,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            shm_PTS.weighted_counts,
            std_PTS.weighted_counts,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            shm_PTS.weighted_sin_sq,
            std_PTS.weighted_sin_sq,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            shm_PTS.weighted_cos_sq,
            std_PTS.weighted_cos_sq,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            shm_PTS.weighted_sincos,
            std_PTS.weighted_sincos,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            shm_PTS.one_over_determinant,
            std_PTS.one_over_determinant,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_array_equal(shm_PTS.pixel_flag, std_PTS.pixel_flag)
        np.testing.assert_array_equal(shm_PTS.old2new_pixel, std_PTS.old2new_pixel)

    def test_SharedMemProcessTimeSamples_IQU(
        self, initint, initfloat, nproc_reduce, rtol, atol
    ):
        solver_type = brahmap.core.SolverType.IQU

        shm_PTS = brahmap.core.SharedMemProcessTimeSamples(
            npix=self.npix,
            pointings=initint.pointings,
            pointings_flag=self.pointings_flag,
            solver_type=solver_type,
            pol_angles=initfloat.pol_angles,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
            nproc_reduce=nproc_reduce,
        )

        std_PTS = brahmap.core.ProcessTimeSamples(
            npix=self.npix,
            pointings=initint.pointings,
            pointings_flag=self.pointings_flag,
            solver_type=solver_type,
            pol_angles=initfloat.pol_angles,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        np.testing.assert_array_equal(shm_PTS.pointings, std_PTS.pointings)
        np.testing.assert_array_equal(shm_PTS.pointings_flag, std_PTS.pointings_flag)
        np.testing.assert_equal(shm_PTS.new_npix, std_PTS.new_npix)
        np.testing.assert_array_equal(shm_PTS.observed_pixels, std_PTS.observed_pixels)
        np.testing.assert_allclose(
            shm_PTS.sin2phi,
            std_PTS.sin2phi,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            shm_PTS.cos2phi,
            std_PTS.cos2phi,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            shm_PTS.weighted_counts,
            std_PTS.weighted_counts,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            shm_PTS.weighted_sin_sq,
            std_PTS.weighted_sin_sq,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            shm_PTS.weighted_cos_sq,
            std_PTS.weighted_cos_sq,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            shm_PTS.weighted_sincos,
            std_PTS.weighted_sincos,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            shm_PTS.weighted_sin,
            std_PTS.weighted_sin,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            shm_PTS.weighted_cos,
            std_PTS.weighted_cos,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            shm_PTS.one_over_determinant,
            std_PTS.one_over_determinant,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_array_equal(shm_PTS.pixel_flag, std_PTS.pixel_flag)
        np.testing.assert_array_equal(shm_PTS.old2new_pixel, std_PTS.old2new_pixel)


if __name__ == "__main__":
    pytest.main(
        [
            f"{__file__}::TestSharedMemProcessTimeSamples::test_SharedMemProcessTimeSamples_I",
            "-v",
            "-s",
        ]
    )

    pytest.main(
        [
            f"{__file__}::TestSharedMemProcessTimeSamples::test_SharedMemProcessTimeSamples_QU",
            "-v",
            "-s",
        ]
    )

    pytest.main(
        [
            f"{__file__}::TestSharedMemProcessTimeSamples::test_SharedMemProcessTimeSamples_IQU",
            "-v",
            "-s",
        ]
    )
