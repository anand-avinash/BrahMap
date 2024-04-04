import pytest
import numpy as np

import brahmap.utilities as bmutils
import helper_ProcessTimeSamples as hpts


class InitCommonParams:
    np.random.seed(12345)
    npix = 128
    nsamples = npix * 6

    pointings_flag = np.ones(nsamples, dtype=bool)
    bad_samples = np.random.randint(low=0, high=nsamples, size=npix)
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


@pytest.mark.parametrize(
    "initint, initfloat, rtol",
    [
        (InitInt32Params(), InitFloat32Params(), 1.5e-4),
        (InitInt64Params(), InitFloat32Params(), 1.5e-4),
        (InitInt32Params(), InitFloat64Params(), 1.5e-5),
        (InitInt64Params(), InitFloat64Params(), 1.5e-5),
    ],
)
class TestProcessTimeSamples(InitCommonParams):
    def test_ProcessTimeSamples_I(self, initint, initfloat, rtol):
        solver_type = hpts.SolverType.I

        cpp_PTS = bmutils.ProcessTimeSamples(
            npix=self.npix,
            pointings=initint.pointings,
            pointings_flag=self.pointings_flag,
            solver_type=solver_type,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        py_PTS = hpts.ProcessTimeSamples(
            npix=self.npix,
            pointings=initint.pointings,
            pointings_flag=self.pointings_flag,
            solver_type=solver_type,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        np.testing.assert_array_equal(cpp_PTS.pointings, py_PTS.pointings)
        np.testing.assert_array_equal(cpp_PTS.pointings_flag, py_PTS.pointings_flag)
        np.testing.assert_equal(cpp_PTS.new_npix, py_PTS.new_npix)
        np.testing.assert_array_equal(cpp_PTS.pixel_mask, py_PTS.pixel_mask)
        np.testing.assert_allclose(
            cpp_PTS.weighted_counts, py_PTS.weighted_counts, rtol=rtol
        )
        np.testing.assert_array_equal(cpp_PTS.pixel_flag, py_PTS.pixel_flag)
        np.testing.assert_array_equal(cpp_PTS.old2new_pixel, py_PTS.old2new_pixel)

    def test_ProcessTimeSamples_QU(self, initint, initfloat, rtol):
        solver_type = hpts.SolverType.QU

        cpp_PTS = bmutils.ProcessTimeSamples(
            npix=self.npix,
            pointings=initint.pointings,
            pointings_flag=self.pointings_flag,
            solver_type=solver_type,
            pol_angles=initfloat.pol_angles,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        py_PTS = hpts.ProcessTimeSamples(
            npix=self.npix,
            pointings=initint.pointings,
            pointings_flag=self.pointings_flag,
            solver_type=solver_type,
            pol_angles=initfloat.pol_angles,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        np.testing.assert_array_equal(cpp_PTS.pointings, py_PTS.pointings)
        np.testing.assert_array_equal(cpp_PTS.pointings_flag, py_PTS.pointings_flag)
        np.testing.assert_equal(cpp_PTS.new_npix, py_PTS.new_npix)
        np.testing.assert_array_equal(cpp_PTS.pixel_mask, py_PTS.pixel_mask)
        np.testing.assert_allclose(cpp_PTS.sin2phi, py_PTS.sin2phi, rtol=rtol)
        np.testing.assert_allclose(cpp_PTS.cos2phi, py_PTS.cos2phi, rtol=rtol)
        np.testing.assert_allclose(
            cpp_PTS.weighted_counts, py_PTS.weighted_counts, rtol=rtol
        )
        np.testing.assert_allclose(
            cpp_PTS.weighted_sin_sq, py_PTS.weighted_sin_sq, rtol=rtol
        )
        np.testing.assert_allclose(
            cpp_PTS.weighted_cos_sq, py_PTS.weighted_cos_sq, rtol=rtol
        )
        np.testing.assert_allclose(
            cpp_PTS.weighted_sincos, py_PTS.weighted_sincos, rtol=rtol
        )
        np.testing.assert_array_equal(cpp_PTS.pixel_flag, py_PTS.pixel_flag)
        np.testing.assert_array_equal(cpp_PTS.old2new_pixel, py_PTS.old2new_pixel)

    def test_ProcessTimeSamples_IQU(self, initint, initfloat, rtol):
        solver_type = hpts.SolverType.IQU

        cpp_PTS = bmutils.ProcessTimeSamples(
            npix=self.npix,
            pointings=initint.pointings,
            pointings_flag=self.pointings_flag,
            solver_type=solver_type,
            pol_angles=initfloat.pol_angles,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        py_PTS = hpts.ProcessTimeSamples(
            npix=self.npix,
            pointings=initint.pointings,
            pointings_flag=self.pointings_flag,
            solver_type=solver_type,
            pol_angles=initfloat.pol_angles,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        np.testing.assert_array_equal(cpp_PTS.pointings, py_PTS.pointings)
        np.testing.assert_array_equal(cpp_PTS.pointings_flag, py_PTS.pointings_flag)
        np.testing.assert_equal(cpp_PTS.new_npix, py_PTS.new_npix)
        np.testing.assert_array_equal(cpp_PTS.pixel_mask, py_PTS.pixel_mask)
        np.testing.assert_allclose(cpp_PTS.sin2phi, py_PTS.sin2phi, rtol=rtol)
        np.testing.assert_allclose(cpp_PTS.cos2phi, py_PTS.cos2phi, rtol=rtol)
        np.testing.assert_allclose(
            cpp_PTS.weighted_counts, py_PTS.weighted_counts, rtol=rtol
        )
        np.testing.assert_allclose(
            cpp_PTS.weighted_sin_sq, py_PTS.weighted_sin_sq, rtol=rtol
        )
        np.testing.assert_allclose(
            cpp_PTS.weighted_cos_sq, py_PTS.weighted_cos_sq, rtol=rtol
        )
        np.testing.assert_allclose(
            cpp_PTS.weighted_sincos, py_PTS.weighted_sincos, rtol=rtol
        )
        np.testing.assert_allclose(cpp_PTS.weighted_sin, py_PTS.weighted_sin, rtol=rtol)
        np.testing.assert_allclose(cpp_PTS.weighted_cos, py_PTS.weighted_cos, rtol=rtol)
        np.testing.assert_array_equal(cpp_PTS.pixel_flag, py_PTS.pixel_flag)
        np.testing.assert_array_equal(cpp_PTS.old2new_pixel, py_PTS.old2new_pixel)


if __name__ == "__main__":
    pytest.main(
        [f"{__file__}::TestProcessTimeSamples::test_ProcessTimeSamples_I", "-v", "-s"]
    )
    pytest.main(
        [f"{__file__}::TestProcessTimeSamples::test_ProcessTimeSamples_QU", "-v", "-s"]
    )
    pytest.main(
        [f"{__file__}::TestProcessTimeSamples::test_ProcessTimeSamples_IQU", "-v", "-s"]
    )
