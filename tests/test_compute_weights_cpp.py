import pytest
import numpy as np
import compute_weights

import helper_ComputeWeights as cw


class InitCommonParams:
    np.random.seed(1234)
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
class TestComputeWeights(InitCommonParams):
    def test_compute_weights_pol_I(self, initint, initfloat, rtol):
        cpp_weighted_counts = np.zeros(self.npix, dtype=initfloat.dtype)
        cpp_pixel_mask = np.zeros(self.npix, dtype=initint.dtype)
        cpp_old2new_pixel = np.zeros(self.npix, dtype=initint.dtype)
        cpp_pixel_flag = np.zeros(self.npix, dtype=bool)

        cpp_new_npix = compute_weights.compute_weights_pol_I(
            self.npix,
            self.nsamples,
            initint.pointings,
            self.pointings_flag,
            initfloat.noise_weights,
            cpp_weighted_counts,
            cpp_pixel_mask,
            cpp_old2new_pixel,
            cpp_pixel_flag,
        )

        (
            py_new_npix,
            py_weighted_counts,
            py_pixel_mask,
            py_old2new_pixel,
            py_pixel_flag,
        ) = cw.computeweights_pol_I(
            self.npix,
            self.nsamples,
            initint.pointings,
            self.pointings_flag,
            initfloat.noise_weights,
            dtype_float=initfloat.dtype,
        )

        cpp_pixel_mask.resize(cpp_new_npix, refcheck=False)

        np.testing.assert_equal(cpp_new_npix, py_new_npix)
        np.testing.assert_allclose(cpp_weighted_counts, py_weighted_counts, rtol=rtol)
        np.testing.assert_array_equal(cpp_pixel_mask, py_pixel_mask)
        np.testing.assert_array_equal(cpp_old2new_pixel, py_old2new_pixel)
        np.testing.assert_array_equal(cpp_pixel_flag, py_pixel_flag)

    def test_compute_weights_pol_QU(self, initint, initfloat, rtol):
        cpp_sin2phi = np.zeros(self.nsamples, dtype=initfloat.dtype)
        cpp_cos2phi = np.zeros(self.nsamples, dtype=initfloat.dtype)

        cpp_weighted_counts = np.zeros(self.npix, dtype=initfloat.dtype)
        cpp_weighted_sin_sq = np.zeros(self.npix, dtype=initfloat.dtype)
        cpp_weighted_cos_sq = np.zeros(self.npix, dtype=initfloat.dtype)
        cpp_weighted_sincos = np.zeros(self.npix, dtype=initfloat.dtype)

        compute_weights.compute_weights_pol_QU(
            self.nsamples,
            initint.pointings,
            self.pointings_flag,
            initfloat.noise_weights,
            initfloat.pol_angles,
            cpp_weighted_counts,
            cpp_sin2phi,
            cpp_cos2phi,
            cpp_weighted_sin_sq,
            cpp_weighted_cos_sq,
            cpp_weighted_sincos,
        )

        (
            py_weighted_counts,
            py_sin2phi,
            py_cos2phi,
            py_weighted_sin_sq,
            py_weighted_cos_sq,
            py_weighted_sincos,
        ) = cw.computeweights_pol_QU(
            self.npix,
            self.nsamples,
            initint.pointings,
            self.pointings_flag,
            initfloat.noise_weights,
            initfloat.pol_angles,
            dtype_float=initfloat.dtype,
        )

        np.testing.assert_allclose(cpp_weighted_counts, py_weighted_counts, rtol=rtol)
        np.testing.assert_allclose(cpp_sin2phi, py_sin2phi, rtol=rtol)
        np.testing.assert_allclose(cpp_cos2phi, py_cos2phi, rtol=rtol)
        np.testing.assert_allclose(cpp_weighted_sin_sq, py_weighted_sin_sq, rtol=rtol)
        np.testing.assert_allclose(cpp_weighted_cos_sq, py_weighted_cos_sq, rtol=rtol)
        np.testing.assert_allclose(cpp_weighted_sincos, py_weighted_sincos, rtol=rtol)

    def test_compute_weights_pol_IQU(self, initint, initfloat, rtol):
        cpp_sin2phi = np.zeros(self.nsamples, dtype=initfloat.dtype)
        cpp_cos2phi = np.zeros(self.nsamples, dtype=initfloat.dtype)

        cpp_weighted_counts = np.zeros(self.npix, dtype=initfloat.dtype)
        cpp_weighted_sin_sq = np.zeros(self.npix, dtype=initfloat.dtype)
        cpp_weighted_cos_sq = np.zeros(self.npix, dtype=initfloat.dtype)
        cpp_weighted_sincos = np.zeros(self.npix, dtype=initfloat.dtype)
        cpp_weighted_sin = np.zeros(self.npix, dtype=initfloat.dtype)
        cpp_weighted_cos = np.zeros(self.npix, dtype=initfloat.dtype)

        compute_weights.compute_weights_pol_IQU(
            self.nsamples,
            initint.pointings,
            self.pointings_flag,
            initfloat.noise_weights,
            initfloat.pol_angles,
            cpp_weighted_counts,
            cpp_sin2phi,
            cpp_cos2phi,
            cpp_weighted_sin_sq,
            cpp_weighted_cos_sq,
            cpp_weighted_sincos,
            cpp_weighted_sin,
            cpp_weighted_cos,
        )

        (
            py_weighted_counts,
            py_sin2phi,
            py_cos2phi,
            py_weighted_sin_sq,
            py_weighted_cos_sq,
            py_weighted_sincos,
            py_weighted_sin,
            py_weighted_cos,
        ) = cw.computeweights_pol_IQU(
            self.npix,
            self.nsamples,
            initint.pointings,
            self.pointings_flag,
            initfloat.noise_weights,
            initfloat.pol_angles,
            dtype_float=initfloat.dtype,
        )

        np.testing.assert_allclose(cpp_weighted_counts, py_weighted_counts, rtol=rtol)
        np.testing.assert_allclose(cpp_sin2phi, py_sin2phi, rtol=rtol)
        np.testing.assert_allclose(cpp_cos2phi, py_cos2phi, rtol=rtol)
        np.testing.assert_allclose(cpp_weighted_sin_sq, py_weighted_sin_sq, rtol=rtol)
        np.testing.assert_allclose(cpp_weighted_cos_sq, py_weighted_cos_sq, rtol=rtol)
        np.testing.assert_allclose(cpp_weighted_sincos, py_weighted_sincos, rtol=rtol)
        np.testing.assert_allclose(cpp_weighted_sin, py_weighted_sin, rtol=rtol)
        np.testing.assert_allclose(cpp_weighted_cos, py_weighted_cos, rtol=rtol)

    def test_get_pix_mask_pol_QU(self, initint, initfloat, rtol):
        (
            weighted_counts,
            __,
            __,
            weighted_sin_sq,
            weighted_cos_sq,
            weighted_sincos,
        ) = cw.computeweights_pol_QU(
            self.npix,
            self.nsamples,
            initint.pointings,
            self.pointings_flag,
            initfloat.noise_weights,
            initfloat.pol_angles,
            dtype_float=initfloat.dtype,
        )

        cpp_pixel_mask = np.zeros(self.npix, initint.dtype)
        cpp_old2new_pixel = np.zeros(self.npix, dtype=initint.dtype)
        cpp_pixel_flag = np.zeros(self.npix, dtype=bool)

        cpp_new_npix = compute_weights.get_pixel_mask_pol(
            2,
            self.npix,
            1.0e3,
            weighted_counts,
            weighted_sin_sq,
            weighted_cos_sq,
            weighted_sincos,
            cpp_pixel_mask,
            cpp_old2new_pixel,
            cpp_pixel_flag,
        )

        cpp_pixel_mask.resize(cpp_new_npix, refcheck=False)

        (
            py_new_npix,
            py_pixel_mask,
            py_old2new_pixel,
            py_pixel_flag,
        ) = cw.get_pix_mask_pol(
            self.npix,
            2,
            1.0e3,
            weighted_counts,
            weighted_sin_sq,
            weighted_cos_sq,
            weighted_sincos,
            dtype_int=initint.dtype,
        )

        np.testing.assert_equal(cpp_new_npix, py_new_npix)
        np.testing.assert_array_equal(cpp_pixel_mask, py_pixel_mask)
        np.testing.assert_array_equal(cpp_old2new_pixel, py_old2new_pixel)
        np.testing.assert_array_equal(cpp_pixel_flag, py_pixel_flag)

    def test_get_pix_mask_pol_IQU(self, initint, initfloat, rtol):
        (
            weighted_counts,
            __,
            __,
            weighted_sin_sq,
            weighted_cos_sq,
            weighted_sincos,
            __,
            __,
        ) = cw.computeweights_pol_IQU(
            self.npix,
            self.nsamples,
            initint.pointings,
            self.pointings_flag,
            initfloat.noise_weights,
            initfloat.pol_angles,
            dtype_float=initfloat.dtype,
        )

        cpp_pixel_mask = np.zeros(self.npix, initint.dtype)
        cpp_old2new_pixel = np.zeros(self.npix, dtype=initint.dtype)
        cpp_pixel_flag = np.zeros(self.npix, dtype=bool)

        cpp_new_npix = compute_weights.get_pixel_mask_pol(
            3,
            self.npix,
            1.0e3,
            weighted_counts,
            weighted_sin_sq,
            weighted_cos_sq,
            weighted_sincos,
            cpp_pixel_mask,
            cpp_old2new_pixel,
            cpp_pixel_flag,
        )

        cpp_pixel_mask.resize(cpp_new_npix, refcheck=False)

        (
            py_new_npix,
            py_pixel_mask,
            py_old2new_pixel,
            py_pixel_flag,
        ) = cw.get_pix_mask_pol(
            self.npix,
            3,
            1.0e3,
            weighted_counts,
            weighted_sin_sq,
            weighted_cos_sq,
            weighted_sincos,
            dtype_int=initint.dtype,
        )

        np.testing.assert_equal(cpp_new_npix, py_new_npix)
        np.testing.assert_array_equal(cpp_pixel_mask, py_pixel_mask)
        np.testing.assert_array_equal(cpp_old2new_pixel, py_old2new_pixel)
        np.testing.assert_array_equal(cpp_pixel_flag, py_pixel_flag)


if __name__ == "__main__":
    pytest.main(
        [f"{__file__}::TestComputeWeights::test_compute_weights_pol_I", "-v", "-s"]
    )
    pytest.main(
        [f"{__file__}::TestComputeWeights::test_compute_weights_pol_QU", "-v", "-s"]
    )
    pytest.main(
        [f"{__file__}::TestComputeWeights::test_compute_weights_pol_IQU", "-v", "-s"]
    )
    pytest.main(
        [f"{__file__}::TestComputeWeights::test_get_pix_mask_pol_QU", "-v", "-s"]
    )
    pytest.main(
        [f"{__file__}::TestComputeWeights::test_get_pix_mask_pol_IQU", "-v", "-s"]
    )