############################ TEST DESCRIPTION ############################
#
# Test defined here are related to the functions defined in the extension
# module `compute_weights`
#
# - class `TestComputeWeights`: This test class implements the tests
#   for all the functions defined in the extension module `compute_weights`.
#   It simply tests if the computations defined in cpp functions produce
#   the same result as their python analog.
#
#   -   `test_compute_weights_pol_{I,QU,IQU}()`: test the computations of
#       `compute_weights.compute_weights_pol_{I,QU,IQU}` functions
#   -   `test_get_pix_mask_pol_{QU,IQU}`: test the computations of
#       `compute_weights.get_pixel_mask_pol()` function for both QU and IQU
#
###########################################################################

import pytest
import numpy as np

import brahmap
from brahmap._extensions import compute_weights

import py_ComputeWeights as cw

brahmap.Initialize()


class InitCommonParams:
    np.random.seed(1234 + brahmap.bMPI.rank)
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
class TestComputeWeights(InitCommonParams):
    def test_compute_weights_pol_I(self, initint, initfloat, rtol):
        cpp_weighted_counts = np.zeros(self.npix, dtype=initfloat.dtype)
        cpp_observed_pixels = np.zeros(self.npix, dtype=initint.dtype)
        cpp_old2new_pixel = np.zeros(self.npix, dtype=initint.dtype)
        cpp_pixel_flag = np.zeros(self.npix, dtype=bool)

        cpp_new_npix = compute_weights.compute_weights_pol_I(
            self.npix,
            self.nsamples,
            initint.pointings,
            self.pointings_flag,
            initfloat.noise_weights,
            cpp_weighted_counts,
            cpp_observed_pixels,
            cpp_old2new_pixel,
            cpp_pixel_flag,
            brahmap.bMPI.comm,
        )

        (
            py_new_npix,
            py_weighted_counts,
            py_observed_pixels,
            py_old2new_pixel,
            py_pixel_flag,
        ) = cw.computeweights_pol_I(
            self.npix,
            self.nsamples,
            initint.pointings,
            self.pointings_flag,
            initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            comm=brahmap.bMPI.comm,
        )

        cpp_observed_pixels.resize(cpp_new_npix, refcheck=False)

        np.testing.assert_equal(cpp_new_npix, py_new_npix)
        np.testing.assert_allclose(cpp_weighted_counts, py_weighted_counts, rtol=rtol)
        np.testing.assert_array_equal(cpp_observed_pixels, py_observed_pixels)
        np.testing.assert_array_equal(cpp_old2new_pixel, py_old2new_pixel)
        np.testing.assert_array_equal(cpp_pixel_flag, py_pixel_flag)

    def test_compute_weights_pol_QU(self, initint, initfloat, rtol):
        cpp_sin2phi = np.zeros(self.nsamples, dtype=initfloat.dtype)
        cpp_cos2phi = np.zeros(self.nsamples, dtype=initfloat.dtype)

        cpp_weighted_counts = np.zeros(self.npix, dtype=initfloat.dtype)
        cpp_weighted_sin_sq = np.zeros(self.npix, dtype=initfloat.dtype)
        cpp_weighted_cos_sq = np.zeros(self.npix, dtype=initfloat.dtype)
        cpp_weighted_sincos = np.zeros(self.npix, dtype=initfloat.dtype)
        cpp_one_over_determinant = np.zeros(self.npix, dtype=initfloat.dtype)

        compute_weights.compute_weights_pol_QU(
            self.npix,
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
            cpp_one_over_determinant,
            brahmap.bMPI.comm,
        )

        (
            py_weighted_counts,
            py_sin2phi,
            py_cos2phi,
            py_weighted_sin_sq,
            py_weighted_cos_sq,
            py_weighted_sincos,
            __,
        ) = cw.computeweights_pol_QU(
            self.npix,
            self.nsamples,
            initint.pointings,
            self.pointings_flag,
            initfloat.noise_weights,
            initfloat.pol_angles,
            dtype_float=initfloat.dtype,
            comm=brahmap.bMPI.comm,
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
        cpp_one_over_determinant = np.zeros(self.npix, dtype=initfloat.dtype)

        compute_weights.compute_weights_pol_IQU(
            self.npix,
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
            cpp_one_over_determinant,
            brahmap.bMPI.comm,
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
            __,
        ) = cw.computeweights_pol_IQU(
            self.npix,
            self.nsamples,
            initint.pointings,
            self.pointings_flag,
            initfloat.noise_weights,
            initfloat.pol_angles,
            dtype_float=initfloat.dtype,
            comm=brahmap.bMPI.comm,
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
            __,
            __,
            __,
            one_over_determinant,
        ) = cw.computeweights_pol_QU(
            self.npix,
            self.nsamples,
            initint.pointings,
            self.pointings_flag,
            initfloat.noise_weights,
            initfloat.pol_angles,
            dtype_float=initfloat.dtype,
            comm=brahmap.bMPI.comm,
        )

        cpp_observed_pixels = np.zeros(self.npix, initint.dtype)
        cpp_old2new_pixel = np.zeros(self.npix, dtype=initint.dtype)
        cpp_pixel_flag = np.zeros(self.npix, dtype=bool)

        cpp_new_npix = compute_weights.get_pixel_mask_pol(
            2,
            self.npix,
            1.0e3,
            weighted_counts,
            one_over_determinant,
            cpp_observed_pixels,
            cpp_old2new_pixel,
            cpp_pixel_flag,
        )

        cpp_observed_pixels.resize(cpp_new_npix, refcheck=False)

        (
            py_new_npix,
            py_observed_pixels,
            py_old2new_pixel,
            py_pixel_flag,
        ) = cw.get_pix_mask_pol(
            self.npix,
            2,
            1.0e3,
            weighted_counts,
            one_over_determinant,
            dtype_int=initint.dtype,
        )

        np.testing.assert_equal(cpp_new_npix, py_new_npix)
        np.testing.assert_array_equal(cpp_observed_pixels, py_observed_pixels)
        np.testing.assert_array_equal(cpp_old2new_pixel, py_old2new_pixel)
        np.testing.assert_array_equal(cpp_pixel_flag, py_pixel_flag)

    def test_get_pix_mask_pol_IQU(self, initint, initfloat, rtol):
        (
            weighted_counts,
            __,
            __,
            __,
            __,
            __,
            __,
            __,
            one_over_determinant,
        ) = cw.computeweights_pol_IQU(
            self.npix,
            self.nsamples,
            initint.pointings,
            self.pointings_flag,
            initfloat.noise_weights,
            initfloat.pol_angles,
            dtype_float=initfloat.dtype,
            comm=brahmap.bMPI.comm,
        )

        cpp_observed_pixels = np.zeros(self.npix, initint.dtype)
        cpp_old2new_pixel = np.zeros(self.npix, dtype=initint.dtype)
        cpp_pixel_flag = np.zeros(self.npix, dtype=bool)

        cpp_new_npix = compute_weights.get_pixel_mask_pol(
            3,
            self.npix,
            1.0e3,
            weighted_counts,
            one_over_determinant,
            cpp_observed_pixels,
            cpp_old2new_pixel,
            cpp_pixel_flag,
        )

        cpp_observed_pixels.resize(cpp_new_npix, refcheck=False)

        (
            py_new_npix,
            py_observed_pixels,
            py_old2new_pixel,
            py_pixel_flag,
        ) = cw.get_pix_mask_pol(
            self.npix,
            3,
            1.0e3,
            weighted_counts,
            one_over_determinant,
            dtype_int=initint.dtype,
        )

        np.testing.assert_equal(cpp_new_npix, py_new_npix)
        np.testing.assert_array_equal(cpp_observed_pixels, py_observed_pixels)
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
