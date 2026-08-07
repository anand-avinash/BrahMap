############################ TEST DESCRIPTION ############################
#
# Test defined here are related to the extension module `repixelize`. All
# the tests defined here simply test if the computations defined in the cpp
# functions produce the same result as their python analog.
#
# - class `TestRepixelization`:
#
#   -   `test_repixelize_pol_{I,QU,IQU}`: tests the computations of
#       `repixelize.repixelize_pol_{I,QU,IQU}()` functions
#
# - class `TestFlagBadPixelSamples`:
#
#   -   `test_flag_bad_pixel_samples()`: tests the computations of
# `repixelize.flag_bad_pixel_samples()` function
#
###########################################################################

import pytest
import numpy as np

import brahmap
from brahmap._extensions import repixelize

import py_ComputeWeights as cw
import py_Repixelization as rp


TOLERANCES = {
    np.float32: {"rtol": 1.5e-4, "atol": 1.0e-5},
    np.float64: {"rtol": 1.5e-5, "atol": 1.0e-10},
}


class TestRepixelization:
    def test_repixelize_pol_I(self, setup_scan):
        initint, initfloat = setup_scan

        tol = TOLERANCES[initfloat.dtype]
        rtol, atol = tol["rtol"], tol["atol"]
        (
            new_npix,
            py_hit_counts,
            py_weighted_counts,
            observed_pixels,
            __,
            __,
        ) = cw.computeweights_pol_I(
            initint.npix,
            initint.nsamples,
            initint.pointings,
            initint.pointings_flag,
            initfloat.noise_weights,
            initfloat.dtype,
            comm=brahmap.MPI_UTILS.comm,
        )

        cpp_hit_counts = py_hit_counts.astype(dtype=initint.dtype)
        cpp_weighted_counts = py_weighted_counts.copy()

        repixelize.repixelize_pol_I(
            new_npix,
            observed_pixels,
            cpp_hit_counts,
            cpp_weighted_counts,
        )

        cpp_hit_counts.resize(new_npix, refcheck=False)
        cpp_weighted_counts.resize(new_npix, refcheck=False)

        py_hit_counts, py_weighted_counts = rp.repixelize_pol_I(
            new_npix,
            observed_pixels,
            py_hit_counts,
            py_weighted_counts,
        )

        np.testing.assert_array_equal(py_hit_counts, cpp_hit_counts)
        np.testing.assert_allclose(
            py_weighted_counts,
            cpp_weighted_counts,
            rtol=rtol,
            atol=atol,
        )

    def test_repixelize_pol_QU(self, setup_scan):
        initint, initfloat = setup_scan

        tol = TOLERANCES[initfloat.dtype]
        rtol, atol = tol["rtol"], tol["atol"]
        (
            py_hit_counts,
            py_weighted_counts,
            __,
            __,
            py_weighted_sin_sq,
            py_weighted_cos_sq,
            py_weighted_sincos,
            py_one_over_determinant,
        ) = cw.computeweights_pol_QU(
            initint.npix,
            initint.nsamples,
            initint.pointings,
            initint.pointings_flag,
            initfloat.noise_weights,
            initfloat.pol_angles,
            dtype_float=initfloat.dtype,
            comm=brahmap.MPI_UTILS.comm,
        )

        new_npix, observed_pixels, __, __ = cw.get_pix_mask_pol(
            initint.npix,
            2,
            1.0e-5,
            py_hit_counts,
            py_one_over_determinant,
            initint.pointings.dtype,
        )

        cpp_hit_counts = py_hit_counts.astype(dtype=initint.dtype)
        cpp_weighted_counts = py_weighted_counts.copy()
        cpp_weighted_sin_sq = py_weighted_sin_sq.copy()
        cpp_weighted_cos_sq = py_weighted_cos_sq.copy()
        cpp_weighted_sincos = py_weighted_sincos.copy()
        cpp_one_over_determinant = py_one_over_determinant.copy()

        repixelize.repixelize_pol_QU(
            new_npix,
            observed_pixels,
            cpp_hit_counts,
            cpp_weighted_counts,
            cpp_weighted_sin_sq,
            cpp_weighted_cos_sq,
            cpp_weighted_sincos,
            cpp_one_over_determinant,
        )

        cpp_hit_counts.resize(new_npix, refcheck=False)
        cpp_weighted_counts.resize(new_npix, refcheck=False)
        cpp_weighted_sin_sq.resize(new_npix, refcheck=False)
        cpp_weighted_cos_sq.resize(new_npix, refcheck=False)
        cpp_weighted_sincos.resize(new_npix, refcheck=False)
        cpp_one_over_determinant.resize(new_npix, refcheck=False)

        (
            py_hit_counts,
            py_weighted_counts,
            py_weighted_sin_sq,
            py_weighted_cos_sq,
            py_weighted_sincos,
            py_one_over_determinant,
        ) = rp.repixelize_pol_QU(
            new_npix,
            observed_pixels,
            py_hit_counts,
            py_weighted_counts,
            py_weighted_sin_sq,
            py_weighted_cos_sq,
            py_weighted_sincos,
            py_one_over_determinant,
        )

        np.testing.assert_array_equal(
            py_hit_counts,
            cpp_hit_counts,
        )
        np.testing.assert_allclose(
            py_weighted_counts,
            cpp_weighted_counts,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            py_weighted_sin_sq,
            cpp_weighted_sin_sq,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            py_weighted_cos_sq,
            cpp_weighted_cos_sq,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            py_weighted_sincos,
            cpp_weighted_sincos,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            cpp_one_over_determinant,
            py_one_over_determinant,
            rtol=rtol,
            atol=atol,
        )

    def test_repixelize_pol_IQU(self, setup_scan):
        initint, initfloat = setup_scan

        tol = TOLERANCES[initfloat.dtype]
        rtol, atol = tol["rtol"], tol["atol"]
        (
            py_hit_counts,
            py_weighted_counts,
            __,
            __,
            py_weighted_sin_sq,
            py_weighted_cos_sq,
            py_weighted_sincos,
            py_weighted_sin,
            py_weighted_cos,
            py_one_over_determinant,
        ) = cw.computeweights_pol_IQU(
            initint.npix,
            initint.nsamples,
            initint.pointings,
            initint.pointings_flag,
            initfloat.noise_weights,
            initfloat.pol_angles,
            dtype_float=initfloat.dtype,
            comm=brahmap.MPI_UTILS.comm,
        )

        new_npix, observed_pixels, __, __ = cw.get_pix_mask_pol(
            initint.npix,
            3,
            1.0e-5,
            py_hit_counts,
            py_one_over_determinant,
            initint.dtype,
        )

        cpp_hit_counts = py_hit_counts.astype(dtype=initint.dtype)
        cpp_weighted_counts = py_weighted_counts.copy()
        cpp_weighted_sin_sq = py_weighted_sin_sq.copy()
        cpp_weighted_cos_sq = py_weighted_cos_sq.copy()
        cpp_weighted_sincos = py_weighted_sincos.copy()
        cpp_weighted_sin = py_weighted_sin.copy()
        cpp_weighted_cos = py_weighted_cos.copy()
        cpp_one_over_determinant = py_one_over_determinant.copy()

        repixelize.repixelize_pol_IQU(
            new_npix,
            observed_pixels,
            cpp_hit_counts,
            cpp_weighted_counts,
            cpp_weighted_sin_sq,
            cpp_weighted_cos_sq,
            cpp_weighted_sincos,
            cpp_weighted_sin,
            cpp_weighted_cos,
            cpp_one_over_determinant,
        )

        cpp_hit_counts.resize(new_npix, refcheck=False)
        cpp_weighted_counts.resize(new_npix, refcheck=False)
        cpp_weighted_sin_sq.resize(new_npix, refcheck=False)
        cpp_weighted_cos_sq.resize(new_npix, refcheck=False)
        cpp_weighted_sincos.resize(new_npix, refcheck=False)
        cpp_weighted_sin.resize(new_npix, refcheck=False)
        cpp_weighted_cos.resize(new_npix, refcheck=False)
        cpp_one_over_determinant.resize(new_npix, refcheck=False)

        (
            py_hit_counts,
            py_weighted_counts,
            py_weighted_sin_sq,
            py_weighted_cos_sq,
            py_weighted_sincos,
            py_weighted_sin,
            py_weighted_cos,
            py_one_over_determinant,
        ) = rp.repixelize_pol_IQU(
            new_npix,
            observed_pixels,
            py_hit_counts,
            py_weighted_counts,
            py_weighted_sin_sq,
            py_weighted_cos_sq,
            py_weighted_sincos,
            py_weighted_sin,
            py_weighted_cos,
            py_one_over_determinant,
        )

        np.testing.assert_array_equal(
            py_hit_counts,
            cpp_hit_counts,
        )
        np.testing.assert_allclose(
            py_weighted_counts,
            cpp_weighted_counts,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            py_weighted_sin_sq,
            cpp_weighted_sin_sq,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            py_weighted_cos_sq,
            cpp_weighted_cos_sq,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            py_weighted_sincos,
            cpp_weighted_sincos,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            py_weighted_sin,
            cpp_weighted_sin,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            py_weighted_cos,
            cpp_weighted_cos,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            cpp_one_over_determinant,
            py_one_over_determinant,
            rtol=rtol,
            atol=atol,
        )


class TestFlagBadPixelSamples:
    def test_flag_bad_pixel_samples(self, setup_scan):
        initint, initfloat = setup_scan

        (
            py_hit_counts,
            __,
            __,
            __,
            __,
            __,
            __,
            __,
            __,
            py_one_over_determinant,
        ) = cw.computeweights_pol_IQU(
            initint.npix,
            initint.nsamples,
            initint.pointings,
            initint.pointings_flag,
            initfloat.noise_weights,
            initfloat.pol_angles,
            dtype_float=initfloat.dtype,
            comm=brahmap.MPI_UTILS.comm,
        )

        __, __, old2new_pixel, pixel_flag = cw.get_pix_mask_pol(
            initint.npix,
            3,
            1.0e-5,
            py_hit_counts,
            py_one_over_determinant,
            initint.dtype,
        )

        py_pointings = initint.pointings.copy()
        cpp_pointings = initint.pointings.copy()

        py_pointings_flag = initint.pointings_flag.copy()
        cpp_pointings_flag = initint.pointings_flag.copy()

        rp.flag_bad_pixel_samples(
            initint.nsamples, pixel_flag, old2new_pixel, py_pointings, py_pointings_flag
        )

        repixelize.flag_bad_pixel_samples(
            initint.nsamples,
            pixel_flag,
            old2new_pixel,
            cpp_pointings,
            cpp_pointings_flag,
        )

        np.testing.assert_array_equal(cpp_pointings, py_pointings)
        np.testing.assert_array_equal(cpp_pointings_flag, py_pointings_flag)


if __name__ == "__main__":
    pytest.main([f"{__file__}::TestRepixelization::test_repixelize_pol_I", "-v", "-s"])
    pytest.main([f"{__file__}::TestRepixelization::test_repixelize_pol_QU", "-v", "-s"])
    pytest.main(
        [f"{__file__}::TestRepixelization::test_repixelize_pol_IQU", "-v", "-s"]
    )
    pytest.main(
        [
            f"{__file__}::TestFlagBadPixelSamples::test_flag_bad_pixel_samples",
            "-v",
            "-s",
        ]
    )
