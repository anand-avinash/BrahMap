import pytest
import numpy as np
import repixelize

import helper_initweigths as iw
import helper_repixelize as rp


class TestRepixelization:
    np.random.seed(3434)
    oldnpix = 128
    nsamples = oldnpix * 10
    w = np.random.random(size=nsamples)
    pixs = np.random.randint(low=0, high=oldnpix, size=nsamples)
    badpixs = np.random.randint(low=0, high=nsamples, size=100)
    pixs[badpixs] = -1
    phi = np.random.uniform(low=-np.pi / 2.0, high=np.pi / 2.0, size=nsamples)
    obspix = np.arange(nsamples)
    threshold = 1.0e3

    def test_repixelization_pol1(self):
        counts, mask = iw.initializeweights_pol1(
            self.nsamples, self.oldnpix, self.w, self.pixs
        )

        (
            cpp_n_new_pix,
            cpp_n_removed_pix,
            cpp_old2new,
            cpp_counts,
            cpp_obspix,
        ) = repixelize.py_repixelization_pol1(
            self.oldnpix, mask, counts.copy(), self.obspix.copy()
        )

        (
            py_n_new_pix,
            py_n_removed_pix,
            py_old2new,
            py_counts,
            py_obspix,
        ) = rp.repixelization_pol1(self.oldnpix, mask, counts, self.obspix)

        assert cpp_n_new_pix == py_n_new_pix
        assert cpp_n_removed_pix == py_n_removed_pix
        np.testing.assert_array_equal(cpp_old2new, py_old2new)
        np.testing.assert_allclose(cpp_counts, py_counts)
        np.testing.assert_array_equal(cpp_obspix, py_obspix)

    def test_repixelization_pol2(self):
        counts, __, __, sin2, cos2, sincos = iw.initializeweights_pol2(
            self.nsamples, self.oldnpix, self.w, self.pixs, self.phi
        )

        mask_pol2, __ = iw.get_mask_pol(counts, sin2, cos2, sincos, self.threshold)

        (
            cpp_n_new_pix,
            cpp_n_removed_pix,
            cpp_old2new,
            cpp_counts,
            cpp_obspix,
            cpp_sin2,
            cpp_cos2,
            cpp_sincos,
        ) = repixelize.py_repixelization_pol2(
            self.oldnpix, mask_pol2, counts, self.obspix, sin2, cos2, sincos
        )

        (
            py_n_new_pix,
            py_n_removed_pix,
            py_old2new,
            py_counts,
            py_obspix,
            py_sin2,
            py_cos2,
            py_sincos,
        ) = rp.repixelization_pol2(
            self.oldnpix, mask_pol2, counts, self.obspix, sin2, cos2, sincos
        )

        assert cpp_n_new_pix == py_n_new_pix
        assert cpp_n_removed_pix == py_n_removed_pix
        np.testing.assert_array_equal(cpp_old2new, py_old2new)
        np.testing.assert_allclose(cpp_counts, py_counts)
        np.testing.assert_array_equal(cpp_obspix, py_obspix)
        np.testing.assert_allclose(cpp_sin2, py_sin2)
        np.testing.assert_allclose(cpp_cos2, py_cos2)
        np.testing.assert_allclose(cpp_sincos, py_sincos)

    def test_repixelization_pol3(self):
        counts, sine, cosine, __, __, sin2, cos2, sincos = iw.initializeweights_pol3(
            self.nsamples, self.oldnpix, self.w, self.pixs, self.phi
        )

        __, mask_pol3 = iw.get_mask_pol(counts, sin2, cos2, sincos, self.threshold)

        (
            cpp_n_new_pix,
            cpp_n_removed_pix,
            cpp_old2new,
            cpp_counts,
            cpp_obspix,
            cpp_sin2,
            cpp_cos2,
            cpp_sincos,
            cpp_sine,
            cpp_cosine,
        ) = repixelize.py_repixelization_pol3(
            self.oldnpix,
            mask_pol3,
            counts,
            self.obspix,
            sin2,
            cos2,
            sincos,
            sine,
            cosine,
        )

        (
            py_n_new_pix,
            py_n_removed_pix,
            py_old2new,
            py_counts,
            py_obspix,
            py_sin2,
            py_cos2,
            py_sincos,
            py_sine,
            py_cosine,
        ) = rp.repixelization_pol3(
            self.oldnpix,
            mask_pol3,
            counts,
            self.obspix,
            sin2,
            cos2,
            sincos,
            sine,
            cosine,
        )

        assert cpp_n_new_pix == py_n_new_pix
        assert cpp_n_removed_pix == py_n_removed_pix
        np.testing.assert_array_equal(cpp_old2new, py_old2new)
        np.testing.assert_allclose(cpp_counts, py_counts)
        np.testing.assert_array_equal(cpp_obspix, py_obspix)
        np.testing.assert_allclose(cpp_sin2, py_sin2)
        np.testing.assert_allclose(cpp_cos2, py_cos2)
        np.testing.assert_allclose(cpp_sincos, py_sincos)
        np.testing.assert_allclose(cpp_sine, py_sine)
        np.testing.assert_allclose(cpp_cosine, py_cosine)


if __name__ == "__main__":
    pytest.main(
        [f"{__file__}::TestRepixelization::test_repixelization_pol1", "-v", "-s"]
    )
    pytest.main(
        [f"{__file__}::TestRepixelization::test_repixelization_pol2", "-v", "-s"]
    )
    pytest.main(
        [f"{__file__}::TestRepixelization::test_repixelization_pol3", "-v", "-s"]
    )
