import pytest
import numpy as np
import BlkDiagPrecondLO_tools

import helper_initweigths as iw
import helper_repixelize as rp
import helper_BlkDiagPrecondLO as bdpclo


class InitCommonParams:
    np.random.seed(123454321)
    oldnpix = 128
    nsamples = oldnpix * 10
    pixs = np.random.randint(low=0, high=oldnpix, size=nsamples)
    badpixs = np.random.randint(low=0, high=oldnpix, size=10)
    pixs[badpixs] = -1
    obspix = np.arange(nsamples)
    threshold = 1.0e3


class InitFloat32Params(InitCommonParams):
    def __init__(self):
        super().__init__()

        self.phi = np.random.uniform(
            low=-np.pi / 2.0, high=np.pi / 2.0, size=self.nsamples
        ).astype(dtype=np.float32)
        self.w = np.random.random(size=self.nsamples).astype(dtype=np.float32)

        (
            self.counts,
            self.sine,
            self.cosine,
            __,
            __,
            self.sin2,
            self.cos2,
            self.sincos,
        ) = iw.initializeweights_pol3(
            self.nsamples, self.oldnpix, self.w, self.pixs, self.phi
        )

        mask_pol2, mask_pol3 = iw.get_mask_pol(
            self.counts, self.sin2, self.cos2, self.sincos, self.threshold
        )

        (
            self.new_npix_2,
            __,
            __,
            self.counts_2,
            __,
            self.sin2_2,
            self.cos2_2,
            self.sincos_2,
        ) = rp.repixelization_pol2(
            self.oldnpix,
            mask_pol2,
            self.counts.copy(),
            self.obspix.copy(),
            self.sin2.copy(),
            self.cos2.copy(),
            self.sincos.copy(),
        )

        (
            self.new_npix_3,
            __,
            __,
            self.counts_3,
            __,
            self.sin2_3,
            self.cos2_3,
            self.sincos_3,
            self.sine_3,
            self.cosine_3,
        ) = rp.repixelization_pol3(
            self.oldnpix,
            mask_pol3,
            self.counts.copy(),
            self.obspix.copy(),
            self.sin2.copy(),
            self.cos2.copy(),
            self.sincos.copy(),
            self.sine.copy(),
            self.cosine.copy(),
        )

        self.vec = np.random.random(size=self.oldnpix * 3).astype(dtype=np.float32)


class InitFloat64Params(InitCommonParams):
    def __init__(self):
        super().__init__()

        self.phi = np.random.uniform(
            low=-np.pi / 2.0, high=np.pi / 2.0, size=self.nsamples
        ).astype(dtype=np.float64)
        self.w = np.random.random(size=self.nsamples).astype(dtype=np.float64)

        (
            self.counts,
            self.sine,
            self.cosine,
            __,
            __,
            self.sin2,
            self.cos2,
            self.sincos,
        ) = iw.initializeweights_pol3(
            self.nsamples, self.oldnpix, self.w, self.pixs, self.phi
        )

        mask_pol2, mask_pol3 = iw.get_mask_pol(
            self.counts, self.sin2, self.cos2, self.sincos, self.threshold
        )

        (
            self.new_npix_2,
            __,
            __,
            self.counts_2,
            __,
            self.sin2_2,
            self.cos2_2,
            self.sincos_2,
        ) = rp.repixelization_pol2(
            self.oldnpix,
            mask_pol2,
            self.counts.copy(),
            self.obspix.copy(),
            self.sin2.copy(),
            self.cos2.copy(),
            self.sincos.copy(),
        )

        (
            self.new_npix_3,
            __,
            __,
            self.counts_3,
            __,
            self.sin2_3,
            self.cos2_3,
            self.sincos_3,
            self.sine_3,
            self.cosine_3,
        ) = rp.repixelization_pol3(
            self.oldnpix,
            mask_pol3,
            self.counts.copy(),
            self.obspix.copy(),
            self.sin2.copy(),
            self.cos2.copy(),
            self.sincos.copy(),
            self.sine.copy(),
            self.cosine.copy(),
        )

        self.vec = np.random.random(size=self.oldnpix * 3).astype(dtype=np.float64)


@pytest.mark.parametrize(
    "initfloat, rtol", [(InitFloat32Params(), 5.0e-6), (InitFloat64Params(), 5.0e-6)]
)
class TestBlkDiagPrecondLO:
    def test_BlkDiagPrecondLO_mult_qu(self, initfloat, rtol):
        cpp_prod = BlkDiagPrecondLO_tools.py_BlkDiagPrecondLO_mult_qu(
            initfloat.new_npix_2,
            initfloat.sin2_2,
            initfloat.cos2_2,
            initfloat.sincos_2,
            initfloat.vec,
        )

        py_prod = bdpclo.BlkDiagPrecondLO_mult_qu(
            initfloat.new_npix_2,
            initfloat.sin2_2,
            initfloat.cos2_2,
            initfloat.sincos_2,
            initfloat.vec,
        )

        np.testing.assert_allclose(
            cpp_prod,
            py_prod,
            rtol=rtol,
        )

    def test_BlkDiagPrecondLO_mult_iqu(self, initfloat, rtol):
        cpp_prod = BlkDiagPrecondLO_tools.py_BlkDiagPrecondLO_mult_iqu(
            initfloat.new_npix_3,
            initfloat.counts_3,
            initfloat.sine_3,
            initfloat.cosine_3,
            initfloat.sin2_3,
            initfloat.cos2_3,
            initfloat.sincos_3,
            initfloat.vec,
        )

        py_prod = bdpclo.BlkDiagPrecondLO_mult_iqu(
            initfloat.new_npix_3,
            initfloat.counts_3,
            initfloat.sine_3,
            initfloat.cosine_3,
            initfloat.sin2_3,
            initfloat.cos2_3,
            initfloat.sincos_3,
            initfloat.vec,
        )

        np.testing.assert_allclose(cpp_prod, py_prod, rtol=rtol)


if __name__ == "__main__":
    pytest.main(
        [
            f"{__file__}::TestBlkDiagPrecondLO::test_BlkDiagPrecondLO_mult_qu",
            "-v",
            "-s",
        ]
    )
    pytest.main(
        [
            f"{__file__}::TestBlkDiagPrecondLO::test_BlkDiagPrecondLO_mult_iqu",
            "-v",
            "-s",
        ]
    )
