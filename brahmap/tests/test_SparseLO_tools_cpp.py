import pytest
import numpy as np
import SparseLO_tools

import helper_SparseLO as hs


class InitCommonParams:
    np.random.seed(123454321)
    oldnpix = 128
    nsamples = oldnpix * 10
    pixs = np.random.randint(low=0, high=oldnpix, size=nsamples)
    badpixs = np.random.randint(low=0, high=oldnpix, size=10)
    pixs[badpixs] = -1
    nrows = nsamples
    ncols = oldnpix


class InitFloat32Params(InitCommonParams):
    def __init__(self):
        super().__init__()

        self.phi = np.random.uniform(
            low=-np.pi / 2.0, high=np.pi / 2.0, size=self.nsamples
        ).astype(dtype=np.float32)
        self.sin = np.sin(2.0 * self.phi)
        self.cos = np.cos(2.0 * self.phi)
        self.vec = np.random.random(size=self.nsamples).astype(dtype=np.float32)


class InitFloat64Params(InitCommonParams):
    def __init__(self):
        super().__init__()

        self.phi = np.random.uniform(
            low=-np.pi / 2.0, high=np.pi / 2.0, size=self.nsamples
        ).astype(dtype=np.float64)
        self.sin = np.sin(2.0 * self.phi)
        self.cos = np.cos(2.0 * self.phi)
        self.vec = np.random.random(size=self.nsamples).astype(dtype=np.float64)


@pytest.mark.parametrize(
    "initfloat, rtol", [(InitFloat32Params(), 1.5e-5), (InitFloat64Params(), 1.5e-5)]
)
class TestSparseLO_mult_I(InitCommonParams):
    def test_mult(self, initfloat, rtol):
        cpp_prod = SparseLO_tools.py_SparseLO_mult(self.nrows, self.pixs, initfloat.vec)
        py_prod = hs.SparseLO_mult(self.nrows, self.pixs, initfloat.vec)
        np.testing.assert_allclose(cpp_prod, py_prod, rtol=rtol)

    def test_rmult(self, initfloat, rtol):
        cpp_prod = SparseLO_tools.py_SparseLO_rmult(
            self.nrows, self.ncols, self.pixs, initfloat.vec
        )
        py_prod = hs.SparseLO_rmult(self.nrows, self.ncols, self.pixs, initfloat.vec)
        np.testing.assert_allclose(cpp_prod, py_prod, rtol=rtol)


@pytest.mark.parametrize(
    "initfloat, rtol", [(InitFloat32Params(), 5.0e-5), (InitFloat64Params(), 5.0e-5)]
)
class TestSparseLO_mult_QU(InitCommonParams):
    def test_mult_qu(self, initfloat, rtol):
        cpp_prod = SparseLO_tools.py_SparseLO_mult_qu(
            self.nrows, self.pixs, initfloat.sin, initfloat.cos, initfloat.vec
        )
        py_prod = hs.SparseLO_mult_qu(
            self.nrows, self.pixs, initfloat.sin, initfloat.cos, initfloat.vec
        )
        np.testing.assert_allclose(cpp_prod, py_prod, rtol=rtol)

    def test_rmult_qu(self, initfloat, rtol):
        cpp_prod = SparseLO_tools.py_SparseLO_rmult_qu(
            self.nrows,
            self.ncols,
            self.pixs,
            initfloat.sin,
            initfloat.cos,
            initfloat.vec,
        )
        py_prod = hs.SparseLO_rmult_qu(
            self.nrows,
            self.ncols,
            self.pixs,
            initfloat.sin,
            initfloat.cos,
            initfloat.vec,
        )
        np.testing.assert_allclose(cpp_prod, py_prod, rtol=rtol)


@pytest.mark.parametrize(
    "initfloat, rtol", [(InitFloat32Params(), 5.0e-5), (InitFloat64Params(), 5.0e-5)]
)
class TestSparseLO_mult_IQU(InitCommonParams):
    def test_mult_iqu(self, initfloat, rtol):
        cpp_prod = SparseLO_tools.py_SparseLO_mult_iqu(
            self.nrows, self.pixs, initfloat.sin, initfloat.cos, initfloat.vec
        )
        py_prod = hs.SparseLO_mult_iqu(
            self.nrows, self.pixs, initfloat.sin, initfloat.cos, initfloat.vec
        )
        np.testing.assert_allclose(cpp_prod, py_prod, rtol=rtol)

    def test_rmult_iqu(self, initfloat, rtol):
        cpp_prod = SparseLO_tools.py_SparseLO_rmult_iqu(
            self.nrows,
            self.ncols,
            self.pixs,
            initfloat.sin,
            initfloat.cos,
            initfloat.vec,
        )
        py_prod = hs.SparseLO_rmult_iqu(
            self.nrows,
            self.ncols,
            self.pixs,
            initfloat.sin,
            initfloat.cos,
            initfloat.vec,
        )
        np.testing.assert_allclose(cpp_prod, py_prod, rtol=rtol)


if __name__ == "__main__":
    pytest.main([f"{__file__}::TestSparseLO_mult_I::test_mult", "-v", "-s"])
    pytest.main([f"{__file__}::TestSparseLO_mult_I::test_rmult", "-v", "-s"])
    pytest.main([f"{__file__}::TestSparseLO_mult_QU::test_mult_qu", "-v", "-s"])
    pytest.main([f"{__file__}::TestSparseLO_mult_QU::test_rmult_qu", "-v", "-s"])
    pytest.main([f"{__file__}::TestSparseLO_mult_IQU::test_mult_iqu", "-v", "-s"])
    pytest.main([f"{__file__}::TestSparseLO_mult_IQU::test_rmult_iqu", "-v", "-s"])
