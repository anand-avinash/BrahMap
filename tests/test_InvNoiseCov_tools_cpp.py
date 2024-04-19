import pytest
import numpy as np
import InvNoiseCov_tools

from brahmap.interfaces import InvNoiseCovLO_Uncorrelated


class InitCommonParams:
    np.random.seed(12343)
    nsamples = 1280


class InitFloat32Params(InitCommonParams):
    def __init__(self) -> None:
        super().__init__()

        self.dtype = np.float32
        self.diag = np.random.random(size=self.nsamples).astype(dtype=self.dtype)
        self.vec = np.random.random(size=self.nsamples).astype(dtype=self.dtype)


class InitFloat64Params(InitCommonParams):
    def __init__(self) -> None:
        super().__init__()

        self.dtype = np.float64
        self.diag = np.random.random(size=self.nsamples).astype(dtype=self.dtype)
        self.vec = np.random.random(size=self.nsamples).astype(dtype=self.dtype)


@pytest.mark.parametrize(
    "initfloat, rtol",
    [
        (InitFloat32Params(), 1.5e-4),
        (InitFloat64Params(), 1.5e-5),
    ],
)
class TestInvNoiseCov_tools(InitCommonParams):
    def test_mult(self, initfloat, rtol):
        cpp_prod = np.zeros(self.nsamples, dtype=initfloat.dtype)

        InvNoiseCov_tools.uncorrelated_mult(
            nsamples=self.nsamples,
            diag=initfloat.diag,
            vec=initfloat.vec,
            prod=cpp_prod,
        )

        py_prod = initfloat.diag * initfloat.vec

        np.testing.assert_allclose(cpp_prod, py_prod, rtol=rtol)


@pytest.mark.parametrize(
    "initfloat, rtol",
    [
        (InitFloat32Params(), 1.5e-4),
        (InitFloat64Params(), 1.5e-5),
    ],
)
class TestInvNoiseCovLO_Uncorrelated(InitCommonParams):
    def test_InvNoiseCovLO_Uncorrelated(self, initfloat, rtol):
        test_lo = InvNoiseCovLO_Uncorrelated(diag=initfloat.diag, dtype=initfloat.dtype)

        cpp_prod = test_lo * initfloat.vec
        py_prod = initfloat.diag * initfloat.vec

        np.testing.assert_allclose(cpp_prod, py_prod, rtol=rtol)


if __name__ == "__main__":
    pytest.main([f"{__file__}::TestInvNoiseCov_tools::test_mult", "-v", "-s"])
    pytest.main(
        [
            f"{__file__}::TestInvNoiseCovLO_Uncorrelated::test_InvNoiseCovLO_Uncorrelated",
            "-v",
            "-s",
        ]
    )
