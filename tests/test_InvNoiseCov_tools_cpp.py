############################ TEST DESCRIPTION ############################
#
# Test defined here are related to the `InvNoiseCovLO_Uncorrelated` class of BrahMap.
#
# - class `TestInvNoiseCov_tools`:
#
#   -   `test_mult`: Here we are testing the computation of `mult()`
# routine defined in the extension module `InvNoiseCov_tools`

# - class `TestInvNoiseCovLO_Uncorrelated`:
#
#   -   `test_InvNoiseCovLO_Uncorrelated`: Here we are testing the
# `mult` method overload of `TestInvNoiseCovLO_Uncorrelated` against its
# explicit computation.
#
###########################################################################


import pytest
import numpy as np

from brahmap._extensions import InvNoiseCov_tools
from brahmap.interfaces import InvNoiseCovLO_Uncorrelated

import brahmap

brahmap.Initialize()


class InitCommonParams:
    np.random.seed(12343 + brahmap.bMPI.rank)
    nsamples_global = 1280

    div, rem = divmod(nsamples_global, brahmap.bMPI.size)
    nsamples = div + (brahmap.bMPI.rank < rem)


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


# Initializing the parameter classes
initfloat32 = InitFloat32Params()
initfloat64 = InitFloat64Params()


@pytest.mark.parametrize(
    "initfloat, rtol",
    [
        (initfloat32, 1.5e-4),
        (initfloat64, 1.5e-5),
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
        (initfloat32, 1.5e-4),
        (initfloat64, 1.5e-5),
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
