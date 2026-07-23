############################ TEST DESCRIPTION ############################
#
# Test defined here are related to the following class of BrahMap.
#
# - class `TestLinAlg_tools`:
#
#   -   `test_mult`: Here we are testing the computation of `mult()`
# routine defined in the extension module `linalg_tools`
#
###########################################################################


import pytest
import numpy as np

from brahmap.math import linalg_tools


TOLERANCES = {
    np.float32: {"rtol": 1.5e-4, "atol": 1.0e-5},
    np.float64: {"rtol": 1.5e-5, "atol": 1.0e-10},
}


class TestLinAlg_tools:
    def test_mult(self, setup_linalg_tools):
        initfloat = setup_linalg_tools

        tol = TOLERANCES[initfloat.dtype]
        rtol, atol = tol["rtol"], tol["atol"]

        nsamples = initfloat.nsamples
        cpp_prod = np.zeros(nsamples, dtype=initfloat.dtype)

        linalg_tools.multiply_array(
            nsamples=nsamples,
            diag=initfloat.diag,
            vec=initfloat.vec,
            prod=cpp_prod,
        )

        py_prod = initfloat.diag * initfloat.vec

        tol = TOLERANCES[initfloat.dtype]
        np.testing.assert_allclose(cpp_prod, py_prod, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([f"{__file__}::TestLinAlg_tools::test_mult", "-v", "-s"])
