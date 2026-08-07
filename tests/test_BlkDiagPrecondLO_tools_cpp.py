############################ TEST DESCRIPTION ############################
#
# Test defined here are related to the functions defined in the extension
# module `BlkDiagPrecondLO_tools`. All the tests defined here simply test if the
# computations defined the cpp functions produce the same results as their
# python analog.
#
# - class `TestBlkDiagPrecondLOToolsCpp`:
#
#   -   `test_I_Cpp`: tests the computations of
# `BlkDiagPrecondLO_tools.BDPLO_mult_I()`
#
# - Same as above, but for QU and IQU
#
###########################################################################

import pytest
import numpy as np

import brahmap
from brahmap._extensions import BlkDiagPrecondLO_tools

import py_BlkDiagPrecondLO_tools as bdplo_tools


TOLERANCES = {
    np.float32: {"rtol": 1.5e-4, "atol": 1.0e-5},
    np.float64: {"rtol": 1.5e-5, "atol": 1.0e-10},
}


class TestBlkDiagPrecondLOToolsCpp:
    def test_I_Cpp(self, setup_scan):
        initint, initfloat = setup_scan

        tol = TOLERANCES[initfloat.dtype]
        rtol, atol = tol["rtol"], tol["atol"]
        solver_type = brahmap.core.SolverType.I

        PTS = brahmap.core.ProcessTimeSamples(
            npix=initint.npix,
            pointings=initint.pointings,
            pointings_flag=initint.pointings_flag,
            solver_type=solver_type,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        vec = np.random.random(PTS.new_npix * PTS.solver_type).astype(
            dtype=initfloat.dtype, copy=False
        )

        cpp_prod = vec / PTS.weighted_counts

        py_prod = bdplo_tools.BDPLO_mult_I(
            PTS.weighted_counts,
            vec,
        )

        np.testing.assert_allclose(cpp_prod, py_prod, rtol=rtol, atol=atol)

    def test_QU_Cpp(self, setup_scan):
        initint, initfloat = setup_scan

        tol = TOLERANCES[initfloat.dtype]
        rtol, atol = tol["rtol"], tol["atol"]
        solver_type = brahmap.core.SolverType.QU

        PTS = brahmap.core.ProcessTimeSamples(
            npix=initint.npix,
            pointings=initint.pointings,
            pointings_flag=initint.pointings_flag,
            solver_type=solver_type,
            pol_angles=initfloat.pol_angles,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        vec = np.random.random(PTS.new_npix * PTS.solver_type).astype(
            dtype=initfloat.dtype, copy=False
        )

        cpp_prod = np.zeros(PTS.new_npix * PTS.solver_type, dtype=initfloat.dtype)
        BlkDiagPrecondLO_tools.BDPLO_mult_QU(
            PTS.new_npix,
            PTS.weighted_sin_sq,
            PTS.weighted_cos_sq,
            PTS.weighted_sincos,
            PTS.one_over_determinant,
            vec,
            cpp_prod,
        )

        py_prod = bdplo_tools.BDPLO_mult_QU(
            PTS.solver_type,
            PTS.new_npix,
            PTS.weighted_sin_sq,
            PTS.weighted_cos_sq,
            PTS.weighted_sincos,
            PTS.one_over_determinant,
            vec,
        )

        np.testing.assert_allclose(cpp_prod, py_prod, rtol=rtol, atol=atol)

    def test_IQU_Cpp(self, setup_scan):
        initint, initfloat = setup_scan

        tol = TOLERANCES[initfloat.dtype]
        rtol, atol = tol["rtol"], tol["atol"]
        solver_type = brahmap.core.SolverType.IQU

        PTS = brahmap.core.ProcessTimeSamples(
            npix=initint.npix,
            pointings=initint.pointings,
            pointings_flag=initint.pointings_flag,
            solver_type=solver_type,
            pol_angles=initfloat.pol_angles,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        vec = np.random.random(PTS.new_npix * PTS.solver_type).astype(
            dtype=initfloat.dtype, copy=False
        )

        cpp_prod = np.zeros(PTS.new_npix * PTS.solver_type, dtype=initfloat.dtype)
        BlkDiagPrecondLO_tools.BDPLO_mult_IQU(
            PTS.new_npix,
            PTS.weighted_counts,
            PTS.weighted_sin_sq,
            PTS.weighted_cos_sq,
            PTS.weighted_sincos,
            PTS.weighted_sin,
            PTS.weighted_cos,
            PTS.one_over_determinant,
            vec,
            cpp_prod,
        )

        py_prod = bdplo_tools.BDPLO_mult_IQU(
            PTS.solver_type,
            PTS.new_npix,
            PTS.weighted_counts,
            PTS.weighted_sin_sq,
            PTS.weighted_cos_sq,
            PTS.weighted_sincos,
            PTS.weighted_sin,
            PTS.weighted_cos,
            PTS.one_over_determinant,
            vec,
        )

        np.testing.assert_allclose(cpp_prod, py_prod, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main(
        [
            f"{__file__}::TestBlkDiagPrecondLOToolsCpp::test_I_Cpp",
            "-v",
            "-s",
        ]
    )
    pytest.main(
        [
            f"{__file__}::TestBlkDiagPrecondLOToolsCpp::test_QU_Cpp",
            "-v",
            "-s",
        ]
    )
    pytest.main(
        [
            f"{__file__}::TestBlkDiagPrecondLOToolsCpp::test_IQU_Cpp",
            "-v",
            "-s",
        ]
    )
