import pytest
import numpy as np
import BlkDiagPrecondLO_tools

import helper_ProcessTimeSamples as hpts
import helper_BlkDiagPrecondLO_tools as bdplo_tools


class InitCommonParams:
    np.random.seed(987)
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
class TestBlkDiagPrecondLOToolsCpp(InitCommonParams):
    def test_I_Cpp(self, initint, initfloat, rtol):
        solver_type = hpts.SolverType.I

        PTS = hpts.ProcessTimeSamples(
            npix=self.npix,
            pointings=initint.pointings,
            pointings_flag=self.pointings_flag,
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

        np.testing.assert_allclose(cpp_prod, py_prod, rtol=rtol)

    def test_QU_Cpp(self, initint, initfloat, rtol):
        solver_type = hpts.SolverType.QU

        PTS = hpts.ProcessTimeSamples(
            npix=self.npix,
            pointings=initint.pointings,
            pointings_flag=self.pointings_flag,
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

        np.testing.assert_allclose(cpp_prod, py_prod, rtol=rtol)

    def test_IQU_Cpp(self, initint, initfloat, rtol):
        solver_type = hpts.SolverType.IQU

        PTS = hpts.ProcessTimeSamples(
            npix=self.npix,
            pointings=initint.pointings,
            pointings_flag=self.pointings_flag,
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

        np.testing.assert_allclose(cpp_prod, py_prod, rtol=rtol)


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
