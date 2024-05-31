import pytest
import numpy as np

import brahmap
from brahmap._extensions import PointingLO_tools

import helper_ProcessTimeSamples as hpts
import helper_PointingLO_tools as hplo_tools


class InitCommonParams:
    np.random.seed(54321)
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

        self.vec = np.random.random(size=self.npix * 3).astype(dtype=self.dtype)
        self.rvec = np.random.random(size=self.nsamples).astype(dtype=self.dtype)


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

        self.vec = np.random.random(size=self.npix * 3).astype(dtype=self.dtype)
        self.rvec = np.random.random(size=self.nsamples).astype(dtype=self.dtype)


@pytest.mark.parametrize(
    "initint, initfloat, rtol",
    [
        (InitInt32Params(), InitFloat32Params(), 1.5e-4),
        (InitInt64Params(), InitFloat32Params(), 1.5e-4),
        (InitInt32Params(), InitFloat64Params(), 1.5e-5),
        (InitInt64Params(), InitFloat64Params(), 1.5e-5),
    ],
)
class TestPointingLOTools_I(InitCommonParams):
    def test_I(self, initint, initfloat, rtol):
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

        nrows = self.nsamples
        ncols = PTS.new_npix * PTS.solver_type

        cpp_mult_prod = np.zeros(nrows, dtype=initfloat.dtype)
        vec = np.resize(initfloat.vec, ncols)

        PointingLO_tools.PLO_mult_I(
            nrows,
            PTS.pointings,
            PTS.pointings_flag,
            vec,
            cpp_mult_prod,
        )
        py_mult_prod = hplo_tools.PLO_mult_I(
            nrows,
            PTS.pointings,
            PTS.pointings_flag,
            vec,
        )

        cpp_rmult_prod = np.zeros(ncols, dtype=initfloat.dtype)
        rvec = initfloat.rvec

        PointingLO_tools.PLO_rmult_I(
            PTS.new_npix,
            nrows,
            PTS.pointings,
            PTS.pointings_flag,
            rvec,
            cpp_rmult_prod,
            brahmap.bMPI.comm,
        )
        py_rmult_prod = hplo_tools.PLO_rmult_I(
            nrows,
            ncols,
            PTS.pointings,
            PTS.pointings_flag,
            rvec,
            brahmap.bMPI.comm,
        )

        np.testing.assert_allclose(cpp_mult_prod, py_mult_prod, rtol=rtol)
        np.testing.assert_allclose(cpp_rmult_prod, py_rmult_prod, rtol=rtol)


@pytest.mark.parametrize(
    "initint, initfloat, rtol",
    [
        (InitInt32Params(), InitFloat32Params(), 1.5e-4),
        (InitInt64Params(), InitFloat32Params(), 1.5e-4),
        (InitInt32Params(), InitFloat64Params(), 1.5e-5),
        (InitInt64Params(), InitFloat64Params(), 1.5e-5),
    ],
)
class TestPointingLOTools_QU(InitCommonParams):
    def test_QU(self, initint, initfloat, rtol):
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

        nrows = self.nsamples
        ncols = PTS.new_npix * PTS.solver_type

        cpp_mult_prod = np.zeros(nrows, dtype=initfloat.dtype)
        vec = np.resize(initfloat.vec, ncols)

        PointingLO_tools.PLO_mult_QU(
            nrows,
            PTS.pointings,
            PTS.pointings_flag,
            PTS.sin2phi,
            PTS.cos2phi,
            vec,
            cpp_mult_prod,
        )
        py_mult_prod = hplo_tools.PLO_mult_QU(
            nrows,
            PTS.pointings,
            PTS.pointings_flag,
            PTS.sin2phi,
            PTS.cos2phi,
            vec,
        )

        cpp_rmult_prod = np.zeros(ncols, dtype=initfloat.dtype)
        rvec = initfloat.rvec

        PointingLO_tools.PLO_rmult_QU(
            PTS.new_npix,
            nrows,
            PTS.pointings,
            PTS.pointings_flag,
            PTS.sin2phi,
            PTS.cos2phi,
            rvec,
            cpp_rmult_prod,
            brahmap.bMPI.comm,
        )
        py_rmult_prod = hplo_tools.PLO_rmult_QU(
            nrows,
            ncols,
            PTS.pointings,
            PTS.pointings_flag,
            PTS.sin2phi,
            PTS.cos2phi,
            rvec,
            brahmap.bMPI.comm,
        )

        np.testing.assert_allclose(cpp_mult_prod, py_mult_prod, rtol=rtol)
        np.testing.assert_allclose(cpp_rmult_prod, py_rmult_prod, rtol=rtol)


@pytest.mark.parametrize(
    "initint, initfloat, rtol",
    [
        (InitInt32Params(), InitFloat32Params(), 1.5e-4),
        (InitInt64Params(), InitFloat32Params(), 1.5e-4),
        (InitInt32Params(), InitFloat64Params(), 1.5e-5),
        (InitInt64Params(), InitFloat64Params(), 1.5e-5),
    ],
)
class TestPointingLOTools_IQU(InitCommonParams):
    def test_IQU(self, initint, initfloat, rtol):
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

        nrows = self.nsamples
        ncols = PTS.new_npix * PTS.solver_type

        cpp_mult_prod = np.zeros(nrows, dtype=initfloat.dtype)
        vec = np.resize(initfloat.vec, ncols)

        PointingLO_tools.PLO_mult_QU(
            nrows,
            PTS.pointings,
            PTS.pointings_flag,
            PTS.sin2phi,
            PTS.cos2phi,
            vec,
            cpp_mult_prod,
        )
        py_mult_prod = hplo_tools.PLO_mult_QU(
            nrows,
            PTS.pointings,
            PTS.pointings_flag,
            PTS.sin2phi,
            PTS.cos2phi,
            vec,
        )

        cpp_rmult_prod = np.zeros(ncols, dtype=initfloat.dtype)
        rvec = initfloat.rvec

        PointingLO_tools.PLO_rmult_QU(
            PTS.new_npix,
            nrows,
            PTS.pointings,
            PTS.pointings_flag,
            PTS.sin2phi,
            PTS.cos2phi,
            rvec,
            cpp_rmult_prod,
            brahmap.bMPI.comm,
        )
        py_rmult_prod = hplo_tools.PLO_rmult_QU(
            nrows,
            ncols,
            PTS.pointings,
            PTS.pointings_flag,
            PTS.sin2phi,
            PTS.cos2phi,
            rvec,
            brahmap.bMPI.comm,
        )

        np.testing.assert_allclose(cpp_mult_prod, py_mult_prod, rtol=rtol)
        np.testing.assert_allclose(cpp_rmult_prod, py_rmult_prod, rtol=rtol)


if __name__ == "__main__":
    pytest.main([f"{__file__}::TestPointingLOTools_I::test_I", "-v", "-s"])
    pytest.main([f"{__file__}::TestPointingLOTools_QU::test_QU", "-v", "-s"])
    pytest.main([f"{__file__}::TestPointingLOTools_IQU::test_IQU", "-v", "-s"])
