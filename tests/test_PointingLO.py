import pytest
import numpy as np
import brahmap

import helper_PointingLO as hplo
import helper_ProcessTimeSamples as hpts


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
class TestPointingLO_I(InitCommonParams):
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

        P_cpp = brahmap.interfaces.PointingLO(PTS)
        P_py = hplo.PointingLO(PTS)

        ncols = PTS.new_npix * PTS.solver_type

        vec = np.resize(initfloat.vec, ncols)
        cpp_mult_prod = P_cpp * vec
        py_mult_prod = P_py * vec

        rvec = initfloat.rvec
        cpp_rmult_prod = P_cpp.T * rvec
        py_rmult_prod = P_py.T * rvec

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
class TestPointingLO_QU(InitCommonParams):
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

        P_cpp = brahmap.interfaces.PointingLO(PTS)
        P_py = hplo.PointingLO(PTS)

        ncols = PTS.new_npix * PTS.solver_type

        vec = np.resize(initfloat.vec, ncols)
        cpp_mult_prod = P_cpp * vec
        py_mult_prod = P_py * vec

        rvec = initfloat.rvec
        cpp_rmult_prod = P_cpp.T * rvec
        py_rmult_prod = P_py.T * rvec

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
class TestPointingLO_IQU(InitCommonParams):
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

        P_cpp = brahmap.interfaces.PointingLO(PTS)
        P_py = hplo.PointingLO(PTS)

        ncols = PTS.new_npix * PTS.solver_type

        vec = np.resize(initfloat.vec, ncols)
        cpp_mult_prod = P_cpp * vec
        py_mult_prod = P_py * vec

        rvec = initfloat.rvec
        cpp_rmult_prod = P_cpp.T * rvec
        py_rmult_prod = P_py.T * rvec

        np.testing.assert_allclose(cpp_mult_prod, py_mult_prod, rtol=rtol)
        np.testing.assert_allclose(cpp_rmult_prod, py_rmult_prod, rtol=rtol)


if __name__ == "__main__":
    pytest.main([f"{__file__}::TestPointingLO_I::test_I", "-v", "-s"])
    pytest.main([f"{__file__}::TestPointingLO_QU::test_QU", "-v", "-s"])
    pytest.main([f"{__file__}::TestPointingLO_IQU::test_IQU", "-v", "-s"])