############################ TEST DESCRIPTION ############################
#
# Test defined here are related to the `PointingLO` class of BrahMap.
# Analogous to this class, in the test suite, we have defined another version
# of `PointingLO` based on only the python routines.
#
# - class `TestPointingLO_I_Cpp`:
#
#   -   `test_I_Cpp`: tests whether the `mult` and `rmult` method overloads
# of the the two versions of `PointingLO` produce the same results.
#
# - Same as above, but for QU and IQU
#
# - class `TestPointingLO_I`:
#
#   -   `test_I`: tests the `mult` and `rmult` method overloads of
# `brahmap.interfaces.PointingLO` against their explicit computations.
#
# - Same as above, but for QU and IQU
#
# Note: For I case `P.T * noise_vector` must be equal to the
# `weighted_counts` vector. For QU case, the resulting vector must have
# `weighted_cos` and `weighted_sin` at alternating position. And for IQU
# case, the resulting vector must have `weighted_counts`, `weighted_cos`
# and `weighted_sin` at alternating positions.
#
###########################################################################

import pytest
import numpy as np
import brahmap

import py_PointingLO as hplo

from mpi4py import MPI


class InitCommonParams:
    np.random.seed(54321 + brahmap.MPI_UTILS.rank)
    npix = 128
    nsamples_global = npix * 6

    div, rem = divmod(nsamples_global, brahmap.MPI_UTILS.size)
    nsamples = div + (brahmap.MPI_UTILS.rank < rem)

    nbad_pixels_global = npix
    div, rem = divmod(nbad_pixels_global, brahmap.MPI_UTILS.size)
    nbad_pixels = div + (brahmap.MPI_UTILS.rank < rem)

    pointings_flag = np.ones(nsamples, dtype=bool)
    bad_samples = np.random.randint(low=0, high=nsamples, size=nbad_pixels)
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


# Initializing the parameter classes
initint32 = InitInt32Params()
initint64 = InitInt64Params()
initfloat32 = InitFloat32Params()
initfloat64 = InitFloat64Params()


@pytest.mark.parametrize(
    "initint, initfloat, rtol, atol",
    [
        (initint32, initfloat32, 1.5e-4, 1.0e-5),
        (initint64, initfloat32, 1.5e-4, 1.0e-5),
        (initint32, initfloat64, 1.5e-5, 1.0e-10),
        (initint64, initfloat64, 1.5e-5, 1.0e-10),
    ],
)
class TestPointingLO_I_Cpp(InitCommonParams):
    def test_I_Cpp(self, initint, initfloat, rtol, atol):
        solver_type = brahmap.core.SolverType.I

        PTS = brahmap.core.ProcessTimeSamples(
            npix=self.npix,
            pointings=initint.pointings,
            pointings_flag=self.pointings_flag,
            solver_type=solver_type,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        P_cpp = brahmap.core.PointingLO(PTS)
        P_py = hplo.PointingLO(PTS)

        ncols = PTS.new_npix * PTS.solver_type

        vec = np.resize(initfloat.vec, ncols)
        cpp_mult_prod = P_cpp * vec
        py_mult_prod = P_py * vec

        rvec = initfloat.rvec
        cpp_rmult_prod = P_cpp.T * rvec
        py_rmult_prod = P_py.T * rvec

        np.testing.assert_allclose(
            cpp_mult_prod,
            py_mult_prod,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            cpp_rmult_prod,
            py_rmult_prod,
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize(
    "initint, initfloat, rtol, atol",
    [
        (initint32, initfloat32, 1.5e-4, 1.0e-5),
        (initint64, initfloat32, 1.5e-4, 1.0e-5),
        (initint32, initfloat64, 1.5e-5, 1.0e-10),
        (initint64, initfloat64, 1.5e-5, 1.0e-10),
    ],
)
class TestPointingLO_QU_Cpp(InitCommonParams):
    def test_QU_Cpp(self, initint, initfloat, rtol, atol):
        solver_type = brahmap.core.SolverType.QU

        PTS = brahmap.core.ProcessTimeSamples(
            npix=self.npix,
            pointings=initint.pointings,
            pointings_flag=self.pointings_flag,
            solver_type=solver_type,
            pol_angles=initfloat.pol_angles,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        P_cpp = brahmap.core.PointingLO(PTS)
        P_py = hplo.PointingLO(PTS)

        ncols = PTS.new_npix * PTS.solver_type

        vec = np.resize(initfloat.vec, ncols)
        cpp_mult_prod = P_cpp * vec
        py_mult_prod = P_py * vec

        rvec = initfloat.rvec
        cpp_rmult_prod = P_cpp.T * rvec
        py_rmult_prod = P_py.T * rvec

        np.testing.assert_allclose(
            cpp_mult_prod,
            py_mult_prod,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            cpp_rmult_prod,
            py_rmult_prod,
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize(
    "initint, initfloat, rtol, atol",
    [
        (initint32, initfloat32, 1.5e-4, 1.0e-5),
        (initint64, initfloat32, 1.5e-4, 1.0e-5),
        (initint32, initfloat64, 1.5e-5, 1.0e-10),
        (initint64, initfloat64, 1.5e-5, 1.0e-10),
    ],
)
class TestPointingLO_IQU_Cpp(InitCommonParams):
    def test_IQU_Cpp(self, initint, initfloat, rtol, atol):
        solver_type = brahmap.core.SolverType.IQU

        PTS = brahmap.core.ProcessTimeSamples(
            npix=self.npix,
            pointings=initint.pointings,
            pointings_flag=self.pointings_flag,
            solver_type=solver_type,
            pol_angles=initfloat.pol_angles,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        P_cpp = brahmap.core.PointingLO(PTS)
        P_py = hplo.PointingLO(PTS)

        ncols = PTS.new_npix * PTS.solver_type

        vec = np.resize(initfloat.vec, ncols)
        cpp_mult_prod = P_cpp * vec
        py_mult_prod = P_py * vec

        rvec = initfloat.rvec
        cpp_rmult_prod = P_cpp.T * rvec
        py_rmult_prod = P_py.T * rvec

        np.testing.assert_allclose(
            cpp_mult_prod,
            py_mult_prod,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            cpp_rmult_prod,
            py_rmult_prod,
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize(
    "initint, initfloat, rtol, atol",
    [
        (initint32, initfloat32, 1.5e-4, 1.0e-5),
        (initint64, initfloat32, 1.5e-4, 1.0e-5),
        (initint32, initfloat64, 1.5e-5, 1.0e-10),
        (initint64, initfloat64, 1.5e-5, 1.0e-10),
    ],
)
class TestPointingLO_I(InitCommonParams):
    def test_I(self, initint, initfloat, rtol, atol):
        solver_type = brahmap.core.SolverType.I

        PTS = brahmap.core.ProcessTimeSamples(
            npix=self.npix,
            pointings=initint.pointings,
            pointings_flag=self.pointings_flag,
            solver_type=solver_type,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        P = brahmap.core.PointingLO(PTS)

        # Test for P.T * <vector>
        weights = P.T * initfloat.noise_weights

        np.testing.assert_allclose(
            PTS.weighted_counts,
            weights,
            rtol=rtol,
            atol=atol,
        )

        # Test for P * <vector>
        ncols = PTS.new_npix * PTS.solver_type
        vec = np.resize(initfloat.vec, ncols)
        signal = P * vec

        signal_test = np.zeros(self.nsamples, dtype=initfloat.dtype)

        for idx in range(self.nsamples):
            pixel = PTS.pointings[idx]
            if PTS.pointings_flag[idx]:
                signal_test[idx] += vec[pixel]

        np.testing.assert_allclose(
            signal,
            signal_test,
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize(
    "initint, initfloat, rtol, atol",
    [
        (initint32, initfloat32, 1.5e-4, 1.0e-5),
        (initint64, initfloat32, 1.5e-4, 1.0e-5),
        (initint32, initfloat64, 1.5e-5, 1.0e-10),
        (initint64, initfloat64, 1.5e-5, 1.0e-10),
    ],
)
class TestPointingLO_QU(InitCommonParams):
    def test_QU(self, initint, initfloat, rtol, atol):
        solver_type = brahmap.core.SolverType.QU

        PTS = brahmap.core.ProcessTimeSamples(
            npix=self.npix,
            pointings=initint.pointings,
            pointings_flag=self.pointings_flag,
            solver_type=solver_type,
            pol_angles=initfloat.pol_angles,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        P = brahmap.core.PointingLO(PTS)

        # Test for P.T * <vector>
        weights = P.T * initfloat.noise_weights

        weighted_sin = np.zeros(PTS.new_npix, dtype=initfloat.dtype)
        weighted_cos = np.zeros(PTS.new_npix, dtype=initfloat.dtype)

        for idx in range(self.nsamples):
            if PTS.pointings_flag[idx]:
                pixel = PTS.pointings[idx]
                weighted_sin[pixel] += PTS.sin2phi[idx] * initfloat.noise_weights[idx]
                weighted_cos[pixel] += PTS.cos2phi[idx] * initfloat.noise_weights[idx]

        brahmap.MPI_UTILS.comm.Allreduce(MPI.IN_PLACE, weighted_sin, MPI.SUM)
        brahmap.MPI_UTILS.comm.Allreduce(MPI.IN_PLACE, weighted_cos, MPI.SUM)

        np.testing.assert_allclose(
            weighted_sin,
            weights[1::2],
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            weighted_cos,
            weights[0::2],
            rtol=rtol,
            atol=atol,
        )

        # Test for P * <vector>
        ncols = PTS.new_npix * PTS.solver_type
        vec = np.resize(initfloat.vec, ncols)
        signal = P * vec

        signal_test = np.zeros(self.nsamples, dtype=initfloat.dtype)

        for idx in range(self.nsamples):
            pixel = PTS.pointings[idx]
            if PTS.pointings_flag[idx]:
                signal_test[idx] += (
                    vec[2 * pixel + 0] * PTS.cos2phi[idx]
                    + vec[2 * pixel + 1] * PTS.sin2phi[idx]
                )

        np.testing.assert_allclose(
            signal,
            signal_test,
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize(
    "initint, initfloat, rtol, atol",
    [
        (initint32, initfloat32, 1.5e-4, 1.0e-5),
        (initint64, initfloat32, 1.5e-4, 1.0e-5),
        (initint32, initfloat64, 1.5e-5, 1.0e-10),
        (initint64, initfloat64, 1.5e-5, 1.0e-10),
    ],
)
class TestPointingLO_IQU(InitCommonParams):
    def test_IQU(self, initint, initfloat, rtol, atol):
        solver_type = brahmap.core.SolverType.IQU

        PTS = brahmap.core.ProcessTimeSamples(
            npix=self.npix,
            pointings=initint.pointings,
            pointings_flag=self.pointings_flag,
            solver_type=solver_type,
            pol_angles=initfloat.pol_angles,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        P = brahmap.core.PointingLO(PTS)

        # Test for P.T * <vector>
        weights = P.T * initfloat.noise_weights

        np.testing.assert_allclose(
            PTS.weighted_counts,
            weights[0::3],
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            PTS.weighted_cos,
            weights[1::3],
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            PTS.weighted_sin,
            weights[2::3],
            rtol=rtol,
            atol=atol,
        )

        # Test for P * <vector>
        ncols = PTS.new_npix * PTS.solver_type
        vec = np.resize(initfloat.vec, ncols)
        signal = P * vec

        signal_test = np.zeros(self.nsamples, dtype=initfloat.dtype)

        for idx in range(self.nsamples):
            pixel = PTS.pointings[idx]
            if PTS.pointings_flag[idx]:
                signal_test[idx] += (
                    vec[3 * pixel + 0] * 1.0
                    + vec[3 * pixel + 1] * PTS.cos2phi[idx]
                    + vec[3 * pixel + 2] * PTS.sin2phi[idx]
                )

        np.testing.assert_allclose(
            signal,
            signal_test,
            rtol=rtol,
            atol=atol,
        )


if __name__ == "__main__":
    pytest.main([f"{__file__}::TestPointingLO_I_Cpp::test_I_Cpp", "-v", "-s"])
    pytest.main([f"{__file__}::TestPointingLO_QU_Cpp::test_QU_Cpp", "-v", "-s"])
    pytest.main([f"{__file__}::TestPointingLO_IQU_Cpp::test_IQU_Cpp", "-v", "-s"])
    pytest.main([f"{__file__}::TestPointingLO_I::test_I", "-v", "-s"])
    pytest.main([f"{__file__}::TestPointingLO_QU::test_QU", "-v", "-s"])
    pytest.main([f"{__file__}::TestPointingLO_IQU::test_IQU", "-v", "-s"])
