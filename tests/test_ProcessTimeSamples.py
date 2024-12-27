############################ TEST DESCRIPTION ############################
#
# Test defined here are related to the `ProcessTimeSamples` class of
# BrahMap. Analogous to this class, in the test suite, we have defined
# another version of `ProcessTimeSamples` based on only the python routines.
#
# - class `TestProcessTimeSamplesCpp`:
#
#   -   `test_ProcessTimeSamples_{I,QU,IQU}_Cpp`: Here we are testing if
# the class attributes of the two versions of `ProcessTimeSamples` matches
# with each other.

# - class `TestProcessTimeSamples`:
#
#   -   `test_ProcessTimeSamples_{I,QU,IQU}`: Here we are testing the
# class attributes of `brahmap.utilities.ProcessTimeSamples` against their
# explicit computations.
#
###########################################################################


import pytest
import numpy as np

import brahmap

import sincos_wrapper as sc
import py_ProcessTimeSamples as hpts

from mpi4py import MPI

brahmap.Initialize()


class InitCommonParams:
    np.random.seed(1234 + brahmap.bMPI.rank)
    npix = 128
    nsamples_global = npix * 6

    div, rem = divmod(nsamples_global, brahmap.bMPI.size)
    nsamples = div + (brahmap.bMPI.rank < rem)

    nbad_pixels_global = npix
    div, rem = divmod(nbad_pixels_global, brahmap.bMPI.size)
    nbad_pixels = div + (brahmap.bMPI.rank < rem)

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


# Initializing the parameter classes
initint32 = InitInt32Params()
initint64 = InitInt64Params()
initfloat32 = InitFloat32Params()
initfloat64 = InitFloat64Params()


@pytest.mark.parametrize(
    "initint, initfloat, rtol",
    [
        (initint32, initfloat32, 1.5e-3),
        (initint64, initfloat32, 1.5e-3),
        (initint32, initfloat64, 1.5e-5),
        (initint64, initfloat64, 1.5e-5),
    ],
)
class TestProcessTimeSamplesCpp(InitCommonParams):
    def test_ProcessTimeSamples_I_Cpp(self, initint, initfloat, rtol):
        solver_type = hpts.SolverType.I

        cpp_PTS = brahmap.utilities.ProcessTimeSamples(
            npix=self.npix,
            pointings=initint.pointings,
            pointings_flag=self.pointings_flag,
            solver_type=solver_type,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        py_PTS = hpts.ProcessTimeSamples(
            npix=self.npix,
            pointings=initint.pointings,
            pointings_flag=self.pointings_flag,
            solver_type=solver_type,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        np.testing.assert_array_equal(cpp_PTS.pointings, py_PTS.pointings)
        np.testing.assert_array_equal(cpp_PTS.pointings_flag, py_PTS.pointings_flag)
        np.testing.assert_equal(cpp_PTS.new_npix, py_PTS.new_npix)
        np.testing.assert_array_equal(cpp_PTS.observed_pixels, py_PTS.observed_pixels)
        np.testing.assert_allclose(
            cpp_PTS.weighted_counts, py_PTS.weighted_counts, rtol=rtol
        )
        np.testing.assert_array_equal(cpp_PTS.pixel_flag, py_PTS.pixel_flag)
        np.testing.assert_array_equal(cpp_PTS.old2new_pixel, py_PTS.old2new_pixel)

    def test_ProcessTimeSamples_QU_Cpp(self, initint, initfloat, rtol):
        solver_type = hpts.SolverType.QU

        cpp_PTS = brahmap.utilities.ProcessTimeSamples(
            npix=self.npix,
            pointings=initint.pointings,
            pointings_flag=self.pointings_flag,
            solver_type=solver_type,
            pol_angles=initfloat.pol_angles,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        py_PTS = hpts.ProcessTimeSamples(
            npix=self.npix,
            pointings=initint.pointings,
            pointings_flag=self.pointings_flag,
            solver_type=solver_type,
            pol_angles=initfloat.pol_angles,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        np.testing.assert_array_equal(cpp_PTS.pointings, py_PTS.pointings)
        np.testing.assert_array_equal(cpp_PTS.pointings_flag, py_PTS.pointings_flag)
        np.testing.assert_equal(cpp_PTS.new_npix, py_PTS.new_npix)
        np.testing.assert_array_equal(cpp_PTS.observed_pixels, py_PTS.observed_pixels)
        np.testing.assert_allclose(cpp_PTS.sin2phi, py_PTS.sin2phi, rtol=rtol)
        np.testing.assert_allclose(cpp_PTS.cos2phi, py_PTS.cos2phi, rtol=rtol)
        np.testing.assert_allclose(
            cpp_PTS.weighted_counts, py_PTS.weighted_counts, rtol=rtol
        )
        np.testing.assert_allclose(
            cpp_PTS.weighted_sin_sq, py_PTS.weighted_sin_sq, rtol=rtol
        )
        np.testing.assert_allclose(
            cpp_PTS.weighted_cos_sq, py_PTS.weighted_cos_sq, rtol=rtol
        )
        np.testing.assert_allclose(
            cpp_PTS.weighted_sincos, py_PTS.weighted_sincos, rtol=rtol
        )
        np.testing.assert_allclose(
            cpp_PTS.one_over_determinant, py_PTS.one_over_determinant, rtol=rtol
        )
        np.testing.assert_array_equal(cpp_PTS.pixel_flag, py_PTS.pixel_flag)
        np.testing.assert_array_equal(cpp_PTS.old2new_pixel, py_PTS.old2new_pixel)

    def test_ProcessTimeSamples_IQU_Cpp(self, initint, initfloat, rtol):
        solver_type = hpts.SolverType.IQU

        cpp_PTS = brahmap.utilities.ProcessTimeSamples(
            npix=self.npix,
            pointings=initint.pointings,
            pointings_flag=self.pointings_flag,
            solver_type=solver_type,
            pol_angles=initfloat.pol_angles,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        py_PTS = hpts.ProcessTimeSamples(
            npix=self.npix,
            pointings=initint.pointings,
            pointings_flag=self.pointings_flag,
            solver_type=solver_type,
            pol_angles=initfloat.pol_angles,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        np.testing.assert_array_equal(cpp_PTS.pointings, py_PTS.pointings)
        np.testing.assert_array_equal(cpp_PTS.pointings_flag, py_PTS.pointings_flag)
        np.testing.assert_equal(cpp_PTS.new_npix, py_PTS.new_npix)
        np.testing.assert_array_equal(cpp_PTS.observed_pixels, py_PTS.observed_pixels)
        np.testing.assert_allclose(cpp_PTS.sin2phi, py_PTS.sin2phi, rtol=rtol)
        np.testing.assert_allclose(cpp_PTS.cos2phi, py_PTS.cos2phi, rtol=rtol)
        np.testing.assert_allclose(
            cpp_PTS.weighted_counts, py_PTS.weighted_counts, rtol=rtol
        )
        np.testing.assert_allclose(
            cpp_PTS.weighted_sin_sq, py_PTS.weighted_sin_sq, rtol=rtol
        )
        np.testing.assert_allclose(
            cpp_PTS.weighted_cos_sq, py_PTS.weighted_cos_sq, rtol=rtol
        )
        np.testing.assert_allclose(
            cpp_PTS.weighted_sincos, py_PTS.weighted_sincos, rtol=rtol
        )
        np.testing.assert_allclose(cpp_PTS.weighted_sin, py_PTS.weighted_sin, rtol=rtol)
        np.testing.assert_allclose(cpp_PTS.weighted_cos, py_PTS.weighted_cos, rtol=rtol)
        np.testing.assert_allclose(
            cpp_PTS.one_over_determinant, py_PTS.one_over_determinant, rtol=rtol
        )
        np.testing.assert_array_equal(cpp_PTS.pixel_flag, py_PTS.pixel_flag)
        np.testing.assert_array_equal(cpp_PTS.old2new_pixel, py_PTS.old2new_pixel)


@pytest.mark.parametrize(
    "initint, initfloat, rtol",
    [
        (initint32, initfloat32, 1.5e-3),
        (initint64, initfloat32, 1.5e-3),
        (initint32, initfloat64, 1.5e-5),
        (initint64, initfloat64, 1.5e-5),
    ],
)
class TestProcessTimeSamples(InitCommonParams):
    def test_ProcessTimeSamples_I(self, initint, initfloat, rtol):
        solver_type = hpts.SolverType.I

        PTS = brahmap.utilities.ProcessTimeSamples(
            npix=self.npix,
            pointings=initint.pointings,
            pointings_flag=self.pointings_flag,
            solver_type=solver_type,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        weighted_counts = np.zeros(PTS.new_npix, dtype=initfloat.dtype)

        for idx in range(self.nsamples):
            if PTS.pointings_flag[idx]:
                pixel = PTS.pointings[idx]
                weighted_counts[pixel] += initfloat.noise_weights[idx]

        brahmap.bMPI.comm.Allreduce(MPI.IN_PLACE, weighted_counts, MPI.SUM)

        np.testing.assert_allclose(PTS.weighted_counts, weighted_counts, rtol=rtol)

    def test_ProcessTimeSamples_QU(self, initint, initfloat, rtol):
        solver_type = hpts.SolverType.QU

        PTS = brahmap.utilities.ProcessTimeSamples(
            npix=self.npix,
            pointings=initint.pointings,
            pointings_flag=self.pointings_flag,
            solver_type=solver_type,
            pol_angles=initfloat.pol_angles,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        sin2phi = np.empty(self.nsamples, dtype=initfloat.dtype)
        cos2phi = np.empty(self.nsamples, dtype=initfloat.dtype)
        sc.sin(self.nsamples, 2.0 * initfloat.pol_angles, sin2phi)
        sc.cos(self.nsamples, 2.0 * initfloat.pol_angles, cos2phi)

        weighted_counts = np.zeros(PTS.new_npix, dtype=initfloat.dtype)
        weighted_sin_sq = np.zeros(PTS.new_npix, dtype=initfloat.dtype)
        weighted_cos_sq = np.zeros(PTS.new_npix, dtype=initfloat.dtype)
        weighted_sincos = np.zeros(PTS.new_npix, dtype=initfloat.dtype)

        for idx in range(self.nsamples):
            if PTS.pointings_flag[idx]:
                pixel = PTS.pointings[idx]
                weighted_counts[pixel] += initfloat.noise_weights[idx]
                weighted_sin_sq[pixel] += (
                    initfloat.noise_weights[idx] * sin2phi[idx] * sin2phi[idx]
                )
                weighted_cos_sq[pixel] += (
                    initfloat.noise_weights[idx] * cos2phi[idx] * cos2phi[idx]
                )
                weighted_sincos[pixel] += (
                    initfloat.noise_weights[idx] * sin2phi[idx] * cos2phi[idx]
                )

        brahmap.bMPI.comm.Allreduce(MPI.IN_PLACE, weighted_counts, MPI.SUM)
        brahmap.bMPI.comm.Allreduce(MPI.IN_PLACE, weighted_sin_sq, MPI.SUM)
        brahmap.bMPI.comm.Allreduce(MPI.IN_PLACE, weighted_cos_sq, MPI.SUM)
        brahmap.bMPI.comm.Allreduce(MPI.IN_PLACE, weighted_sincos, MPI.SUM)

        one_over_determinant = 1.0 / (
            (weighted_cos_sq * weighted_sin_sq) - (weighted_sincos * weighted_sincos)
        )

        np.testing.assert_allclose(PTS.sin2phi, sin2phi, rtol=rtol)
        np.testing.assert_allclose(PTS.cos2phi, cos2phi, rtol=rtol)
        np.testing.assert_allclose(PTS.weighted_counts, weighted_counts, rtol=rtol)
        np.testing.assert_allclose(PTS.weighted_sin_sq, weighted_sin_sq, rtol=rtol)
        np.testing.assert_allclose(PTS.weighted_cos_sq, weighted_cos_sq, rtol=rtol)
        np.testing.assert_allclose(PTS.weighted_sincos, weighted_sincos, rtol=rtol)
        np.testing.assert_allclose(
            PTS.one_over_determinant, one_over_determinant, rtol=rtol
        )

    def test_ProcessTimeSamples_IQU(self, initint, initfloat, rtol):
        solver_type = hpts.SolverType.IQU

        PTS = brahmap.utilities.ProcessTimeSamples(
            npix=self.npix,
            pointings=initint.pointings,
            pointings_flag=self.pointings_flag,
            solver_type=solver_type,
            pol_angles=initfloat.pol_angles,
            noise_weights=initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            update_pointings_inplace=False,
        )

        sin2phi = np.empty(self.nsamples, dtype=initfloat.dtype)
        cos2phi = np.empty(self.nsamples, dtype=initfloat.dtype)
        sc.sin(self.nsamples, 2.0 * initfloat.pol_angles, sin2phi)
        sc.cos(self.nsamples, 2.0 * initfloat.pol_angles, cos2phi)

        weighted_counts = np.zeros(PTS.new_npix, dtype=initfloat.dtype)
        weighted_sin_sq = np.zeros(PTS.new_npix, dtype=initfloat.dtype)
        weighted_cos_sq = np.zeros(PTS.new_npix, dtype=initfloat.dtype)
        weighted_sincos = np.zeros(PTS.new_npix, dtype=initfloat.dtype)
        weighted_sin = np.zeros(PTS.new_npix, dtype=initfloat.dtype)
        weighted_cos = np.zeros(PTS.new_npix, dtype=initfloat.dtype)

        for idx in range(self.nsamples):
            if PTS.pointings_flag[idx]:
                pixel = PTS.pointings[idx]
                weighted_counts[pixel] += initfloat.noise_weights[idx]
                weighted_sin_sq[pixel] += (
                    initfloat.noise_weights[idx] * sin2phi[idx] * sin2phi[idx]
                )
                weighted_cos_sq[pixel] += (
                    initfloat.noise_weights[idx] * cos2phi[idx] * cos2phi[idx]
                )
                weighted_sincos[pixel] += (
                    initfloat.noise_weights[idx] * sin2phi[idx] * cos2phi[idx]
                )
                weighted_sin[pixel] += initfloat.noise_weights[idx] * sin2phi[idx]
                weighted_cos[pixel] += initfloat.noise_weights[idx] * cos2phi[idx]

        brahmap.bMPI.comm.Allreduce(MPI.IN_PLACE, weighted_counts, MPI.SUM)
        brahmap.bMPI.comm.Allreduce(MPI.IN_PLACE, weighted_sin, MPI.SUM)
        brahmap.bMPI.comm.Allreduce(MPI.IN_PLACE, weighted_cos, MPI.SUM)
        brahmap.bMPI.comm.Allreduce(MPI.IN_PLACE, weighted_sin_sq, MPI.SUM)
        brahmap.bMPI.comm.Allreduce(MPI.IN_PLACE, weighted_cos_sq, MPI.SUM)
        brahmap.bMPI.comm.Allreduce(MPI.IN_PLACE, weighted_sincos, MPI.SUM)

        one_over_determinant = 1.0 / (
            weighted_counts
            * (weighted_cos_sq * weighted_sin_sq - weighted_sincos * weighted_sincos)
            - weighted_cos * weighted_cos * weighted_sin_sq
            - weighted_sin * weighted_sin * weighted_cos_sq
            + 2.0 * weighted_cos * weighted_sin * weighted_sincos
        )

        np.testing.assert_allclose(PTS.sin2phi, sin2phi, rtol=rtol)
        np.testing.assert_allclose(PTS.cos2phi, cos2phi, rtol=rtol)
        np.testing.assert_allclose(PTS.weighted_counts, weighted_counts, rtol=rtol)
        np.testing.assert_allclose(PTS.weighted_sin_sq, weighted_sin_sq, rtol=rtol)
        np.testing.assert_allclose(PTS.weighted_cos_sq, weighted_cos_sq, rtol=rtol)
        np.testing.assert_allclose(PTS.weighted_sincos, weighted_sincos, rtol=rtol)
        np.testing.assert_allclose(PTS.weighted_sin, weighted_sin, rtol=rtol)
        np.testing.assert_allclose(PTS.weighted_cos, weighted_cos, rtol=rtol)
        np.testing.assert_allclose(
            PTS.one_over_determinant, one_over_determinant, rtol=rtol
        )


if __name__ == "__main__":
    pytest.main(
        [
            f"{__file__}::TestProcessTimeSamplesCpp::test_ProcessTimeSamples_I_Cpp",
            "-v",
            "-s",
        ]
    )
    pytest.main(
        [
            f"{__file__}::TestProcessTimeSamplesCpp::test_ProcessTimeSamples_QU_Cpp",
            "-v",
            "-s",
        ]
    )
    pytest.main(
        [
            f"{__file__}::TestProcessTimeSamplesCpp::test_ProcessTimeSamples_IQU_Cpp",
            "-v",
            "-s",
        ]
    )
    pytest.main(
        [
            f"{__file__}::TestProcessTimeSamples::test_ProcessTimeSamples_I",
            "-v",
            "-s",
        ]
    )

    pytest.main(
        [
            f"{__file__}::TestProcessTimeSamples::test_ProcessTimeSamples_QU",
            "-v",
            "-s",
        ]
    )

    pytest.main(
        [
            f"{__file__}::TestProcessTimeSamples::test_ProcessTimeSamples_IQU",
            "-v",
            "-s",
        ]
    )
