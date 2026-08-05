import pytest
from brahmap.core import (
    ProcessTimeSamples,
    SharedMemProcessTimeSamples,
    PointingLO,
    BlockDiagonalPreconditionerLO,
    compute_GLS_maps_from_PTS,
    GLSParameters,
)
from .benchmark_utils import setup_common_data, get_solver_types


@pytest.fixture(scope="module")
def data(bench_params):
    return setup_common_data(params=bench_params)


@pytest.fixture(params=get_solver_types())
def stype(request):
    """Fixture to provide solver types one by one.
    See <https://docs.pytest.org/en/stable/how-to/fixtures.html#parametrizing-fixtures>
    """
    return request.param


@pytest.mark.benchmark(group="core::ProcessTimeSamples")
class TestProcessTimeSamples:
    def test_bench_process_time_samples(self, mpi_benchmark, data, stype):
        def run():
            ProcessTimeSamples(
                npix=data["npix"],
                pointings=data["pointings"],
                pointings_flag=data["pointings_flag"],
                solver_type=stype,
                pol_angles=data["pol_angles"],
                noise_weights=data["noise_weights"],
                update_pointings_inplace=False,
            )

        mpi_benchmark(run)

    def test_bench_shmem_process_time_samples(self, mpi_benchmark, data, stype):
        def run():
            shm_PTS = SharedMemProcessTimeSamples(
                npix=data["npix"],
                pointings=data["pointings"],
                pointings_flag=data["pointings_flag"],
                solver_type=stype,
                pol_angles=data["pol_angles"],
                noise_weights=data["noise_weights"],
                update_pointings_inplace=False,
                nproc_reduce=1,
            )
            shm_PTS.free_shmem_arrays()

        mpi_benchmark(run)


@pytest.fixture
def processed_samples(data, stype):
    """Initializes ProcessTimeSamples for ONLY the current stype."""
    return ProcessTimeSamples(
        npix=data["npix"],
        pointings=data["pointings"],
        pointings_flag=data["pointings_flag"],
        solver_type=stype,
        pol_angles=data["pol_angles"],
        noise_weights=data["noise_weights"],
    )


@pytest.fixture
def processed_shmem_samples(data, stype):
    """Initializes SharedMemProcessTimeSamples for ONLY the current stype."""
    shm_PTS = SharedMemProcessTimeSamples(
        npix=data["npix"],
        pointings=data["pointings"],
        pointings_flag=data["pointings_flag"],
        solver_type=stype,
        pol_angles=data["pol_angles"],
        noise_weights=data["noise_weights"],
        nproc_reduce=1,
    )
    yield shm_PTS
    shm_PTS.free_shmem_arrays()


@pytest.mark.benchmark(group="core::LinearOperators")
class TestLinearOperators:
    def test_bench_PointingLO_matvec(
        self,
        benchmark,
        data,
        processed_samples,
    ):
        lo = PointingLO(processed_samples)
        vec = (
            data["rng"]
            .random(lo.ncols)
            .astype(
                processed_samples.dtype_float,
            )
        )
        benchmark(lo.matvec, vec)

    def test_bench_PointingLO_rmatvec(
        self,
        mpi_benchmark,
        data,
        processed_samples,
    ):
        lo = PointingLO(processed_samples)
        vec = (
            data["rng"]
            .random(lo.nrows)
            .astype(
                processed_samples.dtype_float,
            )
        )
        mpi_benchmark(lo.T.matvec, vec)

    def test_bench_shmem_PointingLO_matvec(
        self,
        benchmark,
        data,
        processed_shmem_samples,
    ):
        lo = PointingLO(processed_shmem_samples)
        vec = (
            data["rng"]
            .random(lo.ncols)
            .astype(
                processed_shmem_samples.dtype_float,
            )
        )
        benchmark(lo.matvec, vec)

    def test_bench_shmem_PointingLO_rmatvec(
        self,
        mpi_benchmark,
        data,
        processed_shmem_samples,
    ):
        lo = PointingLO(processed_shmem_samples)
        vec = (
            data["rng"]
            .random(lo.nrows)
            .astype(
                processed_shmem_samples.dtype_float,
            )
        )
        mpi_benchmark(lo.T.matvec, vec)

    def test_bench_BDPLO_matvec(
        self,
        benchmark,
        data,
        processed_samples,
    ):
        lo = BlockDiagonalPreconditionerLO(processed_samples)
        vec = (
            data["rng"]
            .random(lo.nargin)
            .astype(
                processed_samples.dtype_float,
            )
        )
        benchmark(lo.matvec, vec)


@pytest.mark.benchmark(group="core::GLS_solver")
class TestGLS:
    def test_bench_compute_GLS_maps(
        self,
        mpi_benchmark,
        data,
        processed_samples,
        stype,
    ):
        gls_params = GLSParameters(
            solver_type=stype,
            isolver_max_iterations=3,
        )
        tod = (
            data["rng"]
            .random(data["nsamples"])
            .astype(
                processed_samples.dtype_float,
            )
        )
        mpi_benchmark(
            compute_GLS_maps_from_PTS,
            processed_samples,
            tod,
            None,
            gls_params,
        )

    def test_bench_shmem_compute_GLS_maps(
        self,
        mpi_benchmark,
        data,
        processed_shmem_samples,
        stype,
    ):
        gls_params = GLSParameters(
            solver_type=stype,
            isolver_max_iterations=3,
        )
        tod = (
            data["rng"]
            .random(data["nsamples"])
            .astype(
                processed_shmem_samples.dtype_float,
            )
        )
        mpi_benchmark(
            compute_GLS_maps_from_PTS,
            processed_shmem_samples,
            tod,
            None,
            gls_params,
        )
