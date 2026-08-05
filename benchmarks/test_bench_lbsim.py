import tempfile
import pytest
from .benchmark_utils import setup_common_data

# Skip this module if litebird_sim is not installed
litebird_sim = pytest.importorskip(
    modname="litebird_sim",
    reason="Couldn't import `litebird_sim` module",
)
import litebird_sim as lbs  # noqa: E402

from brahmap.lbsim import (  # noqa: E402
    LBSimProcessTimeSamples,
    LBSimSharedMemProcessTimeSamples,
    LBSim_InvNoiseCovLO_UnCorr,
    LBSim_InvNoiseCovLO_Circulant,
    LBSim_InvNoiseCovLO_Toeplitz,
)


@pytest.fixture(scope="module")
def lbsim_data(bench_params):
    """
    Fixture to provide real litebird_sim.Simulation objects.
    Uses a minimal simulation setup for benchmarking.
    """

    size_params = setup_common_data(params=bench_params)
    dtype_float = size_params["dtype_float"]
    nside = size_params["nside"]
    rng = size_params["rng"]

    comm = lbs.MPI_COMM_WORLD
    tmp_dir = tempfile.TemporaryDirectory()
    imo = lbs.Imo(flatfile_location=lbs.PTEP_IMO_LOCATION)

    ### Mission params
    telescope = "MFT"
    channel = "M1-195"
    detector_list = [
        "001_002_030_00A_195_B",
        "001_002_029_45B_195_B",
        "001_002_015_15A_195_T",
        "001_002_047_00A_195_B",
    ]
    imo_version = "vPTEP"
    detector_sampling_freq = 1

    sim = lbs.Simulation(
        base_path=tmp_dir.name,
        start_time=234,
        duration_s=size_params["nsamples"],
        random_seed=65454,
        mpi_comm=comm,
        imo=imo,
    )

    ### Instrument definition
    sim.set_instrument(
        lbs.InstrumentInfo.from_imo(
            imo,
            f"/releases/{imo_version}/satellite/{telescope}/instrument_info",
        )
    )

    ### Detector list
    dets = []
    for n_det in detector_list:
        det = lbs.DetectorInfo.from_imo(
            url=f"/releases/{imo_version}/satellite/{telescope}/{channel}/{n_det}/detector_info",
            imo=imo,
        )
        det.sampling_rate_hz = detector_sampling_freq
        dets.append(det)

    ### Scanning strategy
    sim.set_scanning_strategy(
        imo_url=f"/releases/{imo_version}/satellite/scanning_parameters/"
    )

    sim.create_observations(
        detectors=dets,
        num_of_obs_per_detector=3,
        n_blocks_det=1,
        n_blocks_time=lbs.MPI_COMM_WORLD.size,
        split_list_over_processes=False,
        tod_dtype=dtype_float,
    )

    return nside, dtype_float, sim, rng


@pytest.mark.benchmark(group="LBSim::noise_ops")
class TestLBSimNoiseOps:
    def test_bench_invUnCorr_matvec(self, benchmark, lbsim_data):
        _, dtype_float, sim, rng = lbsim_data
        lo = LBSim_InvNoiseCovLO_UnCorr(obs=sim.observations)
        vec = rng.random(lo.nargin).astype(dtype_float)
        benchmark(lo.matvec, vec)

    def test_bench_invCirculant_matvec(self, benchmark, lbsim_data):
        _, dtype_float, sim, rng = lbsim_data
        nsamples = sim.observations[0].n_samples
        ps = rng.random(nsamples).astype(dtype_float)

        lo = LBSim_InvNoiseCovLO_Circulant(obs=sim.observations, input=ps)
        vec = rng.random(lo.nargin).astype(dtype_float)
        benchmark(lo.matvec, vec)

    def test_bench_invToeplitz_matvec(self, benchmark, lbsim_data):
        _, dtype_float, sim, rng = lbsim_data
        nsamples = sim.observations[0].n_samples
        ps_len = 2 * nsamples - 2
        ps = rng.random(ps_len).astype(dtype_float)

        lo = LBSim_InvNoiseCovLO_Toeplitz(
            obs=sim.observations,
            input=ps,
            extra_kwargs={
                "precond_maxiter": 2,
            },
        )
        vec = rng.random(lo.nargin).astype(dtype_float)
        benchmark(lo.matvec, vec)


@pytest.mark.benchmark(group="LBSim::LBSimProcessTimeSamples")
class TestLBSimPTS:
    def test_bench_LBSimProcessTimeSamples(self, mpi_benchmark, lbsim_data):
        nside, _, sim, _ = lbsim_data
        sim.prepare_pointings()

        mpi_benchmark(
            LBSimProcessTimeSamples,
            nside=nside,
            observations=sim.observations,
        )

    def test_bench_LBSimSharedMemProcessTimeSamples(self, mpi_benchmark, lbsim_data):
        nside, _, sim, _ = lbsim_data
        sim.prepare_pointings()

        active_shm = []

        def run():
            shm_PTS = LBSimSharedMemProcessTimeSamples(
                nside=nside,
                observations=sim.observations,
            )
            active_shm.append(shm_PTS)

        def teardown():
            while active_shm:
                shm_PTS = active_shm.pop()
                shm_PTS.free_shmem_arrays()

        mpi_benchmark(run, teardown=teardown)
