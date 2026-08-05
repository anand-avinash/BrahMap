import pytest
import brahmap
from .benchmark_utils import BENCHMARK_SIZES


def pytest_addoption(parser):
    parser.addoption(
        "--size",
        action="store",
        default="medium",
        help="Benchmark size: small, medium, or large",
    )

    parser.addoption(
        "--dtype-int",
        action="store",
        default="int64",
        help="int32 or int64",
    )
    parser.addoption(
        "--dtype-float",
        action="store",
        default="float64",
        help="float32 or float64",
    )
    parser.addoption(
        "--nsamples",
        action="store",
        type=int,
        help="Global number of samples",
    )
    parser.addoption(
        "--nside",
        action="store",
        type=int,
        help="Nside for the healpix map",
    )
    parser.addoption(
        "--mpi-rounds",
        action="store",
        type=int,
        default=20,
        help="Number of rounds for mpi_benchmark",
    )
    parser.addoption(
        "--mpi-iterations",
        action="store",
        type=int,
        default=1,
        help="Number of iterations for mpi_benchmark",
    )
    parser.addoption(
        "--mpi-warmup-rounds",
        action="store",
        type=int,
        default=0,
        help="Number of warmup rounds for mpi_benchmark",
    )


@pytest.fixture(scope="session")
def bench_params(request):
    return {
        "benchmark_size": request.config.getoption("--size"),
        "dtype_int": request.config.getoption("--dtype-int"),
        "dtype_float": request.config.getoption("--dtype-float"),
        "nsamples": request.config.getoption("--nsamples"),
        "nside": request.config.getoption("--nside"),
    }


def _resolve_params(config):
    """Helper to resolve final benchmark parameters including overrides."""
    size = config.getoption("--size")
    if size in BENCHMARK_SIZES:
        params = BENCHMARK_SIZES[size].copy()
    else:
        params = BENCHMARK_SIZES["medium"].copy()

    nside = config.getoption("--nside")
    nsamples = config.getoption("--nsamples")
    if nside is not None:
        params["npix"] = 12 * nside**2
    if nsamples is not None:
        params["nsamples"] = nsamples

    return {
        "comm_size": brahmap.MPI_UTILS.size,
        "preset_size": size,
        "npix": params["npix"],
        "nsamples": params["nsamples"],
        "dtype_int": config.getoption("--dtype-int"),
        "dtype_float": config.getoption("--dtype-float"),
    }


def pytest_report_header(config):
    """Add benchmark parameters to the terminal report header."""
    if brahmap.MPI_UTILS.rank != 0:
        return []

    params = _resolve_params(config)
    return [
        "=" * 50,
        "BrahMap Benchmark Parameters:",
        f"  Comm size:    {params['comm_size']}",
        f"  Preset size:  {params['preset_size']}",
        f"  npix:         {params['npix']}",
        f"  nsamples:     {params['nsamples']}",
        f"  dtype_int:    {params['dtype_int']}",
        f"  dtype_float:  {params['dtype_float']}",
        "=" * 50,
    ]


def pytest_benchmark_update_machine_info(config, machine_info):
    """Add resolved benchmark parameters to the JSON report."""
    machine_info["benchmark_params"] = _resolve_params(config)


@pytest.fixture
def mpi_benchmark(benchmark, request):
    """A fixture to enable pedantic benchmark in order to enforce fixed
    iterations and rounds across all MPI ranks.
    """
    # For a generic benchmark, the rounds and the iterations are determined
    # individually by each MPI process. If the number of iterations or the
    # rounds are not equal across all MPI ranks, the standard benchmark runs
    # in segmentation faults. Setting the rounds and the iterations to the
    # same value for all MPI ranks avoids this issue.

    # Retrieve default CLI overrides
    cli_rounds = request.config.getoption("--mpi-rounds")
    cli_iterations = request.config.getoption("--mpi-iterations")
    cli_warmup = request.config.getoption("--mpi-warmup-rounds")

    def _run(func, *args, **kwargs):
        # Extract benchmark configuration, defaulting to CLI values (or their defaults)
        rounds = kwargs.pop("rounds", cli_rounds)
        iterations = kwargs.pop("iterations", cli_iterations)
        warmup_rounds = kwargs.pop("warmup_rounds", cli_warmup)
        setup = kwargs.pop("setup", None)

        # Setup function when used in benchmark.pedantic, can also be used to
        # supply the benchmark parameters. The setup functions is called at
        # the first iteration of every round. In some of the cases, we need
        # to supply the zero-ed arrays as the function/class arguments every
        # once in a while to prevent overflow/underflow. There, we can use
        # the setup argument to do so (for example, in `test_bench_extensions.py`)
        # See <https://pytest-benchmark.readthedocs.io/en/latest/pedantic.html#reference>
        return benchmark.pedantic(
            func,
            args=args,
            kwargs=kwargs,
            setup=setup,
            iterations=iterations,
            rounds=rounds,
            warmup_rounds=warmup_rounds,
        )

    return _run
