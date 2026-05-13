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
