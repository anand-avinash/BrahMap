import pytest


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
