import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--size",
        action="store",
        default="medium",
        help="Benchmark size: small, medium, or large",
    )


@pytest.fixture(scope="session")
def bench_size(request):
    return request.config.getoption("--size")
