import pytest
from brahmap.core import (
    NoiseCovLO_Diagonal,
    InvNoiseCovLO_Diagonal,
    NoiseCovLO_Circulant,
    InvNoiseCovLO_Circulant,
    NoiseCovLO_Toeplitz01,
    InvNoiseCovLO_Toeplitz01,
)
from .benchmark_utils import setup_common_data


@pytest.fixture(scope="module")
def data(bench_params):
    return setup_common_data(params=bench_params)


@pytest.mark.benchmark(group="noise_ops::Diagonal")
class TestDiagonalCov:
    def test_bench_diag_matvec(self, benchmark, data):
        size = data["nsamples"]
        lo = NoiseCovLO_Diagonal(size=size, input=data["noise_weights"])
        vec = data["rng"].random(size).astype(data["dtype_float"])
        benchmark(lo.matvec, vec)

    def test_bench_invdiag_matvec(self, benchmark, data):
        size = data["nsamples"]
        lo = InvNoiseCovLO_Diagonal(size=size, input=data["noise_weights"])
        vec = data["rng"].random(size).astype(data["dtype_float"])
        benchmark(lo.matvec, vec)


@pytest.mark.benchmark(group="noise_ops::Circulant")
class TestCirculantCov:
    def test_bench_circ_matvec(self, benchmark, data):
        size = data["nsamples"]
        # For circulant, input size should be size // 2 + 1 if power_spectrum
        ps = data["rng"].random(size // 2 + 1).astype(data["dtype_float"])
        lo = NoiseCovLO_Circulant(size=size, input=ps, input_type="power_spectrum")
        vec = data["rng"].random(size).astype(data["dtype_float"])
        benchmark(lo.matvec, vec)

    def test_bench_invcirc_matvec(self, benchmark, data):
        size = data["nsamples"]
        ps = data["rng"].random(size // 2 + 1).astype(data["dtype_float"])
        lo = InvNoiseCovLO_Circulant(size=size, input=ps, input_type="power_spectrum")
        vec = data["rng"].random(size).astype(data["dtype_float"])
        benchmark(lo.matvec, vec)


@pytest.mark.benchmark(group="noise_ops::Toeplitz")
class TestToeplitzCov:
    def test_bench_toep01_matvec(self, benchmark, data):
        size = data["nsamples"]
        # For toeplitz, power spectrum size is 2*size - 2
        ps = data["rng"].random(2 * size - 2).astype(data["dtype_float"])
        lo = NoiseCovLO_Toeplitz01(size=size, input=ps, input_type="power_spectrum")
        vec = data["rng"].random(size).astype(data["dtype_float"])
        benchmark(lo.matvec, vec)

    def test_bench_invtoep01_matvec(self, benchmark, data):
        size = data["nsamples"]
        ps = data["rng"].random(2 * size - 2).astype(data["dtype_float"])
        lo = InvNoiseCovLO_Toeplitz01(
            size=size,
            input=ps,
            input_type="power_spectrum",
            precond_maxiter=5,
        )
        vec = data["rng"].random(size).astype(data["dtype_float"])
        benchmark(lo.matvec, vec)
