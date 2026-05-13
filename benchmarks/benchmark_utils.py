import numpy as np
import brahmap
from brahmap.core import SolverType


BENCHMARK_SIZES = {
    "small": {"npix": 12*256*256, "nsamples": 10**5},
    "medium": {"npix": 12*512*512, "nsamples": 10**6},
    "large": {"npix": 12*1024*1024, "nsamples": 10**8},
}


def setup_common_data(
    benchmark_size="medium",
    gen_pol_angles=True,
    gen_noise_weights=True,
    **kwargs,
):
    if benchmark_size in BENCHMARK_SIZES:
        params = BENCHMARK_SIZES[benchmark_size]
    else:
        params = BENCHMARK_SIZES["medium"]

    dtype_int = np.int64
    dtype_float = np.float64

    params.update(kwargs)
    npix = params["npix"]
    nsamples = params["nsamples"]

    comm_rank = brahmap.MPI_UTILS.rank
    comm_size = brahmap.MPI_UTILS.size
    rng = np.random.default_rng(seed=[1234, comm_rank])

    div, rem = divmod(nsamples, comm_size)
    local_nsamples = div + (comm_rank < rem)

    pointings = rng.integers(
        low=0,
        high=npix,
        size=local_nsamples,
        dtype=dtype_int,
    )

    nsamples_bad = npix
    div, rem = divmod(nsamples_bad, comm_size)
    local_nsamples_bad = div + (comm_rank < rem)

    pointings_flag = np.ones(local_nsamples, dtype=bool)
    bad_samples = rng.integers(
        low=0,
        high=local_nsamples,
        size=local_nsamples_bad,
    )
    pointings_flag[bad_samples] = False

    if gen_pol_angles:
        pol_angles = rng.uniform(-np.pi / 2, np.pi / 2, local_nsamples).astype(
            dtype=dtype_float,
        )
    if gen_noise_weights:
        noise_weights = rng.random(local_nsamples).astype(dtype=dtype_float)

    return {
        "npix": npix,
        "nsamples_global": nsamples,
        "nsamples": local_nsamples,
        "pointings": pointings,
        "pointings_flag": pointings_flag,
        "pol_angles": pol_angles if gen_pol_angles else None,
        "noise_weights": noise_weights if gen_noise_weights else None,
        "dtype_int": dtype_int,
        "dtype_float": dtype_float,
        "rng": rng,
    }


def get_solver_types():
    return [SolverType.I, SolverType.QU, SolverType.IQU]
