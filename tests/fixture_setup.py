import pytest
import numpy as np


@pytest.fixture(scope="module", params=[np.float32, np.float64])
def setup_linalg_tools(request):
    import numpy as np
    import brahmap

    # Every time this fixture is called, a new set of parameter will be taken
    # from the param list supplied to the fixture with `params` argument
    # <https://docs.pytest.org/en/stable/how-to/fixtures.html#parametrizing-fixtures>
    dtype = request.param

    class InitParams:
        def __init__(self, dtype):
            nsamples_global = 1180
            div, rem = divmod(nsamples_global, brahmap.MPI_UTILS.size)
            self.nsamples = div + (brahmap.MPI_UTILS.rank < rem)

            self.dtype = dtype
            self.diag = np.random.random(size=self.nsamples).astype(dtype=self.dtype)
            self.vec = np.random.random(size=self.nsamples).astype(dtype=self.dtype)

    return InitParams(dtype=dtype)


@pytest.fixture(
    scope="module",
    params=[
        (np.int32, np.float32),
        (np.int64, np.float32),
        (np.int32, np.float64),
        (np.int64, np.float64),
    ],
    ids=[
        "int32-float32",
        "int64-float32",
        "int32-float64",
        "int64-float64",
    ],
)
def setup_scan(request):
    import numpy as np
    import brahmap

    dtype_int, dtype_float = request.param

    class InitCommonParams:
        def __init__(self) -> None:
            np.random.seed(54321 + brahmap.MPI_UTILS.rank)
            self.npix = 128
            nsamples_global = self.npix * 6

            div, rem = divmod(nsamples_global, brahmap.MPI_UTILS.size)
            self.nsamples = div + (brahmap.MPI_UTILS.rank < rem)

            nbad_pixels_global = self.npix
            div, rem = divmod(nbad_pixels_global, brahmap.MPI_UTILS.size)
            nbad_pixels = div + (brahmap.MPI_UTILS.rank < rem)

            self.pointings_flag = np.ones(self.nsamples, dtype=bool)
            bad_samples = np.random.randint(low=0, high=self.nsamples, size=nbad_pixels)
            self.pointings_flag[bad_samples] = False

    class InitIntParams(InitCommonParams):
        def __init__(self, dtype) -> None:
            super().__init__()
            self.dtype = dtype
            self.pointings = np.random.randint(
                low=0, high=self.npix, size=self.nsamples, dtype=self.dtype
            )

    class InitFloatParams(InitCommonParams):
        def __init__(self, dtype) -> None:
            super().__init__()
            self.dtype = dtype
            self.noise_weights = np.random.random(size=self.nsamples).astype(
                dtype=self.dtype
            )
            self.pol_angles = np.random.uniform(
                low=-np.pi / 2.0, high=np.pi / 2.0, size=self.nsamples
            ).astype(dtype=self.dtype)
            self.vec = np.random.random(size=self.npix * 3).astype(dtype=self.dtype)
            self.rvec = np.random.random(size=self.nsamples).astype(dtype=self.dtype)

    initint = InitIntParams(dtype=dtype_int)
    initfloat = InitFloatParams(dtype=dtype_float)

    return initint, initfloat
