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
