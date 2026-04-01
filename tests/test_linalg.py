import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import pytest

import brahmap
from brahmap.math.linalg import cg, parallel_norm


@pytest.fixture
def linear_system():
    rng = np.random.default_rng(seed=867 + brahmap.MPI_UTILS.rank)
    mat_size = 40
    A = rng.random(size=(mat_size, mat_size))
    A = A + A.T  # A symmetric matrix
    A = A + mat_size * np.eye(mat_size)  # A positive definite matrix

    # numpy array to sparse linop
    A_op = brahmap.base.aslinearoperator(A)

    # RHS vector
    b = rng.random(mat_size)
    return A_op, b


def test_cg_against_scipy(linear_system):
    A_op, b = linear_system

    x_scipy, info_scipy = scipy.sparse.linalg.cg(A_op, b, atol=1e-12)

    x_brahmap, info_brahmap = cg(A_op, b, atol=1e-12)

    # testing convergence status
    np.testing.assert_equal(info_brahmap, info_scipy)

    # testing estimated solution
    np.testing.assert_allclose(x_brahmap, x_scipy, rtol=1e-5, atol=1e-5)


def test_parallel_norm():
    if brahmap.MPI_UTILS.rank == 0:
        rng = np.random.default_rng(seed=8671)
        arr = rng.random(10)
    else:
        arr = None

    arr = brahmap.MPI_UTILS.comm.bcast(arr, root=0)

    local_norm = np.linalg.norm(arr) ** 2
    exp_global_norm = np.sqrt(local_norm * brahmap.MPI_UTILS.size)

    global_norm = parallel_norm(arr)

    np.testing.assert_approx_equal(global_norm, exp_global_norm)


if __name__ == "__main__":
    pytest.main([f"{__file__}::test_cg_against_scipy", "-v", "-s"])
    pytest.main([f"{__file__}::test_parallel_norm", "-v", "-s"])
