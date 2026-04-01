import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import pytest

import brahmap
from brahmap.math.linalg import cg


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
