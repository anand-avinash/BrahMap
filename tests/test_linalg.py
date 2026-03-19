import pytest
import numpy as np
import scipy.sparse.linalg as spla
from brahmap.math.linalg import cg
from brahmap.base import MatrixLinearOperator

@pytest.fixture
def linalg_system():
    # Generate a small symmetric positive-definite matrix A and vector b
    np.random.seed(42)
    n = 10
    A_dense = np.random.randn(n, n)
    # Make A symmetric and positive-definite
    A_dense = np.dot(A_dense.T, A_dense) + np.eye(n) * 1e-3
    b = np.random.randn(n)

    # Wrap A in MatrixLinearOperator
    A = MatrixLinearOperator(A_dense)
    return A, A_dense, b

def test_cg_convergence_parallel_true(linalg_system):
    A, A_dense, b = linalg_system

    # Expected result from scipy.sparse.linalg.cg
    x_expected, info_expected = spla.cg(A_dense, b)
    assert info_expected == 0, "Scipy CG did not converge"

    # Test our cg implementation with parallel=True
    x_actual, info_actual = cg(A, b, parallel=True)

    assert info_actual == 0, "Brahmap CG did not converge"
    np.testing.assert_allclose(x_actual, x_expected, rtol=1e-5, atol=1e-5)

def test_cg_convergence_parallel_false(linalg_system):
    A, A_dense, b = linalg_system

    # Expected result from scipy.sparse.linalg.cg
    x_expected, info_expected = spla.cg(A_dense, b)
    assert info_expected == 0, "Scipy CG did not converge"

    # Test our cg implementation with parallel=False
    x_actual, info_actual = cg(A, b, parallel=False)

    assert info_actual == 0, "Brahmap CG did not converge"
    np.testing.assert_allclose(x_actual, x_expected, rtol=1e-5, atol=1e-5)

def test_cg_exact_initial_guess(linalg_system):
    A, A_dense, b = linalg_system

    # Expected result from scipy.sparse.linalg.cg
    x_expected, info_expected = spla.cg(A_dense, b)
    assert info_expected == 0, "Scipy CG did not converge"

    # Use exact solution as initial guess
    x0 = x_expected.copy()

    x_actual, info_actual = cg(A, b, x0=x0, parallel=False)

    # Should converge in 0 iterations (or immediately)
    assert info_actual == 0
    np.testing.assert_allclose(x_actual, x_expected, rtol=1e-5, atol=1e-5)

def test_cg_zero_b(linalg_system):
    A, _, _ = linalg_system
    n = A.shape[0]

    b_zero = np.zeros(n)

    x_actual, info_actual = cg(A, b_zero)

    assert info_actual == 0
    np.testing.assert_allclose(x_actual, np.zeros(n), rtol=1e-5, atol=1e-5)
