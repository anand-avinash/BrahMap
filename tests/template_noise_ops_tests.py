import numpy as np
import pytest


class BaseTestNoiseLO:
    """Base class for testing the implementation of noise covariance (and
    inverse) operators.

    This class provides common test functions to validate noise covariance
    operator implementations defined in BrahMap.

    When subclassed, all test methods defined in this class will be
    automatically discovered and executed by pytest, enabling consistent
    testing of various noise covariance operator implementations.

    Usage:
        - Subclass this base class
        - Initialize the following operators with **same** noise properties:
            - operator1: Noise covariance operator defined with
                `input_type="covariance"`
            - operator2: Noise covariance operator defined with
                `input_type="power_spectrum"`
            - numpy_operator1: Explicit 2D numpy-array representation of
                `operator1` and `operator2`
            - inv_operator1: Inverse noise covariance operator defined with
                `input_type="covariance"`
            - inv_operator2: Inverse noise covariance operator defined with
                `input_type="power_spectrum"`
            - numpy_inv_operator1: Explicit 2D numpy-array representation of
                `inv_operator1` and `inv_operator2`
        - Set appropriate `rtol` and `atol` values
        - Call `super().setup_method(...)` with the arguments defined above
        - Run pytest to execute all inherited test cases
    """

    def setup_method(
        self,
        operator1,
        operator2,
        numpy_operator1,
        inv_operator1,
        inv_operator2,
        numpy_inv_operator1,
        rtol,
        atol,
    ):
        self.operator1 = operator1
        self.operator2 = operator2
        self.numpy_operator1 = numpy_operator1

        self.inv_operator1 = inv_operator1
        self.inv_operator2 = inv_operator2
        self.numpy_inv_operator1 = numpy_inv_operator1

        self.rtol = rtol
        self.atol = atol

        self.ex_operator1 = self.operator1.to_array()
        self.ex_operator2 = self.operator2.to_array()

        self.ex_inv_operator1 = self.inv_operator1.to_array()
        self.ex_inv_operator2 = self.inv_operator2.to_array()

        self.identity_operator = np.eye(
            self.operator1.size,
            dtype=self.operator1.dtype,
        )

    def test_cov_op_explicit_numpy(self):
        """Tests whether the matrix form of operator1 is same as the matrix we
        expect
        """
        np.testing.assert_allclose(
            self.numpy_operator1,
            self.ex_operator1,
            rtol=self.rtol,
        )

    def test_inv_cov_op_explicit_numpy(self):
        """Tests whether the matrix form of inv_operator1 is same as the
        matrix we expect
        """
        np.testing.assert_allclose(
            self.numpy_inv_operator1,
            self.ex_inv_operator1,
            rtol=self.rtol,
        )

    def test_cov_initialization(self):
        """Tests whether the matrix form of operator1 is same as the matrix we expect"""
        np.testing.assert_allclose(
            self.ex_operator1,
            self.ex_operator2,
            rtol=self.rtol,
        )

    def test_inv_cov_initialization(self):
        """Tests whether the matrix form of operator1 and operator2 are same"""
        np.testing.assert_allclose(
            self.ex_inv_operator1,
            self.ex_inv_operator2,
            rtol=self.rtol,
        )

    def test_inversion(self):
        """Tests whether the inv_operator1 is indeed an inverse of operator1"""
        product = self.operator1 * self.inv_operator1
        np.testing.assert_allclose(
            self.identity_operator,
            product.to_array(),
            rtol=self.rtol,
            atol=self.atol,
        )

    @pytest.mark.parametrize(
        "operator, numpy_operator",
        [
            ("operator1", "numpy_operator1"),
            ("operator2", "numpy_operator1"),
            ("inv_operator1", "numpy_inv_operator1"),
            ("inv_operator2", "numpy_inv_operator1"),
        ],
    )
    @pytest.mark.ignore_param_count
    def test_diagonal__ignore_param_count__(self, operator, numpy_operator):
        """Tests whether the `diag` attribute of operators are same as their
        diagonal
        """
        op = getattr(self, operator)
        np_op = getattr(self, numpy_operator)

        op_diag = op.diag
        np_diag = np.diagonal(np_op)
        np.testing.assert_allclose(
            np_diag,
            op_diag,
            rtol=self.rtol,
            atol=self.atol,
        )

    @pytest.mark.parametrize(
        "operator",
        [
            "operator1",
            "operator2",
            "inv_operator1",
            "inv_operator2",
        ],
    )
    @pytest.mark.ignore_param_count
    def test_get_inverse(self, operator):
        """Tests whether the method `get_inverse()` returns the inverse
        operator
        """
        op = getattr(self, operator)
        inverse = op.get_inverse()
        product = op * inverse
        np.testing.assert_allclose(
            self.identity_operator,
            product.to_array(),
            rtol=self.rtol,
            atol=self.atol,
        )
