import numpy as np
import scipy.linalg
import brahmap

from template_noise_ops_tests import BaseTestNoiseLO

import pytest


class NoiseOps_Toeplitz(BaseTestNoiseLO):
    def setup_method(self):
        dtype = getattr(self, "dtype")
        rtol = getattr(self, "rtol")
        atol = getattr(self, "atol")

        seed = 6545
        comm_rank = brahmap.MPI_UTILS.rank
        rng = np.random.default_rng(seed=[seed, comm_rank])

        size = 3 * (comm_rank + 1)

        covariance = rng.random(size, dtype=dtype)

        self.dtype = dtype
        self.size = size
        self.covariance = covariance

        extended_covariance1 = np.concatenate([covariance, covariance[1:-1][::-1]])
        power_spec1 = np.fft.fft(extended_covariance1).real.astype(
            dtype
        )  # power spectrum of size 2n-2

        extended_covariance2 = np.concatenate([covariance, covariance[1:][::-1]])
        power_spec2 = np.fft.fft(extended_covariance2).real.astype(
            dtype
        )  # power spectrum of size 2n-1

        operator1 = brahmap.core.NoiseCovLO_Toeplitz01(
            size=size,
            input=covariance,
            input_type="covariance",
            dtype=dtype,
        )
        operator2 = brahmap.core.NoiseCovLO_Toeplitz01(
            size=size,
            input=power_spec1,  # Power spectrum of size 2n-2
            input_type="power_spectrum",
            dtype=dtype,
        )
        numpy_operator1 = scipy.linalg.toeplitz(covariance)

        inv_operator1 = brahmap.core.InvNoiseCovLO_Toeplitz01(
            size=size,
            input=covariance,
            input_type="covariance",
            dtype=dtype,
        )
        inv_operator2 = brahmap.core.InvNoiseCovLO_Toeplitz01(
            size=size,
            input=power_spec2,  # Power spectrum of size 2n-1; Within this class, we are creating a Toeplitz operator (self.__toeplitz_op) using a power spectrum of size 2n-1. This Toeplitz operator must be same when we use a power spectrum of size 2n-2, that is operator2 defined above.
            input_type="power_spectrum",
            dtype=dtype,
        )
        numpy_inv_operator1 = scipy.linalg.inv(numpy_operator1)

        super().setup_method(
            operator1=operator1,
            operator2=operator2,
            numpy_operator1=numpy_operator1,
            inv_operator1=inv_operator1,
            inv_operator2=inv_operator2,
            numpy_inv_operator1=numpy_inv_operator1,
            rtol=rtol,
            atol=atol,
        )

    @pytest.mark.parametrize(
        "operator, numpy_operator",
        [
            ("operator1", "numpy_operator1"),
            ("operator2", "numpy_operator1"),
            # pytest.param(
            #     "inv_operator1",
            #     "numpy_inv_operator1",
            #     marks=pytest.mark.xfail(
            #         raises=NotImplementedError,
            #         reason=".diag() method is not implemented yet",
            #     ),
            # ),
            # pytest.param(
            #     "inv_operator2",
            #     "numpy_inv_operator1",
            #     marks=pytest.mark.xfail(
            #         raises=NotImplementedError,
            #         reason=".diag() method is not implemented yet",
            #     ),
            # ),
        ],
    )
    def test_diagonal(self, operator, numpy_operator):
        """Tests whether the `diag` attribute of operators are same as their
        diagonal
        """
        op = getattr(self, operator)
        np_op = getattr(self, numpy_operator)

        op_diag = op.diag
        print(op, operator)
        np_diag = np.diagonal(np_op)
        np.testing.assert_allclose(
            op_diag,
            np_diag,
            rtol=self.rtol,
            atol=self.atol,
        )

    @pytest.mark.parametrize(
        "precond_type", [("Strang"), ("TChan"), ("RChan"), ("KK2"), ("explicit")]
    )
    @pytest.mark.ignore_param_count
    def test_Strang_precond(self, precond_type):
        if precond_type == "explicit":
            precond_type = self.ex_inv_operator1

        inv_cov = brahmap.InvNoiseCovLO_Toeplitz01(
            size=self.size,
            input=self.covariance,
            input_type="covariance",
            precond_op=precond_type,
            dtype=self.dtype,
        )

        ex_inv_cov = inv_cov.to_array()

        np.testing.assert_allclose(
            ex_inv_cov,
            self.ex_inv_operator1,
            rtol=self.rtol,
            atol=self.atol,
        )


class TestNoiseOps_Toeplitz_F32(NoiseOps_Toeplitz):
    dtype = np.float32
    rtol = 1.0e-3
    atol = 1.0e-4


class TestNoiseOps_Toeplitz_F64(NoiseOps_Toeplitz):
    dtype = np.float64
    rtol = 1.0e-6
    atol = 1.0e-8
