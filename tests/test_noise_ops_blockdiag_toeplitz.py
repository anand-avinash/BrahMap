import numpy as np
import scipy.linalg
import brahmap


from template_noise_ops_tests import BaseTestNoiseLO


import pytest


class BlockDiagNoiseOps_Toeplitz(BaseTestNoiseLO):
    def setup_method(self):
        dtype = getattr(self, "dtype")
        rtol = getattr(self, "rtol")
        atol = getattr(self, "atol")

        seed = 15436
        comm_rank = brahmap.MPI_UTILS.rank
        rng = np.random.default_rng(seed=[seed, comm_rank])

        nblocks = 3 * (comm_rank + 1)
        block_size = rng.integers(low=4, high=7, size=nblocks)
        total_size = sum(block_size)

        numpy_operator1 = np.zeros((total_size, total_size), dtype=dtype)
        numpy_inv_operator1 = np.zeros_like(numpy_operator1)

        covariance_list = []
        power_spec_list = []
        start_idx = 0

        for idx in range(nblocks):
            covariance = rng.random(block_size[idx], dtype=dtype)
            extended_covariance = np.concatenate([covariance, covariance[1:-1][::-1]])
            power_spec = np.fft.fft(extended_covariance).real.astype(
                dtype
            )  # power spectrum of size 2n-2

            covariance_list.append(covariance)
            power_spec_list.append(power_spec)

            numpy_op = scipy.linalg.toeplitz(covariance)

            end_idx = start_idx + block_size[idx]
            numpy_operator1[start_idx:end_idx, start_idx:end_idx] = numpy_op
            numpy_inv_operator1[start_idx:end_idx, start_idx:end_idx] = np.linalg.inv(
                numpy_op
            )
            start_idx = end_idx

        operator1 = brahmap.core.BlockDiagNoiseCovLO(
            operator=brahmap.core.NoiseCovLO_Toeplitz01,
            block_size=block_size,
            block_input=covariance_list,
            input_type="covariance",
            dtype=dtype,
        )

        operator2 = brahmap.core.BlockDiagNoiseCovLO(
            operator=brahmap.core.NoiseCovLO_Toeplitz01,
            block_size=block_size,
            block_input=power_spec_list,
            input_type="power_spectrum",
            dtype=dtype,
        )

        inv_operator1 = brahmap.core.BlockDiagInvNoiseCovLO(
            operator=brahmap.core.InvNoiseCovLO_Toeplitz01,
            block_size=block_size,
            block_input=covariance_list,
            input_type="covariance",
            dtype=dtype,
            extra_kwargs={
                "precond_op": "TChan",
            },
        )

        inv_operator2 = brahmap.core.BlockDiagInvNoiseCovLO(
            operator=brahmap.core.InvNoiseCovLO_Toeplitz01,
            block_size=block_size,
            block_input=power_spec_list,
            input_type="power_spectrum",
            dtype=dtype,
        )

        return super().setup_method(
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


class TestBlockDiagNoiseOps_Toeplitz_F32(BlockDiagNoiseOps_Toeplitz):
    dtype = np.float32
    rtol = 1.0e-3
    atol = 1.0e-3


class TestBlockDiagNoiseOps_Toeplitz_F64(BlockDiagNoiseOps_Toeplitz):
    dtype = np.float64
    rtol = 1.0e-6
    atol = 1.0e-8
