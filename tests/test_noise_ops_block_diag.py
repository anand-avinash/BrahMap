import numpy as np
import brahmap


from template_noise_ops_tests import BaseTestNoiseLO


class BlockDiagNoiseOps_Diagonal(BaseTestNoiseLO):
    def setup_method(self):
        dtype = getattr(self, "dtype")
        rtol = getattr(self, "rtol")
        atol = getattr(self, "atol")

        seed = 1543
        comm_rank = brahmap.MPI_UTILS.rank
        rng = np.random.default_rng(seed=[seed, comm_rank])

        nblocks = 7 * (comm_rank + 1)
        block_size = rng.integers(low=10, high=20, size=nblocks)
        total_size = sum(block_size)

        numpy_operator1 = np.zeros((total_size, total_size), dtype=dtype)
        numpy_inv_operator1 = np.zeros_like(numpy_operator1)

        covariance_list = []
        power_spec_list = []
        start_idx = 0

        for idx in range(nblocks):
            covariance = rng.random(block_size[idx], dtype=dtype)
            power_spec = np.fft.fft(covariance).real.astype(dtype=dtype)

            covariance = np.fft.ifft(power_spec).real.astype(dtype=dtype)
            power_spec = np.fft.fft(covariance).real.astype(dtype=dtype)

            covariance_list.append(covariance)
            power_spec_list.append(power_spec)

            end_idx = start_idx + block_size[idx]
            numpy_operator1[start_idx:end_idx, start_idx:end_idx] = np.diag(covariance)
            numpy_inv_operator1[start_idx:end_idx, start_idx:end_idx] = np.diag(
                1.0 / covariance
            )
            start_idx = end_idx

        operator1 = brahmap.base.BlockDiagNoiseCovLO(
            operator=brahmap.core.NoiseCovLO_Diagonal,
            block_size=block_size,
            block_input=covariance_list,
            input_type="covariance",
            dtype=dtype,
        )

        operator2 = brahmap.base.BlockDiagNoiseCovLO(
            operator=brahmap.core.NoiseCovLO_Diagonal,
            block_size=block_size,
            block_input=power_spec_list,
            input_type="power_spectrum",
            dtype=dtype,
        )

        inv_operator1 = brahmap.base.BlockDiagInvNoiseCovLO(
            operator=brahmap.core.InvNoiseCovLO_Diagonal,
            block_size=block_size,
            block_input=covariance_list,
            input_type="covariance",
            dtype=dtype,
        )

        inv_operator2 = brahmap.base.BlockDiagInvNoiseCovLO(
            operator=brahmap.core.InvNoiseCovLO_Diagonal,
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


class TestBlockDiagNoiseOps_Diagonal_F32(BlockDiagNoiseOps_Diagonal):
    dtype = np.float32
    rtol = 1.0e-5
    atol = 1.0e-10


class TestBlockDiagNoiseOps_Diagonal_F64(BlockDiagNoiseOps_Diagonal):
    dtype = np.float64
    rtol = 1.0e-7
    atol = 1.0e-15
