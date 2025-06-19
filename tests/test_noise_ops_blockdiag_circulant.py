import numpy as np
import brahmap


from template_noise_ops_tests import BaseTestNoiseLO


class BlockDiagNoiseOps_Circulant(BaseTestNoiseLO):
    def setup_method(self):
        dtype = getattr(self, "dtype")
        rtol = getattr(self, "rtol")
        atol = getattr(self, "atol")

        seed = 1543
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
            power_spec = np.fft.fft(covariance).real.astype(dtype=dtype)

            covariance = np.fft.ifft(power_spec).real.astype(dtype=dtype)
            power_spec = np.fft.fft(covariance).real.astype(dtype=dtype)

            covariance_list.append(covariance)
            power_spec_list.append(power_spec)

            numpy_op = np.zeros(shape=(block_size[idx], block_size[idx]), dtype=dtype)
            for i in range(block_size[idx]):
                for j in range(block_size[idx]):
                    numpy_op[i, j] = covariance[(i - j) % block_size[idx]]

            end_idx = start_idx + block_size[idx]
            numpy_operator1[start_idx:end_idx, start_idx:end_idx] = numpy_op
            numpy_inv_operator1[start_idx:end_idx, start_idx:end_idx] = np.linalg.inv(
                numpy_op
            )
            start_idx = end_idx

        operator1 = brahmap.core.BlockDiagNoiseCovLO(
            operator=brahmap.core.NoiseCovLO_Circulant,
            block_size=block_size,
            block_input=covariance_list,
            input_type="covariance",
            dtype=dtype,
        )

        operator2 = brahmap.core.BlockDiagNoiseCovLO(
            operator=brahmap.core.NoiseCovLO_Circulant,
            block_size=block_size,
            block_input=power_spec_list,
            input_type="power_spectrum",
            dtype=dtype,
        )

        inv_operator1 = brahmap.core.BlockDiagInvNoiseCovLO(
            operator=brahmap.core.InvNoiseCovLO_Circulant,
            block_size=block_size,
            block_input=covariance_list,
            input_type="covariance",
            dtype=dtype,
        )

        inv_operator2 = brahmap.core.BlockDiagInvNoiseCovLO(
            operator=brahmap.core.InvNoiseCovLO_Circulant,
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


class TestBlockDiagNoiseOps_Circulant_F32(BlockDiagNoiseOps_Circulant):
    dtype = np.float32
    rtol = 1.0e-4
    atol = 1.0e-6


class TestBlockDiagNoiseOps_Circulant_F64(BlockDiagNoiseOps_Circulant):
    dtype = np.float64
    rtol = 1.0e-6
    atol = 1.0e-10
