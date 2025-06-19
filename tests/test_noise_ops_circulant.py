import numpy as np
import brahmap

from template_noise_ops_tests import BaseTestNoiseLO


class NoiseOps_Circulant(BaseTestNoiseLO):
    def setup_method(self):
        dtype = getattr(self, "dtype")
        rtol = getattr(self, "rtol")
        atol = getattr(self, "atol")

        seed = 3432
        comm_rank = brahmap.MPI_UTILS.rank
        rng = np.random.default_rng(seed=[seed, comm_rank])

        size = 3 * (comm_rank + 1)

        covariance = rng.random(size, dtype=dtype)
        power_spec = np.fft.fft(covariance).real.astype(dtype=dtype)

        covariance = np.fft.ifft(power_spec).real.astype(dtype=dtype)
        power_spec = np.fft.fft(covariance).real.astype(dtype=dtype)

        operator1 = brahmap.core.NoiseCovLO_Circulant(
            size=size, input=covariance, input_type="covariance", dtype=dtype
        )
        operator2 = brahmap.core.NoiseCovLO_Circulant(
            size=size, input=power_spec, input_type="power_spectrum", dtype=dtype
        )
        numpy_operator1 = np.zeros(shape=(size, size))
        for i in range(size):
            for j in range(size):
                numpy_operator1[i, j] = covariance[(i - j) % size]

        inv_operator1 = brahmap.core.InvNoiseCovLO_Circulant(
            size=size, input=covariance, input_type="covariance", dtype=dtype
        )
        inv_operator2 = brahmap.core.InvNoiseCovLO_Circulant(
            size=size, input=power_spec, input_type="power_spectrum", dtype=dtype
        )
        numpy_inv_operator1 = np.linalg.inv(numpy_operator1)

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


class TestNoiseOps_Circulant_F32(NoiseOps_Circulant):
    dtype = np.float32
    rtol = 1.0e-4
    atol = 1.0e-5


class TestNoiseOps_Circulant_F64(NoiseOps_Circulant):
    dtype = np.float64
    rtol = 1.0e-6
    atol = 1.0e-10
