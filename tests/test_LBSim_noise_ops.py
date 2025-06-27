import tempfile
import pytest
import numpy as np

import brahmap

from mpi4py import MPI

comm_size_global = MPI.COMM_WORLD.size

pytestmark = pytest.mark.skipif(
    comm_size_global > 4,
    reason="These tests are not meant for more than 4 MPI processes",
)

litebird_sim = pytest.importorskip(
    modname="litebird_sim",
    minversion="0.13.0",
    reason="Couldn't import `litebird_sim` module",
)
import litebird_sim as lbs  # noqa: E402


class lbsim_simulation:
    def __init__(self):
        self.comm = lbs.MPI_COMM_WORLD
        tmp_dir = tempfile.TemporaryDirectory()

        self.sim = lbs.Simulation(
            base_path=tmp_dir.name,
            start_time=234,
            duration_s=71,  # Do not increase this number
            random_seed=65454,
            mpi_comm=self.comm,
        )

        self.detector_list = [
            lbs.DetectorInfo(name="det1", sampling_rate_hz=1, net_ukrts=1.0),
            lbs.DetectorInfo(name="det2", sampling_rate_hz=1, net_ukrts=2.0),
            lbs.DetectorInfo(name="det3", sampling_rate_hz=1, net_ukrts=3.0),
            lbs.DetectorInfo(name="det4", sampling_rate_hz=1, net_ukrts=4.0),
        ]

        ### Create observations
        comm_size = self.comm.Get_size()
        if comm_size == 2:
            n_block_det = 2
            n_block_time = 1
        elif comm_size == 4:
            n_block_det = 2
            n_block_time = 2
        else:
            n_block_det = 1
            n_block_time = self.comm.Get_size()

        self.sim.create_observations(
            detectors=self.detector_list,
            num_of_obs_per_detector=3,
            n_blocks_det=n_block_det,
            n_blocks_time=n_block_time,
            split_list_over_processes=False,
        )

        ### RNG used to generate noise properties
        seed = 545454
        self.rng = np.random.default_rng(seed=[seed, self.comm.rank])


lbs_sim = lbsim_simulation()


@pytest.mark.parametrize(
    "lbsim_obj",
    [(lbs_sim)],
)
@pytest.mark.ignore_param_count
class TestLBSim_InvNoiseCovLO_UnCorr:
    def test_LBSim_InvNoiseCov_UnCorr_explicit(self, lbsim_obj):
        """Here the noise variances are specified explicitly"""
        # Assigning the key values
        noise_variance = {
            "det1": 1.5,
            # "det2", # the operator should set it to 1.0 by default
            "det3": 2,
            "det4": 3,
        }

        inv_noise_variance_op = brahmap.lbsim.LBSim_InvNoiseCovLO_UnCorr(
            obs=lbsim_obj.sim.observations,
            noise_variance=noise_variance,
        )

        vec_length = np.sum([obs.tod.size for obs in lbsim_obj.sim.observations])
        vec = np.ones(vec_length)

        prod = inv_noise_variance_op * vec

        # filling the `test_tod` for each detector with its noise variance. Now the flatten `test_tod` should resemble the diagonal of `inv_noise_variance_op`
        for obs in lbsim_obj.sim.observations:
            obs.test_tod = np.empty_like(obs.tod)
            for idx in range(obs.n_detectors):
                obs.test_tod[idx].fill(noise_variance[obs.name[idx]])

        np.testing.assert_allclose(
            prod,
            inv_noise_variance_op.diag,
            rtol=1.0e-4,
            atol=1.0e-5,
        )

        np.testing.assert_allclose(
            inv_noise_variance_op.diag,
            np.concatenate(
                [1.0 / obs.test_tod for obs in lbsim_obj.sim.observations], axis=None
            ),
            rtol=1.0e-4,
            atol=1.0e-5,
        )

    def test_LBSim_InvNoiseCov_UnCorr_IMo(self, lbsim_obj):
        """Here the noise variances are automatically taken from IMo"""
        inv_noise_variance_op = brahmap.lbsim.LBSim_InvNoiseCovLO_UnCorr(
            obs=lbsim_obj.sim.observations,
            noise_variance=None,
        )

        vec_length = np.sum([obs.tod.size for obs in lbsim_obj.sim.observations])
        vec = np.ones(vec_length)

        prod = inv_noise_variance_op * vec

        noise_variance = dict(
            zip(
                lbsim_obj.sim.observations[0].name,
                lbs.mapmaking.common.get_map_making_weights(
                    lbsim_obj.sim.observations[0]
                ),
            )
        )

        # filling the `test_tod` for each detector with its noise variance. Now the flatten `test_tod` should resemble the diagonal of `inv_noise_variance_op`
        for obs in lbsim_obj.sim.observations:
            obs.test_tod = np.empty_like(obs.tod)
            for idx in range(obs.n_detectors):
                obs.test_tod[idx].fill(noise_variance[obs.name[idx]])

        np.testing.assert_allclose(
            prod,
            inv_noise_variance_op.diag,
            rtol=1.0e-4,
            atol=1.0e-5,
        )

        np.testing.assert_allclose(
            inv_noise_variance_op.diag,
            np.concatenate(
                [1.0e4 / obs.test_tod for obs in lbsim_obj.sim.observations], axis=None
            ),
            rtol=1.0e-4,
            atol=1.0e-5,
        )


@pytest.mark.parametrize(
    "lbsim_obj",
    [(lbs_sim)],
)
@pytest.mark.ignore_param_count
class TestLBSim_InvNoiseCovLO_Circulant:
    def test_LBSim_InvNoiseCov_Circulant_dict(self, lbsim_obj):
        """Here the noise covariance and power spectrum are supplied for each detector"""
        covariance_list = {}
        power_spec_list = {}

        for detector in lbsim_obj.detector_list:
            covariance = lbsim_obj.rng.random(
                size=lbsim_obj.sim.observations[0].n_samples
            )
            power_spec = np.fft.fft(covariance).real

            covariance = np.fft.ifft(power_spec).real
            power_spec = np.fft.fft(covariance).real

            covariance_list[detector.name] = covariance
            power_spec_list[detector.name] = power_spec

        lbsim_inv_cov1 = brahmap.LBSim_InvNoiseCovLO_Circulant(
            obs=lbsim_obj.sim.observations,
            input=covariance_list,
            input_type="covariance",
        )

        lbsim_inv_cov2 = brahmap.LBSim_InvNoiseCovLO_Circulant(
            obs=lbsim_obj.sim.observations,
            input=power_spec_list,
            input_type="power_spectrum",
        )

        row_list = []
        inv_cov_LO_list = []
        for obs in lbsim_obj.sim.observations:
            for __, detector in enumerate(obs.name):
                inv_cov_LO_list.append(
                    brahmap.InvNoiseCovLO_Circulant(
                        size=obs.n_samples,
                        input=covariance_list[detector][: obs.n_samples],
                        input_type="covariance",
                    )
                )
                row_list.append(obs.n_samples)

        np.testing.assert_equal(lbsim_inv_cov1.row_size, row_list)
        np.testing.assert_equal(lbsim_inv_cov2.col_size, row_list)
        np.testing.assert_equal(lbsim_inv_cov1.size, sum(row_list))

        inv_cov1_explicit = lbsim_inv_cov1.to_array()
        inv_cov2_explicit = lbsim_inv_cov2.to_array()

        start_row_idx = end_row_idx = 0
        start_col_idx = end_col_idx = 0
        full_inv_cov = np.zeros_like(inv_cov1_explicit)
        for idx in range(lbsim_inv_cov1.num_blocks):
            end_row_idx += row_list[idx]
            end_col_idx += row_list[idx]
            inv_cov_block = inv_cov_LO_list[idx]
            full_inv_cov[
                start_row_idx:end_row_idx, start_col_idx:end_col_idx
            ] = inv_cov_block.to_array()
            start_row_idx = end_row_idx
            start_col_idx = end_col_idx

        np.testing.assert_allclose(
            inv_cov1_explicit,
            full_inv_cov,
            rtol=1.0e-4,
            atol=1.0e-5,
        )
        np.testing.assert_allclose(
            inv_cov2_explicit,
            full_inv_cov,
            rtol=1.0e-4,
            atol=1.0e-5,
        )

    def test_LBSim_InvNoiseCov_Circulant_array(self, lbsim_obj):
        """Here only one common noise covariance and power spectrum is supplied"""

        covariance = lbsim_obj.rng.random(size=lbsim_obj.sim.observations[0].n_samples)
        power_spec = np.fft.fft(covariance).real

        covariance = np.fft.ifft(power_spec).real
        power_spec = np.fft.fft(covariance).real

        lbsim_inv_cov1 = brahmap.LBSim_InvNoiseCovLO_Circulant(
            obs=lbsim_obj.sim.observations,
            input=covariance,
            input_type="covariance",
        )

        lbsim_inv_cov2 = brahmap.LBSim_InvNoiseCovLO_Circulant(
            obs=lbsim_obj.sim.observations,
            input=power_spec,
            input_type="power_spectrum",
        )

        row_list = []
        inv_cov_LO_list = []
        for obs in lbsim_obj.sim.observations:
            for __, __ in enumerate(obs.name):
                inv_cov_LO_list.append(
                    brahmap.InvNoiseCovLO_Circulant(
                        size=obs.n_samples,
                        input=covariance[: obs.n_samples],
                        input_type="covariance",
                    )
                )
                row_list.append(obs.n_samples)

        np.testing.assert_equal(lbsim_inv_cov1.row_size, row_list)
        np.testing.assert_equal(lbsim_inv_cov2.col_size, row_list)
        np.testing.assert_equal(lbsim_inv_cov1.size, sum(row_list))

        inv_cov1_explicit = lbsim_inv_cov1.to_array()
        inv_cov2_explicit = lbsim_inv_cov2.to_array()

        start_row_idx = end_row_idx = 0
        start_col_idx = end_col_idx = 0
        full_inv_cov = np.zeros_like(inv_cov1_explicit)
        for idx in range(lbsim_inv_cov1.num_blocks):
            end_row_idx += row_list[idx]
            end_col_idx += row_list[idx]
            inv_cov_block = inv_cov_LO_list[idx]
            full_inv_cov[
                start_row_idx:end_row_idx, start_col_idx:end_col_idx
            ] = inv_cov_block.to_array()
            start_row_idx = end_row_idx
            start_col_idx = end_col_idx

        np.testing.assert_allclose(
            inv_cov1_explicit,
            full_inv_cov,
            rtol=1.0e-4,
            atol=1.0e-5,
        )
        np.testing.assert_allclose(
            inv_cov2_explicit,
            full_inv_cov,
            rtol=1.0e-4,
            atol=1.0e-5,
        )


@pytest.mark.parametrize(
    "lbsim_obj",
    [(lbs_sim)],
)
@pytest.mark.ignore_param_count
class TestLBSim_InvNoiseCovLO_Toeplitz:
    """Unlike previously, I am creating the inverse covariance operator but running the numerical tests on the covariance operator since it is faster"""

    def test_LBSim_InvNoiseCov_Toeplitz_dict(self, lbsim_obj):
        """Here the noise covariance and power spectrum are supplied for each detector"""
        covariance_list = {}
        power_spec_list = {}

        for detector in lbsim_obj.detector_list:
            covariance = lbsim_obj.rng.random(
                size=lbsim_obj.sim.observations[0].n_samples
            )

            extended_covariance = np.concatenate([covariance, covariance[1:-1][::-1]])
            power_spec = np.fft.fft(extended_covariance).real

            covariance_list[detector.name] = covariance
            power_spec_list[detector.name] = power_spec

        lbsim_cov1 = brahmap.LBSim_InvNoiseCovLO_Toeplitz(
            obs=lbsim_obj.sim.observations,
            input=covariance_list,
            input_type="covariance",
        ).get_inverse()

        lbsim_cov2 = brahmap.LBSim_InvNoiseCovLO_Toeplitz(
            obs=lbsim_obj.sim.observations,
            input=power_spec_list,
            input_type="power_spectrum",
        ).get_inverse()

        row_list = []
        cov_LO_list = []
        for obs in lbsim_obj.sim.observations:
            for __, detector in enumerate(obs.name):
                cov_LO_list.append(
                    brahmap.NoiseCovLO_Toeplitz01(
                        size=obs.n_samples,
                        input=covariance_list[detector][: obs.n_samples],
                        input_type="covariance",
                    )
                )
                row_list.append(obs.n_samples)

        np.testing.assert_equal(lbsim_cov1.row_size, row_list)
        np.testing.assert_equal(lbsim_cov2.col_size, row_list)
        np.testing.assert_equal(lbsim_cov1.size, sum(row_list))

        cov1_explicit = lbsim_cov1.to_array()
        cov2_explicit = lbsim_cov2.to_array()

        start_row_idx = end_row_idx = 0
        start_col_idx = end_col_idx = 0
        full_cov = np.zeros_like(cov1_explicit)
        for idx in range(lbsim_cov1.num_blocks):
            end_row_idx += row_list[idx]
            end_col_idx += row_list[idx]
            cov_block = cov_LO_list[idx]
            full_cov[
                start_row_idx:end_row_idx, start_col_idx:end_col_idx
            ] = cov_block.to_array()
            start_row_idx = end_row_idx
            start_col_idx = end_col_idx

        np.testing.assert_allclose(
            cov1_explicit,
            full_cov,
            rtol=1.0e-4,
            atol=1.0e-5,
        )
        np.testing.assert_allclose(
            cov2_explicit,
            full_cov,
            rtol=1.0e-4,
            atol=1.0e-5,
        )

    def test_LBSim_InvNoiseCov_Toeplitz_array(self, lbsim_obj):
        """Here only one common noise covariance and power spectrum is supplied"""

        covariance = lbsim_obj.rng.random(size=lbsim_obj.sim.observations[0].n_samples)

        extended_covariance = np.concatenate([covariance, covariance[1:-1][::-1]])
        power_spec = np.fft.fft(extended_covariance).real

        lbsim_cov1 = brahmap.LBSim_InvNoiseCovLO_Toeplitz(
            obs=lbsim_obj.sim.observations,
            input=covariance,
            input_type="covariance",
        ).get_inverse()

        lbsim_cov2 = brahmap.LBSim_InvNoiseCovLO_Toeplitz(
            obs=lbsim_obj.sim.observations,
            input=power_spec,
            input_type="power_spectrum",
        ).get_inverse()

        row_list = []
        cov_LO_list = []
        for obs in lbsim_obj.sim.observations:
            for __, __ in enumerate(obs.name):
                cov_LO_list.append(
                    brahmap.NoiseCovLO_Toeplitz01(
                        size=obs.n_samples,
                        input=covariance[: obs.n_samples],
                        input_type="covariance",
                    )
                )
                row_list.append(obs.n_samples)

        np.testing.assert_equal(lbsim_cov1.row_size, row_list)
        np.testing.assert_equal(lbsim_cov2.col_size, row_list)
        np.testing.assert_equal(lbsim_cov1.size, sum(row_list))

        cov1_explicit = lbsim_cov1.to_array()
        cov2_explicit = lbsim_cov2.to_array()

        start_row_idx = end_row_idx = 0
        start_col_idx = end_col_idx = 0
        full_cov = np.zeros_like(cov1_explicit)
        for idx in range(lbsim_cov1.num_blocks):
            end_row_idx += row_list[idx]
            end_col_idx += row_list[idx]
            cov_block = cov_LO_list[idx]
            full_cov[
                start_row_idx:end_row_idx, start_col_idx:end_col_idx
            ] = cov_block.to_array()
            start_row_idx = end_row_idx
            start_col_idx = end_col_idx

        np.testing.assert_allclose(
            cov1_explicit,
            full_cov,
            rtol=1.0e-4,
            atol=1.0e-5,
        )
        np.testing.assert_allclose(
            cov2_explicit,
            full_cov,
            rtol=1.0e-4,
            atol=1.0e-5,
        )
