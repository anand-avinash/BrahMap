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
    def __init__(self, nside):
        self.comm = lbs.MPI_COMM_WORLD
        tmp_dir = tempfile.TemporaryDirectory()

        self.sim = lbs.Simulation(
            base_path=tmp_dir.name,
            start_time=234,
            duration_s=351,
            random_seed=65454,
            mpi_comm=self.comm,
        )

        detector_list = [
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
            detectors=detector_list,
            num_of_obs_per_detector=3,
            n_blocks_det=n_block_det,
            n_blocks_time=n_block_time,
            split_list_over_processes=False,
        )


lbs_sim = lbsim_simulation(nside=16)


@pytest.mark.parametrize(
    "lbsim_obj",
    [(lbs_sim)],
)
class TestLBSim_InvNoiseCovLO_UnCorr:
    def test_LBSim_InvNoiseCov_UnCorr_explicit(self, lbsim_obj):
        """Here the noise variances are specified explicitly"""
        # Assigning the key values same as their `det_idx`
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

        # filling the `test_tod` for each detector with its `det_idx`. Now the flatten `test_tod` should resemble the diagonal of `inv_noise_variance_op`
        for obs in lbsim_obj.sim.observations:
            obs.test_tod = np.empty_like(obs.tod)
            for idx in range(obs.n_detectors):
                obs.test_tod[idx].fill(noise_variance[obs.name[idx]])

        np.testing.assert_allclose(prod, inv_noise_variance_op.diag)

        np.testing.assert_allclose(
            inv_noise_variance_op.diag,
            np.concatenate(
                [1.0 / obs.test_tod for obs in lbsim_obj.sim.observations], axis=None
            ),
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

        # filling the `test_tod` for each detector with its `det_idx`. Now the flatten `test_tod` should resemble the diagonal of `inv_noise_variance_op`
        for obs in lbsim_obj.sim.observations:
            obs.test_tod = np.empty_like(obs.tod)
            for idx in range(obs.n_detectors):
                obs.test_tod[idx].fill(noise_variance[obs.name[idx]])

        np.testing.assert_allclose(prod, inv_noise_variance_op.diag)

        np.testing.assert_allclose(
            inv_noise_variance_op.diag,
            np.concatenate(
                [1.0e4 / obs.test_tod for obs in lbsim_obj.sim.observations], axis=None
            ),
        )
