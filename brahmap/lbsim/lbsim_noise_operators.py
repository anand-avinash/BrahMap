from typing import List, Union

import numpy as np
import litebird_sim as lbs

from ..core import InvNoiseCovLO_Diagonal


class LBSim_InvNoiseCovLO_UnCorr(InvNoiseCovLO_Diagonal):
    """This operator class accepts `noise_variance` as a dictionary of the noise variance associated with detector names. It will then arrange the blocks of inverse noise variance in the same way as tods in the `obs_list` are distributed"""

    def __init__(
        self,
        obs: Union[lbs.Observation, List[lbs.Observation]],
        noise_variance: Union[dict, None] = None,
        dtype=None,
    ):
        if isinstance(obs, lbs.Observation):
            obs_list = [obs]
        else:
            obs_list = obs

        if dtype is None:
            dtype = np.float64

        # number of observations
        n_obs = len(obs_list)

        # number of detectors per observation
        n_dets_per_obs = len(
            obs_list[0].det_idx
        )  # assuming all obs on a proc are associated to same set of detectors

        # local length of tod for a given detector in a given observation
        tod_size = np.zeros([n_dets_per_obs, n_obs], dtype=int)

        # populating tod_size
        for obs_idx, obs in enumerate(obs_list):
            _, _, tod_size[:, obs_idx] = obs._get_local_start_time_start_and_n_samples()

        # length of local diagonal operator
        diagonal_len = tod_size.sum()

        # diagonal array of the local diagonal operator
        diagonal = np.ones(diagonal_len, dtype=dtype)

        if noise_variance is not None:
            noise_dict_keys = list(noise_variance.keys())

            # list of the name of the detectors available on the current rank
            det_list = list(obs_list[0].name)

            # setting the `noise_variance`` to 1 for the detectors whose noise variance is not provided in the dictionary
            det_no_variance = np.setdiff1d(det_list, noise_dict_keys)
            for detector in det_no_variance:
                noise_variance[detector] = 1.0

            # populating the diagonal array of the local diagonal operator
            start_idx = 0
            for obs_idx in range(n_obs):
                for det_list_idx, detector in enumerate(det_list):
                    end_idx = start_idx + tod_size[det_list_idx, obs_idx]

                    # filling a range of the diagonal array with the noise variance value
                    diagonal[start_idx:end_idx].fill(noise_variance[detector])
                    start_idx = end_idx

        super(LBSim_InvNoiseCovLO_UnCorr, self).__init__(
            size=diagonal_len, input=diagonal, dtype=dtype
        )
