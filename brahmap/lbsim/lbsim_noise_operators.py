from typing import List, Union

import numpy as np
import litebird_sim as lbs

from ..core import (
    InvNoiseCovLO_Diagonal,
    InvNoiseCovLO_Circulant,
    BlockDiagInvNoiseCovLO,
)


class LBSim_InvNoiseCovLO_UnCorr(BlockDiagInvNoiseCovLO):
    """The assumption is that at a given MPI process, all observations
    contain same set of detectors"""

    def __init__(
        self,
        obs: Union[lbs.Observation, List[lbs.Observation]],
        noise_variance: Union[dict, None] = None,
        dtype=np.float64,
    ):
        if isinstance(obs, lbs.Observation):
            obs_list = [obs]
        else:
            obs_list = obs

        if noise_variance is None:
            noise_variance = dict(
                zip(
                    obs_list[0].name,
                    lbs.mapmaking.common.get_map_making_weights(obs_list[0]),
                )
            )

        # setting the `noise_variance` to 1 for the detectors whose noise variance is not provided in the dictionary
        det_no_variance = np.setdiff1d(obs_list[0].name, list(noise_variance.keys()))
        for detector in det_no_variance:
            noise_variance[detector] = 1.0

        block_size = []
        block_input = []

        for obs in obs_list:
            for det_idx in range(obs.n_detectors):
                block_size.append(obs.n_samples)
                block_input.append(noise_variance[obs.name[det_idx]])

        super(LBSim_InvNoiseCovLO_UnCorr, self).__init__(
            InvNoiseCovLO_Diagonal,
            block_size=block_size,
            block_input=block_input,
            input_type="covariance",
            dtype=dtype,
        )


class LBSim_InvNoiseCovLO_Circulant(BlockDiagInvNoiseCovLO):
    def __init__(
        self,
        obs: Union[lbs.Observation, List[lbs.Observation]],
        input: dict,
        input_type: str = "power_spectrum",
        dtype=np.float64,
    ):
        if isinstance(obs, lbs.Observation):
            obs_list = [obs]
        else:
            obs_list = obs

        block_size = []
        block_input = []

        for obs in obs_list:
            for det_idx in range(obs.n_detectors):
                block_size.append(obs.n_samples)
                block_input.append(input[obs.name[det_idx]])

        super(LBSim_InvNoiseCovLO_Circulant, self).__init__(
            InvNoiseCovLO_Circulant,
            block_size=block_size,
            block_input=block_input,
            input_type=input_type,
            dtype=dtype,
        )
