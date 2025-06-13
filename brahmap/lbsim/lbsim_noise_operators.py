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

    # Keep a note of the hard-coded factor of 1e4

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
                    lbs.mapmaking.common.get_map_making_weights(obs_list[0]) / 1.0e4,
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
        input: Union[dict, Union[np.ndarray, List]],
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
            if isinstance(input, dict):
                # if input is a dict
                for det_idx in range(obs.n_detectors):
                    block_size.append(obs.n_samples)
                    if obs.n_samples < len(input[obs.name[det_idx]]):
                        resized_input = self._resize_power_spectrum(
                            new_size=obs.n_samples,
                            input=input[obs.name[det_idx]],
                            input_type=input_type,
                            dtype=dtype,
                        )
                        block_input.append(resized_input)
                    else:
                        block_input.append(input[obs.name[det_idx]])
            else:
                # if input is an array or a list, it will be taken as same for all the detectors available in the observation
                for det_idx in range(obs.n_detectors):
                    block_size.append(obs.n_samples)
                    if obs.n_samples < len(input):
                        resized_input = self._resize_power_spectrum(
                            new_size=obs.n_samples,
                            input=input,
                            input_type=input_type,
                            dtype=dtype,
                        )
                        block_input.append(resized_input)
                    else:
                        block_input.append(input)

        super(LBSim_InvNoiseCovLO_Circulant, self).__init__(
            InvNoiseCovLO_Circulant,
            block_size=block_size,
            block_input=block_input,
            input_type=input_type,
            dtype=dtype,
        )

    def _resize_power_spectrum(self, new_size, input, input_type, dtype):
        if input_type == "covariance":
            input = input[:new_size]
            return input
        elif input_type == "power_spectrum":
            input = np.fft.ifft(input)[:new_size]
            input = np.fft.fft(input).real.astype(dtype=dtype, copy=False)
            return input
