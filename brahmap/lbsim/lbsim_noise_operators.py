from typing import List, Union, Literal

import numpy as np
import litebird_sim as lbs

from ..core import (
    InvNoiseCovLO_Diagonal,
    InvNoiseCovLO_Circulant,
    InvNoiseCovLO_Toeplitz01,
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
        input_type: Literal["covariance", "power_spectrum"] = "power_spectrum",
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
                    resized_input = self._resize_input(
                        new_size=obs.n_samples,
                        input=input[obs.name[det_idx]],
                        input_type=input_type,
                        dtype=dtype,
                    )
                    block_input.append(resized_input)
            else:
                for det_idx in range(obs.n_detectors):
                    # if input is an array or a list, it will be taken as same for all the detectors available in the observation
                    block_size.append(obs.n_samples)
                    resized_input = self._resize_input(
                        new_size=obs.n_samples,
                        input=input,
                        input_type=input_type,
                        dtype=dtype,
                    )
                    block_input.append(resized_input)

        super(LBSim_InvNoiseCovLO_Circulant, self).__init__(
            InvNoiseCovLO_Circulant,
            block_size=block_size,
            block_input=block_input,
            input_type=input_type,
            dtype=dtype,
        )

    def _resize_input(self, new_size, input, input_type, dtype):
        if input_type == "covariance":
            # if the size of the returned array is smaller than new_size, it
            # will be captured by the InvNoiseCovLO_Circulant class
            # automatically

            # Slicing the input array here is probably not the best choice as
            # it breaks the symmetry of the covariance and renders it
            # non-circulant. Same goes for slicing the covariance computed
            # through power spectrum. The best solution would be to create per
            # observation, per detector operators independently and supply them
            # to `BlockDiagInvNoiseCovLO`
            return input[:new_size]
        elif input_type == "power_spectrum":
            input_size = len(input)
            if input_size > new_size:
                new_input = np.fft.ifft(input)[:new_size]  # new covariance
                new_input = np.fft.fft(new_input).real.astype(
                    dtype=dtype,
                    copy=False,
                )  # new ps
                return new_input
            else:
                # If input size is equal to expected size, it will be fine.
                # If it is smaller, InvNoiseCovLO_Circulant class will
                # throw an error automatically
                return input


class LBSim_InvNoiseCovLO_Toeplitz(BlockDiagInvNoiseCovLO):
    """Note that the observation length is either n or n-1."""

    def __init__(
        self,
        obs: Union[lbs.Observation, List[lbs.Observation]],
        input: Union[dict, Union[np.ndarray, List]],
        input_type: Literal["covariance", "power_spectrum"] = "power_spectrum",
        operator=InvNoiseCovLO_Toeplitz01,
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
                    resized_input = self._resize_input(
                        new_size=obs.n_samples,
                        input=input[obs.name[det_idx]],
                        input_type=input_type,
                        dtype=dtype,
                    )
                    block_input.append(resized_input)
            else:
                # if input is an array or a list, it will be taken as same for all the detectors available in the observation
                for det_idx in range(obs.n_detectors):
                    block_size.append(obs.n_samples)
                    resized_input = self._resize_input(
                        new_size=obs.n_samples,
                        input=input,
                        input_type=input_type,
                        dtype=dtype,
                    )
                    block_input.append(resized_input)

        super(LBSim_InvNoiseCovLO_Toeplitz, self).__init__(
            operator,
            block_size=block_size,
            block_input=block_input,
            input_type=input_type,
            dtype=dtype,
        )

    def _resize_input(self, new_size, input, input_type, dtype):
        if input_type == "covariance":
            # if the size of the returned array is smaller than new_size, it
            # will be captured by the InvNoiseCovLO_Toeplitz0x class
            # automatically
            return input[:new_size]
        elif input_type == "power_spectrum":
            input_size = len(input)
            ex_size1 = 2 * new_size - 1  # expected size of ps array (2n-1)
            ex_size2 = 2 * new_size - 2  # expected size of ps array (2n-2)
            if input_size > ex_size2 and input_size > ex_size1:
                new_input = np.fft.ifft(input)[
                    :new_size
                ]  # covariance of size `new_size`
                new_input = np.concatenate(
                    [new_input, new_input[1:-1][::-1]]
                )  # full covariance of size `2*new_size - 2`
                new_input = np.fft.fft(new_input).real.astype(
                    dtype=dtype, copy=False
                )  # full ps of size `2*new_size - 2`
                return new_input
            else:
                # If input size is equal to expected size, it will be fine.
                # If it is smaller, InvNoiseCovLO_Toeplitz0x class will
                # throw an error automatically
                return input
