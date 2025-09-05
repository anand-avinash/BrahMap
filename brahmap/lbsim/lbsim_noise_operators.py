from typing import List, Union, Literal, Dict, Any
import numbers

import numpy as np
import litebird_sim as lbs

from ..base import InvNoiseCovLinearOperator

from ..core import (
    InvNoiseCovLO_Diagonal,
    InvNoiseCovLO_Circulant,
    InvNoiseCovLO_Toeplitz01,
    BlockDiagInvNoiseCovLO,
)

from ..math import DTypeFloat

from ..mpi import MPI_RAISE_EXCEPTION


class LBSim_InvNoiseCovLO_UnCorr(BlockDiagInvNoiseCovLO):
    """_summary_

    The assumption is that at a given MPI process, all observations
    contain same set of detectors

    Parameters
    ----------
    obs : Union[lbs.Observation, List[lbs.Observation]]
        _description_
    noise_variance : Union[dict, DTypeFloat, None], optional
        _description_, by default None
    dtype : DTypeFloat, optional
        _description_, by default np.float64
    """

    # Keep a note of the hard-coded factor of 1e4

    def __init__(
        self,
        obs: Union[lbs.Observation, List[lbs.Observation]],
        noise_variance: Union[dict, DTypeFloat, None] = None,
        dtype: DTypeFloat = np.float64,
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
        elif isinstance(noise_variance, numbers.Number):
            noise_variance = dict(
                zip(
                    obs_list[0].name,
                    [noise_variance] * len(obs_list[0].name),
                )
            )

        # setting the `noise_variance` to 1 for the detectors whose noise variance is not provided in the dictionary
        det_no_variance = np.setdiff1d(obs_list[0].name, list(noise_variance.keys()))
        for detector in det_no_variance:
            noise_variance[detector] = 1.0

        block_size = []

        if len(set(noise_variance.values())) == 1:
            # That is, when all values in noise variance is the same
            block_input = {}
            for obs in obs_list:
                for det_idx in range(obs.n_detectors):
                    block_size.append(obs.n_samples)
                    if obs.n_samples not in block_input.keys():
                        block_input[obs.n_samples] = noise_variance[obs.name[0]]
        else:
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
    """_summary_

    Parameters
    ----------
    obs : Union[lbs.Observation, List[lbs.Observation]]
        _description_
    input : Union[dict, Union[np.ndarray, List]]
        _description_
    input_type : Literal["covariance", "power_spectrum"], optional
        _description_, by default "power_spectrum"
    dtype : DTypeFloat, optional
        _description_, by default np.float64
    """

    def __init__(
        self,
        obs: Union[lbs.Observation, List[lbs.Observation]],
        input: Union[dict, Union[np.ndarray, List]],
        input_type: Literal["covariance", "power_spectrum"] = "power_spectrum",
        dtype: DTypeFloat = np.float64,
    ):
        if isinstance(obs, lbs.Observation):
            obs_list = [obs]
        else:
            obs_list = obs

        block_size = []

        if isinstance(input, dict):
            block_input = []

            for obs in obs_list:
                # if input is a dict
                for det_idx in range(obs.n_detectors):
                    block_size.append(obs.n_samples)

                    resized_input = self.__resize_input(
                        new_size=obs.n_samples,
                        input=input[obs.name[det_idx]],
                        input_type=input_type,
                        dtype=dtype,
                    )

                    block_input.append(resized_input)

        elif isinstance(input, (np.ndarray, list)):
            block_input = {}

            for obs in obs_list:
                for det_idx in range(obs.n_detectors):
                    # if input is an array or a list, it will be taken as same for all the detectors available in the observation
                    block_size.append(obs.n_samples)

                    if obs.n_samples not in block_input.keys():
                        resized_input = self.__resize_input(
                            new_size=obs.n_samples,
                            input=input,
                            input_type=input_type,
                            dtype=dtype,
                        )

                        block_input[obs.n_samples] = resized_input
        else:
            MPI_RAISE_EXCEPTION(
                condition=True,
                exception=ValueError,
                message="The input must be an array or a list or a dictionary that maps detector names to their covariance/power spectrum",
            )

        super(LBSim_InvNoiseCovLO_Circulant, self).__init__(
            InvNoiseCovLO_Circulant,
            block_size=block_size,
            block_input=block_input,
            input_type=input_type,
            dtype=dtype,
        )

    def __resize_input(self, new_size, input, input_type, dtype):
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
    """_summary_

    Note that the observation length is either n or n-1.

    Parameters
    ----------
    obs : Union[lbs.Observation, List[lbs.Observation]]
        _description_
    input : Union[dict, Union[np.ndarray, List]]
        _description_
    input_type : Literal["covariance", "power_spectrum"], optional
        _description_, by default "power_spectrum"
    operator : InvNoiseCovLinearOperator, optional
        _description_, by default InvNoiseCovLO_Toeplitz01
    dtype : DTypeFloat, optional
        _description_, by default np.float64
    extra_kwargs : Dict[str, Any], optional
        _description_, by default {}
    """

    def __init__(
        self,
        obs: Union[lbs.Observation, List[lbs.Observation]],
        input: Union[dict, Union[np.ndarray, List]],
        input_type: Literal["covariance", "power_spectrum"] = "power_spectrum",
        operator: InvNoiseCovLinearOperator = InvNoiseCovLO_Toeplitz01,
        dtype: DTypeFloat = np.float64,
        extra_kwargs: Dict[str, Any] = {},
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
            extra_kwargs=extra_kwargs,
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
