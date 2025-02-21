import numpy as np
import warnings
from dataclasses import dataclass, asdict
import healpy as hp
import litebird_sim as lbs
from typing import List, Union

from brahmap.utilities import LowerTypeCastWarning

from brahmap.linop import DiagonalOperator

from brahmap.mapmakers import (
    GLSParameters,
    GLSResult,
    compute_GLS_maps_from_PTS,
)

from brahmap.interfaces import ToeplitzLO, BlockLO, InvNoiseCovLO_Uncorrelated

from brahmap.utilities import ProcessTimeSamples, SolverType

import brahmap


@dataclass
class LBSimGLSParameters(GLSParameters):
    return_processed_samples: bool = False
    output_coordinate_system: lbs.CoordinateSystem = lbs.CoordinateSystem.Galactic


@dataclass
class LBSimGLSResult(GLSResult):
    nside: int
    coordinate_system: lbs.CoordinateSystem


class LBSim_InvNoiseCovLO_UnCorr(InvNoiseCovLO_Uncorrelated):
    """This operator class accepts `inverse_noise_variance` as a dictionary of the inverse of noise variance associated with detector names. It will then arrange the blocks of inverse noise variance in the same way as tods in the `obs_list` are distributed"""

    def __init__(
        self,
        obs: Union[lbs.Observation, List[lbs.Observation]],
        inverse_noise_variance: Union[dict, None] = None,
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

        if inverse_noise_variance is not None:
            noise_dict_keys = list(inverse_noise_variance.keys())

            # list of the name of the detectors available on the current rank
            det_list = list(obs_list[0].name)

            # setting the `inverse_noise_variance`` to 1 for the detectors whose inverse noise variance is not provided in the dictionary
            det_no_variance = np.setdiff1d(det_list, noise_dict_keys)
            for detector in det_no_variance:
                inverse_noise_variance[detector] = 1.0

            # populating the diagonal array of the local diagonal operator
            start_idx = 0
            for obs_idx in range(n_obs):
                for det_list_idx, detector in enumerate(det_list):
                    end_idx = start_idx + tod_size[det_list_idx, obs_idx]

                    # filling a range of the diagonal array with the inverse noise variance value
                    diagonal[start_idx:end_idx].fill(inverse_noise_variance[detector])
                    start_idx = end_idx

        super(LBSim_InvNoiseCovLO_UnCorr, self).__init__(diag=diagonal, dtype=dtype)


class LBSimProcessTimeSamples(ProcessTimeSamples):
    def __init__(
        self,
        nside: int,
        observations: Union[lbs.Observation, List[lbs.Observation]],
        pointings_flag: Union[np.ndarray, None] = None,
        solver_type: SolverType = SolverType.IQU,
        noise_weights: Union[np.ndarray, None] = None,
        output_coordinate_system: lbs.CoordinateSystem = lbs.CoordinateSystem.Galactic,
        threshold: float = 1.0e-5,
        dtype_float=np.float64,
    ):
        if brahmap.bMPI is None:
            brahmap.Initialize()

        self.__nside = nside
        npix = hp.nside2npix(self.nside)

        if isinstance(observations, lbs.Observation):
            self.__obs_list = [observations]
        else:
            self.__obs_list = observations

        pix_indices = np.empty(
            (
                len(self.obs_list),
                self.obs_list[0].n_detectors,
                self.obs_list[0].n_samples,
            ),
            dtype=int,
        )
        pol_angles = np.empty(
            (
                len(self.obs_list),
                self.obs_list[0].n_detectors,
                self.obs_list[0].n_samples,
            ),
            dtype=dtype_float,
        )

        for obs_idx, obs in enumerate(self.obs_list):
            for det_idx in range(obs.n_detectors):
                if hasattr(obs, "pointing_matrix"):
                    cur_pointings = obs.pointing_matrix[det_idx]
                    hwp_angle = getattr(obs, "hwp_angle", None)

                    if not np.can_cast(obs.pointing_matrix.dtype, dtype_float):
                        warnings.warn(
                            f"`obs.pointing_matrix` has been casted from higher dtype to lower one. You might want to call `LBSimProcessTimeSamples` with `dtype_float={obs.pointing_matrix.dtype}`",
                            LowerTypeCastWarning,
                        )

                    if hwp_angle is not None and not np.can_cast(
                        hwp_angle.dtype, dtype_float
                    ):
                        warnings.warn(
                            f"`obs.hwp_angle` has been casted from higher dtype to lower one. You might want to call `LBSimProcessTimeSamples` with `dtype_float={hwp_angle.dtype}`",
                            LowerTypeCastWarning,
                        )

                else:
                    cur_pointings, hwp_angle = obs.get_pointings(
                        detector_idx=det_idx, pointings_dtype=dtype_float
                    )

                self.__coordinate_system = output_coordinate_system
                if self.coordinate_system == lbs.CoordinateSystem.Galactic:
                    cur_pointings = lbs.coordinates.rotate_coordinates_e2g(
                        cur_pointings
                    )

                if hwp_angle is None:
                    pol_angles[obs_idx, det_idx] = (
                        obs.pol_angle_rad[det_idx] + cur_pointings[:, 2]
                    )
                else:
                    pol_angles[obs_idx, det_idx] = (
                        2.0 * hwp_angle
                        - obs.pol_angle_rad[det_idx]
                        + cur_pointings[:, 2]
                    )

                pix_indices[obs_idx, det_idx] = hp.ang2pix(
                    nside, cur_pointings[:, 0], cur_pointings[:, 1]
                )

            del cur_pointings, hwp_angle

        pix_indices = np.concatenate(pix_indices, axis=None)
        pol_angles = np.concatenate(pol_angles, axis=None)

        super().__init__(
            npix=npix,
            pointings=pix_indices,
            pointings_flag=pointings_flag,
            solver_type=solver_type,
            pol_angles=pol_angles,
            noise_weights=noise_weights,
            threshold=threshold,
            dtype_float=dtype_float,
            update_pointings_inplace=True,
        )

    @property
    def obs_list(self):
        return self.__obs_list

    @property
    def nside(self):
        return self.__nside

    @property
    def coordinate_system(self):
        return self.__coordinate_system


def LBSim_compute_GLS_maps(
    nside: int,
    observations: Union[lbs.Observation, List[lbs.Observation]],
    component: str = "tod",
    pointings_flag: Union[np.ndarray, None] = None,
    inv_noise_cov_operator: Union[
        ToeplitzLO,
        BlockLO,
        DiagonalOperator,
        LBSim_InvNoiseCovLO_UnCorr,
        InvNoiseCovLO_Uncorrelated,
        None,
    ] = None,
    threshold: float = 1.0e-5,
    dtype_float=None,
    LBSim_gls_parameters: LBSimGLSParameters = LBSimGLSParameters(),
) -> Union[LBSimGLSResult, tuple[LBSimProcessTimeSamples, LBSimGLSResult]]:
    if inv_noise_cov_operator is None:
        noise_weights = None
    else:
        noise_weights = inv_noise_cov_operator.diag

    processed_samples = LBSimProcessTimeSamples(
        nside=nside,
        observations=observations,
        pointings_flag=pointings_flag,
        solver_type=LBSim_gls_parameters.solver_type,
        noise_weights=noise_weights,
        output_coordinate_system=LBSim_gls_parameters.output_coordinate_system,
        threshold=threshold,
        dtype_float=dtype_float,
    )

    time_ordered_data = np.concatenate(
        [getattr(obs, component) for obs in observations], axis=None
    )

    gls_result = compute_GLS_maps_from_PTS(
        processed_samples=processed_samples,
        time_ordered_data=time_ordered_data,
        inv_noise_cov_operator=inv_noise_cov_operator,
        gls_parameters=LBSim_gls_parameters,
    )

    gls_result = LBSimGLSResult(
        nside=nside,
        coordinate_system=LBSim_gls_parameters.output_coordinate_system,
        **asdict(gls_result),
    )

    if LBSim_gls_parameters.return_processed_samples is True:
        return processed_samples, gls_result
    else:
        return gls_result
