import numpy as np
from dataclasses import dataclass, asdict
import healpy as hp
import litebird_sim as lbs
from typing import List, Union

from brahmap.linop import DiagonalOperator

from brahmap.mapmakers import GLSParameters, GLSResult, compute_GLS_maps

from brahmap.interfaces import ToeplitzLO, BlockLO, InvNoiseCovLO_Uncorrelated

from brahmap.utilities import ProcessTimeSamples


@dataclass
class LBSimGLSParameters(GLSParameters):
    return_processed_samples: bool = False
    output_coordinate_system: lbs.CoordinateSystem = lbs.CoordinateSystem.Galactic


@dataclass
class LBSimGLSResult(GLSResult):
    nside: int
    coordinate_system: lbs.CoordinateSystem = lbs.CoordinateSystem.Galactic


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


def LBSim_compute_GLS_maps_from_obs(
    nside: int,
    obs: Union[lbs.Observation, List[lbs.Observation]],
    pointings_flag: Union[np.ndarray, List[np.ndarray], None] = None,
    inv_noise_cov_diagonal: Union[
        LBSim_InvNoiseCovLO_UnCorr, InvNoiseCovLO_Uncorrelated, None
    ] = None,
    threshold: float = 1.0e-5,
    dtype_float=None,
    LBSimGLSParameters: LBSimGLSParameters = LBSimGLSParameters(),
    component: str = "tod",
) -> Union[LBSimGLSResult, tuple[ProcessTimeSamples, LBSimGLSResult]]:
    if isinstance(obs, lbs.Observation):
        obs_list = [obs]
    else:
        obs_list = obs

    pointings = np.concatenate([ob.pointings for ob in obs_list]).reshape((-1, 2))
    pol_angles = np.concatenate(
        [ob.psi for ob in obs_list], axis=None
    )  # `axis=None` returns flattened arrays
    tod = np.concatenate([getattr(obs, component) for obs in obs_list], axis=None)

    lbsim_gls_result = LBSim_compute_GLS_maps(
        nside=nside,
        pointings=pointings,
        tod=tod,
        pointings_flag=pointings_flag,
        pol_angles=pol_angles,
        inv_noise_cov_operator=inv_noise_cov_diagonal,
        threshold=threshold,
        dtype_float=dtype_float,
        update_pointings_inplace=True,
        LBSimGLSParameters=LBSimGLSParameters,
    )

    if LBSimGLSParameters.return_processed_samples is True:
        processed_samples, lbsim_gls_result = lbsim_gls_result
        return processed_samples, lbsim_gls_result
    else:
        return lbsim_gls_result


def LBSim_compute_GLS_maps(
    nside: int,
    pointings: np.ndarray,
    tod: np.ndarray,
    pointings_flag: np.ndarray = None,
    pol_angles: np.ndarray = None,
    inv_noise_cov_operator: Union[
        ToeplitzLO, BlockLO, DiagonalOperator, InvNoiseCovLO_Uncorrelated, None
    ] = None,
    threshold: float = 1.0e-5,
    dtype_float=None,
    update_pointings_inplace: bool = True,
    LBSimGLSParameters: LBSimGLSParameters = LBSimGLSParameters(),
) -> Union[LBSimGLSResult, tuple[ProcessTimeSamples, LBSimGLSResult]]:
    npix = hp.nside2npix(nside)

    if LBSimGLSParameters.output_coordinate_system == lbs.CoordinateSystem.Galactic:
        pointings, pol_angles = lbs.coordinates.rotate_coordinates_e2g(
            pointings_ecl=pointings, pol_angle_ecl=pol_angles
        )

    pointings = hp.ang2pix(nside, pointings[:, 0], pointings[:, 1])

    temp_result = compute_GLS_maps(
        npix=npix,
        pointings=pointings,
        time_ordered_data=tod,
        pointings_flag=pointings_flag,
        pol_angles=pol_angles,
        inv_noise_cov_operator=inv_noise_cov_operator,
        threshold=threshold,
        dtype_float=dtype_float,
        update_pointings_inplace=update_pointings_inplace,
        GLSParameters=LBSimGLSParameters,
    )

    if LBSimGLSParameters.return_processed_samples is True:
        processed_samples, gls_result = temp_result
    else:
        gls_result = temp_result

    lbsim_gls_result = LBSimGLSResult(
        nside=nside,
        coordinate_system=LBSimGLSParameters.output_coordinate_system,
        **asdict(gls_result),
    )

    if LBSimGLSParameters.return_processed_samples is True:
        return processed_samples, lbsim_gls_result
    else:
        return lbsim_gls_result
