import numpy as np
from dataclasses import dataclass, asdict
import healpy as hp
import litebird_sim as lbs
from typing import List, Union

from brahmap import MPI_RAISE_EXCEPTION

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


def sample_counts(obs_list):
    """Returns an array of size `n_obs x n_dets` containing the size of TOD for each detector in each observation.

    Args:
        obs_list (_type_): List of observation

    Returns:
        np.ndarray: sample count array
    """
    ndet_global = obs_list[0]._n_detectors_global
    count_arr = np.zeros((len(obs_list), ndet_global), dtype=int)
    for obs_idx, obs in enumerate(obs_list):
        for idx in range(ndet_global):
            # Assuming that the length of tod for each detector is same
            count_arr[obs_idx, idx] = obs.tod[0].shape[0]
    return count_arr


def start_idx(arr, obs_idx, det_idx):
    """Returns the starting index of TOD of detector `det_idx` in observation `obs_idx` in flattened array of TODs concatenated over observations. That is, the starting index of TOD of given det and obs index in a flat array given by `np.concatenate([getattr(obs, component) for obs in obs_list], axis=None)`. `arr` must be the output of `sample_count()` function."""
    idx = 0
    for i in range(obs_idx):
        idx += sum(arr[i][:])
    idx += sum(arr[obs_idx][:det_idx])
    return idx


def end_idx(arr, obs_idx, det_idx):
    """Similar to `start_idx()` function, but this one returns the ending index"""
    idx = start_idx(arr, obs_idx, det_idx)
    idx += arr[obs_idx, det_idx]
    return idx


def det_sample_count_idx(arr, obs_idx, det_idx):
    """
    This function returns the starting index of the TOD of a given detector for a given `obs_idx`
    """
    return sum(arr[:, det_idx][:obs_idx])


class LBSim_InvNoiseCovLO_UnCorr(InvNoiseCovLO_Uncorrelated):
    """Here the `noise_variance` must be a dictionary of noise variance associated with detector names. This operator will arrange the blocks of noise variance in the same way as tods in the obs_list are distributed."""

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

        tod_counts = sample_counts(obs_list)
        diag_len = tod_counts.sum()
        tod_len = tod_counts.sum(axis=0)
        det_list = list(obs_list[0].name)

        if noise_variance is None:
            diagonal = np.ones(diag_len, dtype=(np.float64 if dtype is None else dtype))
        else:
            noise_dict_keys = list(noise_variance.keys())
            if dtype is None:
                dtype = noise_variance[noise_dict_keys[0]].dtype
            for detector in det_list:
                if detector not in noise_dict_keys:
                    idx = det_list.index(detector)
                    noise_variance[detector] = np.ones(tod_len[idx])

                MPI_RAISE_EXCEPTION(
                    condition=(
                        len(noise_variance[detector])
                        != tod_len[det_list.index(detector)]
                    ),
                    exception=ValueError,
                    message=f"Incorrect length of noise variance for detector {detector}",
                )

            diagonal = np.empty(diag_len, dtype=dtype)
            for obs_idx, obs in enumerate(obs_list):
                for det_idx in obs.det_idx:
                    stdiagidx = start_idx(tod_counts, obs_idx, det_idx)
                    endiagidx = end_idx(tod_counts, obs_idx, det_idx)
                    sttodidx = det_sample_count_idx(tod_counts, obs_idx, det_idx)
                    entodidx = (
                        det_sample_count_idx(tod_counts, obs_idx, det_idx)
                        + tod_counts[obs_idx, det_idx]
                    )

                    diagonal[stdiagidx:endiagidx] = noise_variance[det_list[det_idx]][
                        sttodidx:entodidx
                    ]

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
