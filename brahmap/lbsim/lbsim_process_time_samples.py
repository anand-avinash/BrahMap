from typing import List, Union, Optional

import numpy as np
import healpy as hp
import litebird_sim as lbs

from ..core import SolverType, ProcessTimeSamples


class LBSimProcessTimeSamples(ProcessTimeSamples):
    def __init__(
        self,
        nside: int,
        observations: Union[lbs.Observation, List[lbs.Observation]],
        pointings: Union[np.ndarray, List[np.ndarray], None] = None,
        hwp: Optional[lbs.HWP] = None,
        pointings_flag: Union[np.ndarray, None] = None,
        solver_type: SolverType = SolverType.IQU,
        noise_weights: Union[np.ndarray, None] = None,
        output_coordinate_system: lbs.CoordinateSystem = lbs.CoordinateSystem.Galactic,
        threshold: float = 1.0e-5,
        dtype_float=np.float64,
    ):
        self.__nside = nside
        self.__coordinate_system = output_coordinate_system
        npix = hp.nside2npix(self.nside)

        (
            self.__obs_list,
            ptg_list,
        ) = lbs.pointings_in_obs._normalize_observations_and_pointings(
            observations=observations, pointings=pointings
        )

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

        for obs_idx, (obs, curr_pointings) in enumerate(zip(self.obs_list, ptg_list)):
            if hwp is None:
                hwp_angle = None
            else:
                hwp_angle = lbs.pointings_in_obs._get_hwp_angle(
                    obs=obs, hwp=hwp, pointing_dtype=dtype_float
                )

            for det_idx in range(obs.n_detectors):
                (
                    curr_pointings_det,
                    hwp_angle,
                ) = lbs.pointings_in_obs._get_pointings_array(
                    detector_idx=det_idx,
                    pointings=curr_pointings,
                    hwp_angle=hwp_angle,
                    output_coordinate_system=output_coordinate_system,
                    pointings_dtype=dtype_float,
                )

                pol_angles[obs_idx, det_idx] = lbs.pointings_in_obs._get_pol_angle(
                    curr_pointings_det=curr_pointings_det,
                    hwp_angle=hwp_angle,
                    pol_angle_detectors=obs.pol_angle_rad[det_idx],
                )

                pix_indices[obs_idx, det_idx] = hp.ang2pix(
                    nside, curr_pointings_det[:, 0], curr_pointings_det[:, 1]
                )

            del hwp_angle, curr_pointings_det

        del curr_pointings

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
