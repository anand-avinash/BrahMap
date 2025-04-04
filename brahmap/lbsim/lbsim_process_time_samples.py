import warnings
from typing import List, Union

import numpy as np
import healpy as hp
import litebird_sim as lbs

from ..core import SolverType, ProcessTimeSamples

from ..utilities import LowerTypeCastWarning


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
