from typing import List, Any

import numpy as np
import numpy.typing as npt
import healpy as hp
import litebird_sim as lbs
from ducc0.healpix import Healpix_Base

from ..core import SolverType, ProcessTimeSamples
from ..math import DTypeFloat
from ..mpi import MPI_UTILS

# For backwards compatibility with lbs v0.17.0 and earlier
if hasattr(lbs, "observation_utilities"):
    pointing_tools = lbs.observation_utilities
else:
    pointing_tools = lbs.pointings_in_obs


class LBSimProcessTimeSamples(ProcessTimeSamples):
    """A data container to store the pre-processed and pre-computed arrays and
    metadata from `litebird_sim` observations.

    Similar to [`ProcessTimeSamples`][brahmap.core.ProcessTimeSamples],
    this container object can be used to create pointing operators,
    block-diagonal preconditioners, etc. as required for map-making.

    Parameters
    ----------
    nside : int
        The HEALPix $N_{side}$ resolution parameter defining the number of pixels
    observations : lbs.Observation | List[lbs.Observation]
        An instance of the `Observation` class or a list of the same
    pointings : npt.NDArray[np.number] | List[npt.NDArray[np.number]] | None, optional
        Array of detector pointing indices mapping time samples to observed sky pixels,
        by default `None`
    hwp : lbs.HWP | None, optional
        The Half-Wave Plate (HWP) angles or configuration, by default `None`
    pointings_flag : npt.NDArray[np.bool_] | None, optional
        Boolean array indicating valid pointing samples, by default `None`.
        The `True` value indicates a valid pointing, and the `False`
        value indicates a bad pointing. If set as `None`, all the
        pointings are considered valid
    solver_type : SolverType, optional
        The level of map-making solver to construct ($I$, $QU$, or
        $IQU$), by default `SolverType.IQU`
    noise_weights : npt.NDArray[np.number] | None, optional
        Array of noise inverse noise variance for each time sample, by
        default `None`. If set as `None`, inverse noise variance is set to 1 for each
        time sample
    output_coordinate_system : lbs.CoordinateSystem, optional
        The celestial coordinate system to use for the generated output maps, by
        default `lbs.CoordinateSystem.Galactic`
    threshold : float, optional
        The condition number threshold used to flag degenerate or under-sampled
        pixels, by default `1.0e-5`
    dtype_float : DTypeFloat, optional
        The data type to use for floating point arrays, by default
        `np.float64`
    """

    def __init__(
        self,
        nside: int,
        observations: lbs.Observation | List[lbs.Observation],
        pointings: npt.NDArray[np.number] | List[npt.NDArray[np.number]] | None = None,
        hwp: lbs.HWP | None = None,
        pointings_flag: npt.NDArray[np.bool_] | None = None,
        solver_type: SolverType = SolverType.IQU,
        noise_weights: npt.NDArray[np.number] | None = None,
        output_coordinate_system: lbs.CoordinateSystem = lbs.CoordinateSystem.Galactic,
        threshold: float = 1.0e-5,
        dtype_float: DTypeFloat = np.float64,
    ) -> None:
        self.__nside = nside
        self.__coordinate_system = output_coordinate_system
        npix = hp.nside2npix(self.nside)
        hpx = Healpix_Base(nside, "RING")

        (
            self.__obs_list,
            ptg_list,
        ) = pointing_tools._normalize_observations_and_pointings(
            observations=observations, pointings=pointings
        )

        num_total_samples = 0
        for obs in self.obs_list:
            num_total_samples += obs.n_detectors * obs.n_samples

        pix_indices = np.empty(num_total_samples, dtype=int)
        pol_angles = np.empty(num_total_samples, dtype=dtype_float)

        start_idx = 0
        end_idx = 0
        for obs_idx, (obs, curr_pointings) in enumerate(zip(self.obs_list, ptg_list)):
            if hwp is None:
                hwp_angle = None
            else:
                hwp_angle = pointing_tools._get_hwp_angle(
                    obs=obs, hwp=hwp, pointing_dtype=dtype_float
                )

            curr_pointings_det: Any = None

            for det_idx in range(obs.n_detectors):
                (
                    curr_pointings_det,
                    hwp_angle,
                ) = pointing_tools._get_pointings_array(
                    detector_idx=det_idx,
                    pointings=curr_pointings,
                    hwp_angle=hwp_angle,
                    output_coordinate_system=output_coordinate_system,
                    pointings_dtype=dtype_float,
                    nthreads=MPI_UTILS.nthreads_per_process,
                )

                end_idx += obs.n_samples

                pol_angles[start_idx:end_idx] = pointing_tools._get_pol_angle(
                    curr_pointings_det=curr_pointings_det,
                    hwp_angle=hwp_angle,
                    pol_angle_detectors=obs.pol_angle_rad[det_idx],
                )

                pix_indices[start_idx:end_idx] = hpx.ang2pix(
                    curr_pointings_det[:, :2],
                    nthreads=MPI_UTILS.nthreads_per_process,
                )

                start_idx = end_idx

            del hwp_angle, curr_pointings_det

        del curr_pointings

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
    def obs_list(self) -> List[lbs.Observation]:
        """A list of the parsed `litebird_sim` observations.

        Returns
        -------
        List[lbs.Observation]
            The list of observations
        """
        return self.__obs_list

    @property
    def nside(self) -> int:
        """The HEALPix resolution parameter.

        Returns
        -------
        int
            The $N_{side}$ parameter
        """
        return self.__nside

    @property
    def coordinate_system(self) -> lbs.CoordinateSystem:
        """The output celestial coordinate system used in data processing.

        Returns
        -------
        lbs.CoordinateSystem
            The configured coordinate system
        """
        return self.__coordinate_system
