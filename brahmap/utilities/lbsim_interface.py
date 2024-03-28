import numpy as np
import healpy as hp
import litebird_sim as lbs

from .process_time_samples import ProcessTimeSamples, SolverType


def lbs_process_timesamples(
    nside: int,
    pointings: np.ndarray,
    pointings_flag: np.ndarray = None,
    solver_type: SolverType = SolverType.IQU,
    pol_angles: np.ndarray = None,
    noise_weights: np.ndarray = None,
    threshold_cond: float = 1.0e3,
    galactic_coords: bool = True,
):
    """This function accepts the pointing and polarization angle arrays from `litebird_sim`, rotates them from elliptic to galactic coordinate system, generates the pixel indices of the pointings and then passes them to :func:`ProcessTimeSamples`.

    Args:

    - ``nside`` (int): nside for the output map
    - ``pointings`` (np.ndarray): An array of detector pointings of shape (nsamp, 2)
    - ``pol_angles`` (np.ndarray): A 1-d array of polarization angle
    - ``pol_idx`` (int): Type of map-making to use. Defaults to 3.
    - ``w`` (np.ndarray): array with noise weights , :math:`w_t= N^{-1} _{tt}`, computed by :func:`BlockLO.build_blocks`. If it is  not set :func:`ProcessTimeSamples.initializeweights` assumes it to be a :func:`numpy.ones` array. Defaults to None.
    - ``threshold_cond`` (float): Sets the condition number threshold to mask bad conditioned pixels (it's used in polarization cases). Defaults to 1.e3.
    - ``obspix`` (np.ndarray): Map from the internal pixelization to an external one, i.e. HEALPIX, it has to be modified when pathological pixels are not taken into account. It not set, it is assumed to be `numpy.arange(npix). Defaults to None.
    - ``galactic_coords`` (bool, optional): Say yes if you want your result in galactic coordinates. Defaults to True.

    Returns:

    - ``pointings`` (np.ndarray): Pointings as pixel index
    - ``ProcessTimeSamples``: ProcessTimeSamples class
    """
    if galactic_coords:
        pointings, pol_angles = lbs.coordinates.rotate_coordinates_e2g(
            pointings_ecl=pointings, pol_angle_ecl=pol_angles
        )

    pointings = hp.ang2pix(nside, pointings[:, 0], pointings[:, 1])

    return pointings, ProcessTimeSamples(
        npix=hp.nside2npix(nside),
        pointings=pointings,
        pointings_flag=pointings_flag,
        solver_type=solver_type,
        pol_angles=pol_angles,
        noise_weights=noise_weights,
        threshold_cond=threshold_cond,
    )
