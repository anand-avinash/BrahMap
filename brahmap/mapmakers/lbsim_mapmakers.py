import numpy as np
from dataclasses import dataclass, asdict
import healpy as hp
import litebird_sim as lbs

from brahmap.mapmakers import GLSParameters, GLSResult, compute_GLS_maps
from brahmap.linop import DiagonalOperator
from brahmap.interfaces import ToeplitzLO, BlockLO, InvNoiseCovLO_Uncorrelated
from brahmap.utilities import ProcessTimeSamples


@dataclass
class LBSimGLSParameters(GLSParameters):
    output_coordinate_system: lbs.CoordinateSystem = lbs.CoordinateSystem.Galactic


@dataclass
class LBSimGLSResult(GLSResult):
    nside: int
    coordinate_system: lbs.CoordinateSystem = lbs.CoordinateSystem.Galactic


def LBSim_compute_GLS_maps(
    nside: int,
    pointings: np.ndarray,
    tod: np.ndarray,
    pointings_flag: np.ndarray = None,
    pol_angles: np.ndarray = None,
    inv_noise_cov_operator: (
        ToeplitzLO | BlockLO | DiagonalOperator | InvNoiseCovLO_Uncorrelated
    ) = None,
    threshold_cond: float = 1.0e3,
    dtype_float=None,
    update_pointings_inplace: bool = True,
    LBSimGLSParameters: LBSimGLSParameters = LBSimGLSParameters(),
) -> LBSimGLSResult | tuple[ProcessTimeSamples, LBSimGLSResult]:
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
        threshold_cond=threshold_cond,
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
