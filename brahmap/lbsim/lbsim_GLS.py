import gc
from typing import List, Union
from dataclasses import dataclass, asdict

import numpy as np
import litebird_sim as lbs

from ..core import GLSParameters, GLSResult, compute_GLS_maps_from_PTS, DTypeNoiseCov

from ..lbsim import LBSimProcessTimeSamples, DTypeLBSNoiseCov

from ..math import DTypeFloat


@dataclass
class LBSimGLSParameters(GLSParameters):
    return_processed_samples: bool = False
    output_coordinate_system: lbs.CoordinateSystem = lbs.CoordinateSystem.Galactic


@dataclass
class LBSimGLSResult(GLSResult):
    nside: int
    coordinate_system: lbs.CoordinateSystem


def LBSim_compute_GLS_maps(
    nside: int,
    observations: Union[lbs.Observation, List[lbs.Observation]],
    component: str = "tod",
    pointings_flag: Union[np.ndarray, None] = None,
    inv_noise_cov_operator: Union[DTypeNoiseCov, DTypeLBSNoiseCov, None] = None,
    threshold: float = 1.0e-5,
    dtype_float: Union[DTypeFloat, None] = None,
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
        del processed_samples
        gc.collect()
        return gls_result
