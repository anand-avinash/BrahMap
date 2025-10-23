import gc
from typing import List, Union, Optional
from dataclasses import dataclass, asdict

import numpy as np
import litebird_sim as lbs

from ..core import GLSParameters, GLSResult, compute_GLS_maps_from_PTS, DTypeNoiseCov

from ..lbsim import LBSimProcessTimeSamples, DTypeLBSNoiseCov

from ..math import DTypeFloat


@dataclass
class LBSimGLSParameters(GLSParameters):
    """A class to encapsulate the parameters used for GLS map-making with
    `litebird_sim` data

    Parameters
    ----------
    solver_type : SolverType
        _description_
    use_iterative_solver : bool
        _description_
    isolver_threshold : float
        _description_
    isolver_max_iterations : int
        _description_
    callback_function : Callable
        _description_
    return_processed_samples : bool
        _description_
    return_hit_map : bool
        _description_
    return_processed_samples : bool
        _description_
    output_coordinate_system : lbs.CoordinateSystem
        _description_
    """

    return_processed_samples: bool = False
    output_coordinate_system: lbs.CoordinateSystem = lbs.CoordinateSystem.Galactic


@dataclass
class LBSimGLSResult(GLSResult):
    """A class to store the results of the GLs map-making done with `litebird_sim` data

    Parameters
    ----------
    solver_type : SolverType
        _description_
    npix : int
        _description_
    new_npix : int
        _description_
    GLS_maps : np.ndarray
        _description_
    hit_map : np.ndarray
        _description_
    convergence_status : bool
        _description_
    num_iterations : int
        _description_
    GLSParameters : GLSParameters
        _description_
    nside : int
        _description_
    coordinate_system : lbs.CoordinateSystem
        _description_
    """

    nside: int
    coordinate_system: lbs.CoordinateSystem


def LBSim_compute_GLS_maps(
    nside: int,
    observations: Union[lbs.Observation, List[lbs.Observation]],
    pointings: Union[np.ndarray, List[np.ndarray], None] = None,
    hwp: Optional[lbs.HWP] = None,
    components: Union[str, List[str]] = "tod",
    pointings_flag: Optional[np.ndarray] = None,
    inv_noise_cov_operator: Union[DTypeNoiseCov, DTypeLBSNoiseCov, None] = None,
    threshold: float = 1.0e-5,
    dtype_float: Optional[DTypeFloat] = None,
    LBSim_gls_parameters: LBSimGLSParameters = LBSimGLSParameters(),
) -> Union[LBSimGLSResult, tuple[LBSimProcessTimeSamples, LBSimGLSResult]]:
    """_summary_

    Parameters
    ----------
    nside : int
        _description_
    observations : Union[lbs.Observation, List[lbs.Observation]]
        _description_
    pointings : Union[np.ndarray, List[np.ndarray], None], optional
        _description_, by default None
    hwp : Optional[lbs.HWP], optional
        _description_, by default None
    components : Union[str, List[str]], optional
        _description_, by default "tod"
    pointings_flag : Optional[np.ndarray], optional
        _description_, by default None
    inv_noise_cov_operator : Union[DTypeNoiseCov, DTypeLBSNoiseCov, None], optional
        _description_, by default None
    threshold : float, optional
        _description_, by default 1.0e-5
    dtype_float : Optional[DTypeFloat], optional
        _description_, by default None
    LBSim_gls_parameters : LBSimGLSParameters, optional
        _description_, by default LBSimGLSParameters()

    Returns
    -------
    Union[LBSimGLSResult, tuple[LBSimProcessTimeSamples, LBSimGLSResult]]
        _description_
    """
    if inv_noise_cov_operator is None:
        noise_weights = None
    else:
        noise_weights = inv_noise_cov_operator.diag

    processed_samples = LBSimProcessTimeSamples(
        nside=nside,
        observations=observations,
        pointings=pointings,
        hwp=hwp,
        pointings_flag=pointings_flag,
        solver_type=LBSim_gls_parameters.solver_type,
        noise_weights=noise_weights,
        output_coordinate_system=LBSim_gls_parameters.output_coordinate_system,
        threshold=threshold,
        dtype_float=dtype_float,
    )

    if isinstance(components, str):
        components = [components]

    if len(components) > 1:
        lbs.mapmaking.destriper._sum_components_into_obs(
            obs_list=processed_samples.obs_list,
            target=components[0],
            other_components=components[1:],
            factor=1.0,
        )

    time_ordered_data = np.concatenate(
        [getattr(obs, components[0]) for obs in processed_samples.obs_list], axis=None
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

    if LBSim_gls_parameters.return_processed_samples:
        return processed_samples, gls_result
    else:
        del processed_samples
        gc.collect()
        return gls_result
