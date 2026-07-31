import gc
from typing import List
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import litebird_sim as lbs

from ..base import DTypeNoiseCov

from ..core import GLSParameters, GLSResult, compute_GLS_maps_from_PTS

from ..lbsim import (
    LBSimProcessTimeSamples,
    LBSimSharedMemProcessTimeSamples,
    DTypeLBSNoiseCov,
)

from ..math import DTypeFloat


@dataclass
class LBSimGLSParameters(GLSParameters):
    """A data class encapsulating the configuration parameters for the
    Generalized Least Squares (GLS) map-making algorithm with `litebird_sim` data.

    Attributes
    ----------
    solver_type : SolverType
        The map-making solver configuration to use (e.g. $I$, $QU$, $IQU$)
    use_iterative_solver : bool
        Whether to enforce the use of an iterative solver (like PCG) for map-making
    isolver_threshold : float
        The numerical tolerance threshold for the iterative solver to
        declare convergence
    isolver_max_iterations : int
        The maximum number of iterations allowed for the iterative solver
    callback_function : Callable
        A callable function executed at each iteration of the solver
    return_processed_samples : bool
        Whether the GLS solver function should return the processed time
        samples container
    return_hit_map : bool
        Whether the function should return the pixel hit map
    shmem_return_copy : bool
        Whether the linear operators (PointingLO, BlockDiagonalPreconditionerLO)
        should return copies of the shared memory buffer during matrix-vector
        products. Only applicable when using shared memory process time
        samples class instances, by default `True`
    output_coordinate_system : lbs.CoordinateSystem
        The celestial coordinate system to use for the generated output maps
    """

    return_processed_samples: bool = False
    output_coordinate_system: lbs.CoordinateSystem = lbs.CoordinateSystem.Galactic


@dataclass
class LBSimGLSResult(GLSResult):
    """A data class storing the output results of the GLS map-making done with
    `litebird_sim` data.

    Attributes
    ----------
    solver_type : SolverType
        The map-making solver configuration (e.g. $I$, $QU$, $IQU$)
    npix : int
        The number of pixels in the sky map
    new_npix : int
        The number of valid pixels actually observed and processed
    GLS_maps : npt.NDArray[np.number]
        The final Generalized Least Squares (GLS) estimated sky maps
    hit_map : npt.NDArray[np.number] | None
        The array representing the total number of hits per pixel
    convergence_status : bool
        A boolean indicating whether the iterative solver successfully converged
    num_iterations : int
        The total number of iterations actually performed by the solver before stopping
    GLSParameters : LBSimGLSParameters
        The input parameters configuration used for the GLS map-making
    nside : int
        The HEALPix resolution parameter defining the number of pixels
    coordinate_system : lbs.CoordinateSystem
        The coordinate system to use for map-making (e.g., Galactic, Ecliptic)
    GLSParameters : GLSParameters
        The input parameters configuration used for the GLS map-making
    """

    nside: int
    coordinate_system: lbs.CoordinateSystem


def LBSim_compute_GLS_maps(
    nside: int,
    observations: lbs.Observation | List[lbs.Observation],
    pointings: npt.NDArray[np.number] | List[npt.NDArray[np.number]] | None = None,
    hwp: lbs.HWP | None = None,
    components: str | List[str] = "tod",
    pointings_flag: npt.NDArray[np.bool_] | None = None,
    inv_noise_cov_operator: DTypeNoiseCov | DTypeLBSNoiseCov | None = None,
    threshold: float = 1.0e-5,
    dtype_float: DTypeFloat = np.float64,
    LBSim_gls_parameters: LBSimGLSParameters = LBSimGLSParameters(),
    x0: npt.NDArray[np.number] | None = None,
    use_shared_memory: bool = False,
    nproc_reduce: int = 1,
) -> (
    LBSimGLSResult
    | tuple[LBSimProcessTimeSamples | LBSimSharedMemProcessTimeSamples, LBSimGLSResult]
):
    """Computes the Generalized Least Squares (GLS) maps from
    `litebird_sim` observations.

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
    components : str | List[str], optional
        A string or list defining the TOD components to be used for map-making, by
        default `"tod"`
    pointings_flag : npt.NDArray[np.bool_] | None, optional
        Boolean array indicating valid pointing samples, by default `None`.
        The `True` value indicates a valid pointing, and the `False`
        value indicates a bad pointing. If set as `None`, all the
        pointings are considered valid
    inv_noise_cov_operator : DTypeNoiseCov | DTypeLBSNoiseCov | None, optional
        The inverse noise covariance linear operator ($N^{-1}$), by default `None`
    threshold : float, optional
        The condition number threshold used to flag degenerate or
        under-sampled pixels, by default `1.0e-5`
    dtype_float : DTypeFloat, optional
        The data type to use for floating point arrays, by default
        `np.float64`
    LBSim_gls_parameters : LBSimGLSParameters, optional
        The parameter configuration dictating the map-making behavior, by
        default `LBSimGLSParameters()`
    x0 : npt.NDArray[np.number] | None, optional
        Initial guess for the GLS solution in the form of interleaved
        maps (e.g. $[I_1, Q_1, U_1, I_2, Q_2, U_2, ]\dots]$), by default `None`
    use_shared_memory : bool, optional
        Whether to use MPI shared memory based process time samples, by
        default `False`
    nproc_reduce : int, optional
        Number of processes used in parallel reduction within nodes for
        shared memory mode. See
        [`SharedMemProcessTimeSamples`][brahmap.mpi.SharedMemProcessTimeSamples]
        for more details. By default `1`

    Returns
    -------
    LBSimGLSResult | tuple[LBSimProcessTimeSamples | LBSimSharedMemProcessTimeSamples, LBSimGLSResult]
        The dataclass containing the final output from the GLS map-maker,
        optionally returning the processed samples container
    """
    if inv_noise_cov_operator is None:
        noise_weights = None
    else:
        noise_weights = inv_noise_cov_operator.diag

    if use_shared_memory:
        processed_samples: (
            LBSimProcessTimeSamples | LBSimSharedMemProcessTimeSamples
        ) = LBSimSharedMemProcessTimeSamples(
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
            nproc_reduce=nproc_reduce,
        )
    else:
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
        x0=x0,
    )

    lbsim_gls_result = LBSimGLSResult(
        nside=nside,
        coordinate_system=LBSim_gls_parameters.output_coordinate_system,
        **gls_result.__dict__,
    )

    if LBSim_gls_parameters.return_processed_samples:
        return processed_samples, lbsim_gls_result
    else:
        del processed_samples
        gc.collect()
        return lbsim_gls_result
