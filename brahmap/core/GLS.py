import gc

import numpy as np
import numpy.typing as npt

from dataclasses import dataclass
from typing import Callable


from ..base import DTypeNoiseCov

from ..core import (
    SolverType,
    ProcessTimeSamples,
    PointingLO,
    BlockDiagonalPreconditionerLO,
    InvNoiseCovLO_Diagonal,
)

from ..math import cg, DTypeFloat


@dataclass
class GLSParameters:
    """A data class encapsulating the configuration parameters for the
    Generalized Least Squares (GLS) map-making algorithm.

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
        Whether the function should the pixel hit map
    """

    solver_type: SolverType = SolverType.IQU
    use_iterative_solver: bool = True
    isolver_threshold: float = 1.0e-12
    isolver_max_iterations: int = 100
    callback_function: Callable | None = None
    return_processed_samples: bool = False
    return_hit_map: bool = False


@dataclass
class GLSResult:
    """A data class storing the output results of the Generalized Least
    Squares (GLS) map-making algorithm.

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
    GLSParameters : GLSParameters
        The input parameters configuration used for the GLS map-making
    """

    solver_type: SolverType
    npix: int
    new_npix: int
    GLS_maps: npt.NDArray[np.number]
    hit_map: npt.NDArray[np.number] | None
    convergence_status: bool
    num_iterations: int
    GLSParameters: GLSParameters


def separate_map_vectors(
    map_vector: npt.NDArray[np.number],
    processed_samples: ProcessTimeSamples,
) -> npt.NDArray[np.number]:
    r"""Separates the interleaved Stokes parameter maps into distinct components.

    The output maps of the GLS solver are typically interleaved in the form
    $[I_1, Q_1, U_1, I_2, Q_2, U_2, \dots]$. Following standard conventions,
    this function reshapes and separates the Stokes parameters into individual
    maps such as $[I_1, I_2, \dots]$, $[Q_1, Q_2, \dots]$, and $[U_1, U_2, \dots]$.

    Parameters
    ----------
    map_vector : npt.NDArray[np.number]
        The 1D vector representing the flattened interleaved sky map

    processed_samples : ProcessTimeSamples
        The pre-processed time samples object containing pointing and
        map-making metadata

    Returns
    -------
    npt.NDArray[np.number]
        The final separated output maps with masked pathological pixels
    """
    try:
        map_vector = np.reshape(
            map_vector,
            (int(processed_samples.solver_type), processed_samples.new_npix),
            order="F",
        )
    except TypeError:
        # `newshape` parameter has been deprecated since numpy 2.1.0. This part should
        # be removed once the support is dropped for lower version
        map_vector = np.reshape(
            map_vector,
            newshape=(int(processed_samples.solver_type), processed_samples.new_npix),
            order="F",
        )

    output_maps = np.ma.MaskedArray(
        data=np.empty(processed_samples.npix, dtype=processed_samples.dtype_float),
        mask=~processed_samples.pixel_flag,
        fill_value=-1.6375e30,
    )

    output_maps = np.tile(A=output_maps, reps=(int(processed_samples.solver_type), 1))

    for idx in range(int(processed_samples.solver_type)):
        output_maps[idx][~output_maps[idx].mask] = map_vector[idx]

    return output_maps


def compute_GLS_maps_from_PTS(
    processed_samples: ProcessTimeSamples,
    time_ordered_data: npt.NDArray[np.number],
    inv_noise_cov_operator: DTypeNoiseCov | None = None,
    gls_parameters: GLSParameters = GLSParameters(),
    x0: npt.NDArray[np.number] | None = None,
) -> GLSResult:
    r"""Computes the Generalized Least Squares (GLS) maps using a
    pre-instantiated `ProcessTimeSamples` instance.

    Parameters
    ----------
    processed_samples : ProcessTimeSamples
        The pre-processed time samples object containing pointing and
        map-making metadata
    time_ordered_data : npt.NDArray[np.number]
        The 1D vector representing the time-ordered data (TOD) streams
    inv_noise_cov_operator : DTypeNoiseCov | None, optional
        The inverse noise covariance linear operator ($N^{-1}$), by
        default `None`. If `None`, the identity matrix will be used as the
        inverse noise covariance.
    gls_parameters : GLSParameters, optional
        The parameter configuration dictating the map-making behavior, by
        default `GLSParameters()`
    x0 : npt.NDArray[np.number] | None, optional
        Initial guess for the GLS solution in the form of interleaved
        maps (e.g. $[I_1, Q_1, U_1, I_2, Q_2, U_2, \dots]$), by default `None`

    Returns
    -------
    GLSResult
        The dataclass containing the final output from the GLS map-maker
    """
    time_ordered_data = np.asarray(time_ordered_data)
    if processed_samples.nsamples != len(time_ordered_data):
        raise ValueError(
            f"Size of `pointings` must be equal to the size of `time_ordered_data` "
            f"array:\nlen(pointings) = {processed_samples.nsamples}\n"
            f"len(time_ordered_data) = {len(time_ordered_data)}"
        )

    try:
        time_ordered_data = time_ordered_data.astype(
            dtype=processed_samples.dtype_float, casting="safe", copy=False
        )
    except TypeError:
        raise TypeError(
            f"The `time_ordered_data` array has higher dtype than "
            f"`processed_samples.dtype_float={processed_samples.dtype_float}`. "
            f"Please compute `processed_samples` again with "
            f"`dtype_float={time_ordered_data.dtype}`"
        )

    if inv_noise_cov_operator is None:
        inv_noise_cov_operator = InvNoiseCovLO_Diagonal(
            size=processed_samples.nsamples, dtype=processed_samples.dtype_float
        )
    else:
        if inv_noise_cov_operator.shape[0] != processed_samples.nsamples:
            raise ValueError(
                f"The shape of `inv_noise_cov_operator` must be same as "
                f"`(len(time_ordered_data), len(time_ordered_data))`:\n"
                f"len(time_ordered_data) = {len(time_ordered_data)}\n"
                f"inv_noise_cov_operator.shape = ({inv_noise_cov_operator.shape}, "
                f"{inv_noise_cov_operator.shape})"
            )

    pointing_operator = PointingLO(
        processed_samples=processed_samples, solver_type=gls_parameters.solver_type
    )

    blockdiagprecond_operator = BlockDiagonalPreconditionerLO(
        processed_samples=processed_samples, solver_type=gls_parameters.solver_type
    )

    b = pointing_operator.T * inv_noise_cov_operator * time_ordered_data

    num_iterations = 0
    if gls_parameters.use_iterative_solver:

        def callback_function(x, r, norm_residual) -> None:
            nonlocal num_iterations
            num_iterations += 1
            if gls_parameters.callback_function is not None:
                gls_parameters.callback_function(x, r, norm_residual)

        A = pointing_operator.T * inv_noise_cov_operator * pointing_operator

        map_vector, pcg_status = cg(
            A=A,  # type: ignore
            b=b,  # type: ignore
            x0=x0,
            atol=gls_parameters.isolver_threshold,
            maxiter=gls_parameters.isolver_max_iterations,
            M=blockdiagprecond_operator,
            callback=callback_function,
            parallel=False,
        )
    else:
        pcg_status = 0
        map_vector = blockdiagprecond_operator * b

    output_maps = separate_map_vectors(
        map_vector=map_vector,  # type: ignore
        processed_samples=processed_samples,
    )

    if gls_parameters.return_hit_map:
        hit_map = processed_samples.get_hit_counts()
    else:
        hit_map = None

    if pcg_status != 0:
        convergence_status = False
    else:
        convergence_status = True

    gls_result = GLSResult(
        solver_type=processed_samples.solver_type,
        npix=processed_samples.npix,
        new_npix=processed_samples.new_npix,
        GLS_maps=output_maps,
        hit_map=hit_map,
        convergence_status=convergence_status,
        num_iterations=num_iterations,
        GLSParameters=gls_parameters,
    )

    return gls_result


def compute_GLS_maps(
    npix: int,
    pointings: npt.NDArray[np.integer],
    time_ordered_data: npt.NDArray[np.number],
    pointings_flag: npt.NDArray[np.bool_] | None = None,
    pol_angles: npt.NDArray[np.number] | None = None,
    inv_noise_cov_operator: DTypeNoiseCov | None = None,
    threshold: float = 1.0e-5,
    dtype_float: DTypeFloat | None = None,
    update_pointings_inplace: bool = True,
    gls_parameters: GLSParameters = GLSParameters(),
    x0: npt.NDArray[np.number] | None = None,
) -> GLSResult | tuple[ProcessTimeSamples, GLSResult]:
    r"""Computes the Generalized Least Squares (GLS) maps directly from
    raw pointing information and time-ordered data.

    Parameters
    ----------
    npix : int
        Number of pixels on which the map-making has to be done (e.g.
        `healpy.nside2npix(nside)`)
    pointings : npt.NDArray[np.integer]
        A 1-d array of pixel indices pointing to the sky map for each time sample
    time_ordered_data : npt.NDArray[np.number]
        The 1D vector representing the time-ordered data (TOD) streams
    pointings_flag : npt.NDArray[np.bool_] | None, optional
        A 1-d boolean array where `True` indicates a valid pointing and
        `False` flags a bad pointing, by default `None`. If set as `None`,
        all the pointings are considered valid
    pol_angles : npt.NDArray[np.number] | None, optional
        A 1-d array containing the polarization orientation angles of the
        detectors for each sample, by default `None`
    inv_noise_cov_operator : DTypeNoiseCov | None, optional
        The inverse noise covariance linear operator ($N^{-1}$), by default `None`
    threshold : float, optional
        The condition number threshold used to flag degenerate or
        under-sampled pixels, by default `1.0e-5`
    dtype_float : DTypeFloat | None, optional
        The data type used for floating-point arrays, by default `None`
    update_pointings_inplace : bool, optional
        Whether to update the pointing arrays in-place, by default `True`
    gls_parameters : GLSParameters, optional
        The parameter configuration dictating the map-making behavior, by
        default `GLSParameters()`
    x0 : npt.NDArray[np.number] | None, optional
        Initial guess for the GLS solution in the form of interleaved
        maps (e.g. $[I_1, Q_1, U_1, I_2, Q_2, U_2, \dots]$), by default `None`

    Returns
    -------
    GLSResult | tuple[ProcessTimeSamples, GLSResult]
        GLSResult
        The dataclass containing the final output from the GLS map-maker,
        optionally returning the processed samples container
    """
    if dtype_float is None:
        if pol_angles is None:
            dtype_float = time_ordered_data.dtype  # type: ignore
        else:
            dtype_float = np.promote_types(pol_angles.dtype, time_ordered_data.dtype)

    if pol_angles is not None:
        pol_angles = pol_angles.astype(dtype=dtype_float, copy=False)

    if inv_noise_cov_operator is None:
        noise_weights = None
    else:
        noise_weights = inv_noise_cov_operator.diag

    processed_samples = ProcessTimeSamples(
        npix=npix,
        pointings=pointings,
        pointings_flag=pointings_flag,
        solver_type=gls_parameters.solver_type,
        pol_angles=pol_angles,
        noise_weights=noise_weights,
        threshold=threshold,
        dtype_float=dtype_float,
        update_pointings_inplace=update_pointings_inplace,
    )

    gls_result = compute_GLS_maps_from_PTS(
        processed_samples=processed_samples,
        time_ordered_data=time_ordered_data.astype(dtype=dtype_float, copy=False),
        inv_noise_cov_operator=inv_noise_cov_operator,
        gls_parameters=gls_parameters,
        x0=x0,
    )

    if gls_parameters.return_processed_samples:
        return processed_samples, gls_result
    else:
        del processed_samples
        gc.collect()
        return gls_result
