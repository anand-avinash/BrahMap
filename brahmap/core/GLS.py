import gc

import numpy as np

from dataclasses import dataclass
from typing import Union, Callable

from ..mpi import MPI_RAISE_EXCEPTION

from ..core import (
    SolverType,
    ProcessTimeSamples,
    PointingLO,
    BlockDiagonalPreconditionerLO,
    InvNoiseCovLO_Uncorrelated,
    DTypeNoiseCov,
)

from ..math import cg, DTypeFloat


@dataclass
class GLSParameters:
    solver_type: SolverType = SolverType.IQU
    use_preconditioner: bool = True
    preconditioner_threshold: float = 1.0e-12
    preconditioner_max_iterations: int = 100
    callback_function: Callable = None
    return_processed_samples: bool = False
    return_hit_map: bool = False


@dataclass
class GLSResult:
    solver_type: SolverType
    npix: int
    new_npix: int
    GLS_maps: np.ndarray
    hit_map: np.ndarray
    convergence_status: bool
    num_iterations: int
    GLSParameters: GLSParameters


def separate_map_vectors(
    map_vector: np.ndarray, processed_samples: ProcessTimeSamples
) -> np.ndarray:
    try:
        map_vector = np.reshape(
            map_vector,
            shape=(processed_samples.solver_type, processed_samples.new_npix),
            order="F",
        )
    except TypeError:
        # `newshape` parameter has been deprecated since numpy 2.1.0. This part should be removed once the support is dropped for lower version
        map_vector = np.reshape(
            map_vector,
            newshape=(processed_samples.solver_type, processed_samples.new_npix),
            order="F",
        )

    output_maps = np.ma.MaskedArray(
        data=np.empty(processed_samples.npix, dtype=processed_samples.dtype_float),
        mask=~processed_samples.pixel_flag,
        fill_value=-1.6375e30,
    )

    output_maps = np.tile(A=output_maps, reps=(processed_samples.solver_type, 1))

    for idx in range(processed_samples.solver_type):
        output_maps[idx][~output_maps[idx].mask] = map_vector[idx]

    return output_maps


def compute_GLS_maps_from_PTS(
    processed_samples: ProcessTimeSamples,
    time_ordered_data: np.ndarray,
    inv_noise_cov_operator: Union[DTypeNoiseCov, None] = None,
    gls_parameters: GLSParameters = GLSParameters(),
) -> GLSResult:
    MPI_RAISE_EXCEPTION(
        condition=(processed_samples.nsamples != len(time_ordered_data)),
        exception=ValueError,
        message=f"Size of `pointings` must be equal to the size of `time_ordered_data` array:\nlen(pointings) = {processed_samples.nsamples}\nlen(time_ordered_data) = {len(time_ordered_data)}",
    )

    try:
        time_ordered_data = time_ordered_data.astype(
            dtype=processed_samples.dtype_float, casting="safe", copy=False
        )
    except TypeError:
        raise TypeError(
            f"The `time_ordered_data` array has higher dtype than `processed_samples.dtype_float={processed_samples.dtype_float}`. Please compute `processed_samples` again with `dtype_float={time_ordered_data.dtype}`"
        )

    if inv_noise_cov_operator is None:
        inv_noise_cov_operator = InvNoiseCovLO_Uncorrelated(
            diag=np.ones(
                processed_samples.nsamples,
                dtype=processed_samples.dtype_float,
            )
        )
    else:
        MPI_RAISE_EXCEPTION(
            condition=(inv_noise_cov_operator.shape[0] != processed_samples.nsamples),
            exception=ValueError,
            message=f"The shape of `inv_noise_cov_operator` must be same as `(len(time_ordered_data), len(time_ordered_data))`:\nlen(time_ordered_data) = {len(time_ordered_data)}\ninv_noise_cov_operator.shape = ({inv_noise_cov_operator.shape}, {inv_noise_cov_operator.shape})",
        )

    pointing_operator = PointingLO(
        processed_samples=processed_samples, solver_type=gls_parameters.solver_type
    )

    blockdiagprecond_operator = BlockDiagonalPreconditionerLO(
        processed_samples=processed_samples, solver_type=gls_parameters.solver_type
    )

    b = pointing_operator.T * inv_noise_cov_operator * time_ordered_data

    num_iterations = 0
    if gls_parameters.use_preconditioner:

        def callback_function(x, r, norm_residual):
            nonlocal num_iterations
            num_iterations += 1
            if gls_parameters.callback_function is not None:
                gls_parameters.callback_function(x, r, norm_residual)

        A = pointing_operator.T * inv_noise_cov_operator * pointing_operator

        map_vector, pcg_status = cg(
            A=A,
            b=b,
            atol=gls_parameters.preconditioner_threshold,
            maxiter=gls_parameters.preconditioner_max_iterations,
            M=blockdiagprecond_operator,
            callback=callback_function,
        )
    else:
        pcg_status = 0
        map_vector = blockdiagprecond_operator * b

    output_maps = separate_map_vectors(
        map_vector=map_vector, processed_samples=processed_samples
    )

    if gls_parameters.return_hit_map is True:
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
    pointings: np.ndarray,
    time_ordered_data: np.ndarray,
    pointings_flag: Union[np.ndarray, None] = None,
    pol_angles: Union[np.ndarray, None] = None,
    inv_noise_cov_operator: Union[DTypeNoiseCov, None] = None,
    threshold: float = 1.0e-5,
    dtype_float: Union[DTypeFloat, None] = None,
    update_pointings_inplace: bool = True,
    gls_parameters: GLSParameters = GLSParameters(),
) -> Union[GLSResult, tuple[ProcessTimeSamples, GLSResult]]:
    if dtype_float is None:
        if pol_angles is None:
            dtype_float = time_ordered_data.dtype
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
    )

    if gls_parameters.return_processed_samples is True:
        return processed_samples, gls_result
    else:
        del processed_samples
        gc.collect()
        return gls_result
