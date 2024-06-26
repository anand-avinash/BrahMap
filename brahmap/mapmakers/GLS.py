import numpy as np
import scipy
from dataclasses import dataclass
from typing import Union, Callable

from brahmap import MPI_RAISE_EXCEPTION

from brahmap.linop import DiagonalOperator

from brahmap.utilities import ProcessTimeSamples, SolverType

from brahmap.interfaces import (
    PointingLO,
    ToeplitzLO,
    BlockLO,
    BlockDiagonalPreconditionerLO,
    InvNoiseCovLO_Uncorrelated,
)


@dataclass
class GLSParameters:
    solver_type: SolverType = SolverType.IQU
    use_preconditioner: bool = True
    preconditioner_threshold: float = 1.0e-5
    preconditioner_max_iterations: int = 100
    callback_function: Callable = None
    return_processed_samples: bool = True
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


def compute_GLS_maps(
    npix: int,
    pointings: np.ndarray,
    time_ordered_data: np.ndarray,
    pointings_flag: Union[np.ndarray, None] = None,
    pol_angles: Union[np.ndarray, None] = None,
    inv_noise_cov_operator: Union[
        ToeplitzLO, BlockLO, DiagonalOperator, InvNoiseCovLO_Uncorrelated, None
    ] = None,
    threshold: float = 1.0e-5,
    dtype_float=None,
    update_pointings_inplace: bool = True,
    GLSParameters: GLSParameters = GLSParameters(),
) -> Union[GLSResult, tuple[ProcessTimeSamples, GLSResult]]:
    MPI_RAISE_EXCEPTION(
        condition=(len(pointings) != len(time_ordered_data)),
        exception=ValueError,
        message=f"Size of `pointings` must be equal to the size of `time_ordered_data` array:\nlen(pointings) = {len(pointings)}\nlen(time_ordered_data) = {len(time_ordered_data)}",
    )

    if dtype_float is None:
        if pol_angles is not None:
            dtype_float = pol_angles.dtype
        else:
            dtype_float = np.float64

    if inv_noise_cov_operator is None:
        inv_noise_cov_operator = InvNoiseCovLO_Uncorrelated(
            diag=np.ones(len(pointings)), dtype=dtype_float
        )
    else:
        MPI_RAISE_EXCEPTION(
            condition=(inv_noise_cov_operator.shape[0] != len(time_ordered_data)),
            exception=ValueError,
            message=f"The shape of `inv_noise_cov_operator` must be same as `(len(time_ordered_data), len(time_ordered_data))`:\nlen(time_ordered_data) = {len(time_ordered_data)}\ninv_noise_cov_operator.shape = {inv_noise_cov_operator.shape}",
        )

    processed_samples = ProcessTimeSamples(
        npix=npix,
        pointings=pointings,
        pointings_flag=pointings_flag,
        solver_type=GLSParameters.solver_type,
        pol_angles=pol_angles,
        noise_weights=inv_noise_cov_operator.diag,
        threshold=threshold,
        dtype_float=dtype_float,
        update_pointings_inplace=update_pointings_inplace,
    )

    inv_noise_cov_operator.pointings_flag = processed_samples.pointings_flag

    pointing_operator = PointingLO(processed_samples=processed_samples)

    blockdiagprecond_operator = BlockDiagonalPreconditionerLO(
        processed_samples=processed_samples
    )

    b = pointing_operator.T * inv_noise_cov_operator * time_ordered_data

    num_iterations = 0
    if GLSParameters.use_preconditioner:

        def callback_function(x):
            nonlocal num_iterations
            num_iterations += 1
            if GLSParameters.callback_function is not None:
                GLSParameters.callback_function(x)

        A = pointing_operator.T * inv_noise_cov_operator * pointing_operator
        map_vector, pcg_status = scipy.sparse.linalg.cg(
            A=A,
            b=b,
            rtol=GLSParameters.preconditioner_threshold,
            maxiter=GLSParameters.preconditioner_max_iterations,
            M=blockdiagprecond_operator,
            callback=callback_function,
        )
    else:
        pcg_status = 0
        map_vector = blockdiagprecond_operator * b

    map_vector = np.reshape(
        a=map_vector,
        newshape=(processed_samples.solver_type, processed_samples.new_npix),
        order="F",
    )

    output_maps = np.ma.MaskedArray(
        data=np.empty(processed_samples.npix, dtype=dtype_float),
        mask=~processed_samples.pixel_flag,
        fill_value=-1.6375e30,
    )

    output_maps = np.tile(A=output_maps, reps=(processed_samples.solver_type, 1))

    for idx in range(processed_samples.solver_type):
        output_maps[idx][~output_maps[idx].mask] = map_vector[idx]

    if GLSParameters.return_hit_map is True:
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
        GLSParameters=GLSParameters,
    )

    if GLSParameters.return_processed_samples is True:
        return processed_samples, gls_result
    else:
        return gls_result
