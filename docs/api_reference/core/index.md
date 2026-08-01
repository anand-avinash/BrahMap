# Core Interface

<!-- markdownlint-disable MD013 -->

## Data pre-processing

| Class | Description |
| :--- | :--- |
| [`SolverType`](SolverType.md) | Enumeration class defining the map-making solver configuration ($I$, $QU$, or $IQU$). |
| [`ProcessTimeSamples`](ProcessTimeSamples.md) | Standard container for pre-processed pointing information. |
| [`SharedMemProcessTimeSamples`](SharedMemProcessTimeSamples.md) | MPI shared-memory optimized container for pre-processed pointing information. |

## Linear operators for map-making

| Class | Description |
| :--- | :--- |
| [`PointingLO`](PointingLO.md) | Pointing operator mapping time-ordered data to and from the sky map. |
| [`BlockDiagonalPreconditionerLO`](BlockDiagonalPreconditionerLO.md) | Block-diagonal preconditioner operator for iterative map-making solvers. |

## Noise covariance (and their inverse) operators

| Class | Description |
| :--- | :--- |
| [`NoiseCovLO_Diagonal`](NoiseCovLO_Diagonal.md) | Diagonal noise covariance operator for uncorrelated detector noise. |
| [`NoiseCovLO_Circulant`](NoiseCovLO_Circulant.md) | Circulant noise covariance operator for stationary detector noise with the covariance modeled as a circulant matrix. |
| [`NoiseCovLO_Toeplitz01`](NoiseCovLO_Toeplitz01.md) | Toeplitz noise covariance operator for stationary detector noise with the covariance modeled as a Toeplitz matrix. |
| [`BlockDiagNoiseCovLO`](BlockDiagNoiseCovLO.md) | Block-diagonal noise covariance operator for multi-detector/multi-observation systems. |
| [`InvNoiseCovLO_Diagonal`](InvNoiseCovLO_Diagonal.md) | Inverse diagonal noise covariance operator for uncorrelated detector noise. |
| [`InvNoiseCovLO_Circulant`](InvNoiseCovLO_Circulant.md) | Inverse circulant noise covariance operator for stationary detector noise with the covariance modeled as a circulant matrix. |
| [`InvNoiseCovLO_Toeplitz01`](InvNoiseCovLO_Toeplitz01.md) | Inverse Toeplitz noise covariance operator for stationary detector noise with the covariance modeled as a Toeplitz matrix. |
| [`BlockDiagInvNoiseCovLO`](BlockDiagInvNoiseCovLO.md) | Inverse block-diagonal noise covariance operator for multi-detector/multi-observation systems. |

## GLS map-making functions and tools

| Class / Function | Description |
| :--- | :--- |
| [`GLSParameters`](GLSParameters.md) | Configuration parameters class for the core GLS map-making solver. |
| [`compute_GLS_maps_from_PTS`](compute_GLS_maps_from_PTS.md) | Computes GLS maps directly using a pre-constructed [`ProcessTimeSamples`](ProcessTimeSamples.md) or [`SharedMemProcessTimeSamples`](SharedMemProcessTimeSamples.md) object. |
| [`compute_GLS_maps`](compute_GLS_maps.md) | Computes GLS maps from raw pointing and time-ordered data inputs. |
| [`separate_map_vectors`](separate_map_vectors.md) | Splits a consolidated map vector into its temperature and polarization components. |
| [`GLSResult`](GLSResult.md) | Result container holding the reconstructed maps, solver iteration details, and convergence status. |

<!-- markdownlint-enable MD013 -->
