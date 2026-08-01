# `litebird_sim` Interface

<!-- markdownlint-disable MD013 -->

## Data pre-processing

| Class | Description |
| :--- | :--- |
| [`LBSimProcessTimeSamples`](LBSimProcessTimeSamples.md) | Standard container for pre-processed pointing information and time-ordered data from `litebird_sim` observations. |
| [`LBSimSharedMemProcessTimeSamples`](LBSimSharedMemProcessTimeSamples.md) | MPI shared-memory optimized container for pre-processed pointing information and time-ordered data from `litebird_sim` observations. |

## Noise covariance (and their inverse) operators

| Class | Description |
| :--- | :--- |
| [`LBSim_InvNoiseCovLO_UnCorr`](LBSim_InvNoiseCovLO_UnCorr.md) | Inverse noise covariance operator for uncorrelated/white detector noise. |
| [`LBSim_InvNoiseCovLO_Circulant`](LBSim_InvNoiseCovLO_Circulant.md) | Inverse noise covariance operator for stationary noise with the covariance modeled as a circulant matrix. |
| [`LBSim_InvNoiseCovLO_Toeplitz`](LBSim_InvNoiseCovLO_Toeplitz.md) | Inverse noise covariance operator for stationary noise with the covariance modeled as a Toeplitz matrix. |

## GLS map-making functions and tools

| Class / Function | Description |
| :--- | :--- |
| [`LBSimGLSParameters`](LBSimGLSParameters.md) | Configuration parameters class for the GLS map-making solver. |
| [`LBSim_compute_GLS_maps`](LBSim_compute_GLS_maps.md) | Computes the GLS temperature and polarization maps from `litebird_sim` observations. |
| [`LBSimGLSResult`](LBSimGLSResult.md) | Result container holding the reconstructed maps, solver iteration details, and convergence status. |

<!-- markdownlint-enable MD013 -->
