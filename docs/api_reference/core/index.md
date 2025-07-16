# Core Interfaces

## Data pre-processing

- [`SolverType`](SolverType.md)
- [`ProcessTimeSamples`](ProcessTimeSamples.md)

## Linear operators for map-making

- [`PointingLO`](PointingLO.md)
- [`BlockDiagonalPreconditionerLO`](BlockDiagonalPreconditionerLO.md)

## Noise covariance (and their inverse) operators

- [`NoiseCovLO_Diagonal`](NoiseCovLO_Diagonal.md)
- [`NoiseCovLO_Circulant`](NoiseCovLO_Circulant.md)
- [`NoiseCovLO_Toeplitz01`](NoiseCovLO_Toeplitz01.md)
- [`BlockDiagNoiseCovLO`](BlockDiagNoiseCovLO.md)
- [`InvNoiseCovLO_Diagonal`](InvNoiseCovLO_Diagonal.md)
- [`InvNoiseCovLO_Circulant`](InvNoiseCovLO_Circulant.md)
- [`InvNoiseCovLO_Toeplitz01`](InvNoiseCovLO_Toeplitz01.md)
- [`BlockDiagInvNoiseCovLO`](BlockDiagInvNoiseCovLO.md)

## GLS map-making functions and tools

- [`GLSParameters`](GLSParameters.md)
- [`compute_GLS_maps_from_PTS`](compute_GLS_maps_from_PTS.md)
- [`compute_GLS_maps`](compute_GLS_maps.md)
- [`separate_map_vectors`](separate_map_vectors.md)
- [`GLSResult`](GLSResult.md)
