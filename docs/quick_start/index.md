# BrahMap quick start guide

Complete example scripts and notebooks can be found
[here](https://github.com/anand-avinash/BrahMap/tree/main/examples).

By default, `BrahMap` performs all the operations over global MPI communicator
(`MPI.COMM_WORLD`). To modify this behavior, one can specify a different MPI
communicator through the function
`brahmap.MPI_UTILS.update_communicator(comm=...)`. This function must be
called before calling any other `BrahMap` functions.

## General map-making

A generic map-making using `BrahMap` roughly involve four steps:

1. Pre-processing the pointing information (assuming that signal contains
   uncorrelated noise)

    ```python
    # Creating the inverse white noise covariance operator
    inv_cov = brahmap.InvNoiseCovLO_Uncorrelated(
        diag=[...],         # Diagonal elements of the inverse of white noise
                            # covariance matrix

        dtype=np.float64,   # Numerical precision of the operator
    )

    # Pre-processing the pointing information
    processed_samples = brahmap.ProcessTimeSamples(
        npix=npix,                  # Number of pixels on which the map-making 
                                    # has to be done

        pointings=pointings,        # A 1-d of pointing indices

        pol_angles=pol_angles,      # A 1-d array containing the polarization angles
                                    # of the detectors

        noise_weights=inv_cov.diag, # A 1-d array of noise weights, or the diagonal
                                    # elements of the inverse noise covariance matrix

        dtype_float=numpy.float64,  # Numerical precision to be used throughout the
                                    # map-making
    )
    ```

2. Creating linear operators

    ```python
    # Pointing operator
    pointing_LO = brahmap.PointingLO(processed_samples)

    # Block-diagonal preconditioner
    precond_LO = brahmap.BlockDiagonalPreconditionerLO(processed_samples)
    ```

3. Performing the map-making (GLS in this example)

    ```python
    A = pointing_LO.T * inv_cov * pointing_LO
    b = pointing_LO.T * inv_cov * tod_array

    # Solving for x in the linear equation A.x=b using preconditioner
    map_vector = scipy.sparse.linalg.cg(A, b, M=precond_LO)
    ```

4. Post-processing to produce the sky maps. Note that `map_vector` from
   previous step contains maps in the form $[I_1, Q_1, U_1, I_2, Q_2, U_2, \dots]$.
   This has to be separated into I, Q, and U maps while taking care of the
   unobserved (masked) pixels.

    ```python
    # Separate I, Q, and U maps
    output_maps = separate_map_vectors(map_vector, processed_samples)
    
    # output_maps[0] --> I map
    # output_maps[1] --> Q map
    # output_maps[2] --> U map
    ```

## GLS map-making

`BrahMap` provides a simple wrapper function for GLS map-making as well:

```python
# Creating the inverse white noise covariance operator
inv_cov = brahmap.InvNoiseCovLO_Uncorrelated(
    diag=[...],         # Diagonal elements of the inverse of white noise covariance
                        # matrix

    dtype=np.float64,   # Numerical precision of the operator
)

# Performing the GLS map-making
gls_result = brahmap.compute_GLS_maps(
    npix=npix,                      # Number of pixels on which the map-making
                                    # has to be done

    pointings=pointings,            # A 1-d of pointing indices

    time_ordered_data=tod_array,    # A 1-d array of time-ordered-data

    pol_angles=pol_angles,          # A 1-d array containing the polarization angles
                                    # of the detectors

    inv_noise_cov_operator=inv_cov, # Inverse noise covariance operator

    dtype_float=np.float64,         # Numerical precision to be used in map-making
)
```

`gls_result` obtained above is an instance of the class `GLSResult`. The
output maps can be accessed from this object with `gls_result.GLS_maps`.

## GLS map-making with `litebird_sim` data

`BrahMap` is also integrated with the *LiteBIRD* simulation framework
`litebird_sim`. It provides a wrapper function for GLS map-making that uses a
list of `Observation` instances and a suitable inverse noise covariance
operator to produce the sky maps.

The MPI communicator used in map-making must be the one that contains
exclusively all the data needed for map-making. In case of `litebird_sim`, the
communicator `lbs.MPI_COMM_GRID.COMM_OBS_GRID` is a subset of
`lbs.MPI_COMM_WORLD`, and it excludes the MPI processes that do not contain
any detectors (and TODs). Therefore, it is a suitable communicator to be
used in map-making. Therefore, communicator used by
`BrahMap` must be updated as following before using any other `BrahMap`
function with `litebird_sim` data:
`brahmap.MPI_UTILS.update_communicator(comm=lbs.MPI_COMM_GRID.COMM_OBS_GRID)`

```python
# Creating the inverse white noise covariance operator
inv_cov = brahmap.InvNoiseCovLO_Uncorrelated(
    diag=[...],       # Diagonal elements of the inverse of white noise covariance
                      # matrix

    dtype=np.float64, # Numerical precision of the operator
)

# Performing the GLS map-making
gls_result = brahmap.LBSim_compute_GLS_maps(
    nside=nside,                    # Nside parameter for the output healpix map
    observations=sim.observations,  # List of observations from litebird_sim
    component="tod",                # TOD component to be used in map-making
    inv_noise_cov_operator=inv_cov, # Inverse noise covariance operator
    dtype_float=np.float64,         # Numerical precision to be used in map-making
)
```

`gls_result` obtained above is an instance of the class `LBSimGLSResult`. The
output maps can be accessed from this object with `gls_result.GLS_maps`.
