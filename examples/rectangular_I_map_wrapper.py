import numpy as np
import brahmap


###########################################################
####### Producing the input maps, pointings and TOD #######
###########################################################


### Random number generator
seed = 5455
rng = np.random.default_rng(seed=seed)


### Numerical precisions
dtype_int = np.int32  # Numerical precision for pointing indices
dtype_float = np.float64  # Numerical precision for TOD and output maps


### Map dimensions
pix_x = 16
pix_y = 32
npix = pix_x * pix_y


### Number of samples
nsamples_global = npix * 6  # Global number of samples

div, rem = divmod(nsamples_global, brahmap.MPI_UTILS.size)
nsamples = div + (brahmap.MPI_UTILS.rank < rem)  # Local number of samples


### Number of bad samples
nbad_samples_global = npix  # Global number of bad samples

div, rem = divmod(nbad_samples_global, brahmap.MPI_UTILS.size)
nbad_samples = div + (brahmap.MPI_UTILS.rank < rem)  # Local number of bad samples


### Generating random pointing indices
pointings = rng.integers(low=0, high=npix, size=nsamples, dtype=dtype_int)


### Generating pointing flags
# Samples marked with flag `False` are considered as the bad samples and are excluded from map-making
pointings_flag = np.ones(nsamples, dtype=bool)
bad_samples = rng.integers(low=0, high=nsamples, size=nbad_samples)
pointings_flag[bad_samples] = False


### Generating random input map
if brahmap.MPI_UTILS.rank == 0:
    input_map = rng.uniform(low=-10.0, high=10.0, size=npix).astype(dtype=dtype_float)
else:
    input_map = None


### Broadcasting input map to all MPI processes
input_map = brahmap.MPI_UTILS.comm.bcast(input_map, 0)


### Scanning the input map
tod = np.zeros(nsamples, dtype=dtype_float)

for idx, pixel in enumerate(pointings):
    tod[idx] += input_map[pixel]


#####################################
####### Producing the GLS map #######
#####################################


### Creating an inverse noise covariance operator (unit diagonal operator in this case)
inv_cov = brahmap.InvNoiseCovLO_Uncorrelated(
    diag=np.ones(nsamples, dtype=dtype_float), dtype=dtype_float
)


### Defining the parameters used for GLS map-making
# Since we are solving only for I map, it is necessary to set the solver type to I
gls_params = brahmap.GLSParameters(
    solver_type=brahmap.SolverType.I,
)


### Computing the GLS map
gls_result = brahmap.compute_GLS_maps(
    npix=npix,
    pointings=pointings,
    time_ordered_data=tod,
    pointings_flag=pointings_flag,
    inv_noise_cov_operator=inv_cov,
    dtype_float=dtype_float,
    gls_parameters=gls_params,
)

# The output `gls_result` in the previous cell is an instance of
# `GLSResult`. The output maps can be accessed from it with
# `gls_result.GLS_maps`.

#######################################################
####### The output map is produced successfully #######
#######################################################


##############################
### Validating the results ###
##############################

np.testing.assert_allclose(input_map, gls_result.GLS_maps[0])
