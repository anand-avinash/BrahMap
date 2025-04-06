import numpy as np
import scipy as sp

import healpy as hp
import brahmap


###########################################################
####### Producing the input maps, pointings and TOD #######
###########################################################


### Random number generator
seed = 5455 + brahmap.MPI_UTILS.rank
rng = np.random.default_rng(seed=seed)


### Numerical precisions
dtype_int = np.int32  # Numerical precision for pointing indices
dtype_float = np.float64  # Numerical precision for TOD and output maps


### Map dimensions
nside = 128
npix = hp.nside2npix(nside)


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


### Generating random detector polarization angles
pol_angles = rng.uniform(low=-np.pi / 2.0, high=np.pi / 2.0, size=nsamples).astype(
    dtype=dtype_float
)


### Generating pointing flags
# Samples marked with flag `False` are considered as the bad samples and are excluded from map-making
pointings_flag = np.ones(nsamples, dtype=bool)
bad_samples = rng.integers(low=0, high=nsamples, size=nbad_samples)
pointings_flag[bad_samples] = False


### Generating random input maps
if brahmap.MPI_UTILS.rank == 0:
    input_maps = np.empty((2, npix), dtype=dtype_float)
    input_maps[0] = rng.uniform(low=-10.0, high=10.0, size=npix)
    input_maps[1] = rng.uniform(low=-3.0, high=3.0, size=npix)
else:
    input_maps = None


### Broadcasting input maps to all MPI processes
input_maps = brahmap.MPI_UTILS.comm.bcast(input_maps, 0)


### Scanning the input maps
tod = np.zeros(nsamples, dtype=dtype_float)

for idx, pixel in enumerate(pointings):
    tod[idx] += input_maps[0][pixel] * np.cos(2.0 * pol_angles[idx])
    tod[idx] += input_maps[1][pixel] * np.sin(2.0 * pol_angles[idx])


#####################################
####### Producing the GLS map #######
#####################################


### Creating an inverse noise covariance operator (unit diagonal operator in this case)
# Note that each MPI process has its own instance of local linear operator,
# but it performs the matrix-vector multiplication in global space. Same
# applies for all the linear operators we define in this script.
inv_cov = brahmap.InvNoiseCovLO_Diagonal(
    size=nsamples,
    input=1.0,
    dtype=dtype_float,
)

### Processing the pointing information
processed_samples = brahmap.ProcessTimeSamples(
    npix=npix,
    pointings=pointings,
    pointings_flag=pointings_flag,
    solver_type=brahmap.SolverType.QU,
    pol_angles=pol_angles,
    noise_weights=inv_cov.diag,
    dtype_float=dtype_float,
)


### Creating pointing operator
pointing_LO = brahmap.PointingLO(processed_samples)


### Creating block-diagonal preconditioner
precond_LO = brahmap.BlockDiagonalPreconditionerLO(processed_samples)


### Creating operators for linear equation A.x=b
A = pointing_LO.T * inv_cov * pointing_LO
b = pointing_LO.T * inv_cov * tod


### Solving GLS equation using pcg
# We are solving for x in the linear equation A.x = b
# Note that our linear operators are distributed across multiple MPI
# processes. As a result, we should compute the PCG solution across the entire
# MPI communicator. Sparse conjugate gradient solver of SciPy, however,
# computes some quantities locally that makes it unsuitable for our case. With
# BrahMap, it is possible to change the behavior of internal computations in
# `scipy.sparse.linalg.cg()` to make all the computations global. It can be
# done by calling the sparse conjugate solver within the context manager
# `brahmap.utilities.modify_numpy_context()`.
with brahmap.utilities.modify_numpy_context():
    map_vector, _ = sp.sparse.linalg.cg(A=A, b=b, M=precond_LO)

# We provide the above example of using sparse conjugate gradient solver with
# parallelization for complete information to the user. BrahMap, in fact, also
# provides a native sparse conjugate gradient solver that can be used instead
# of SciPy solver without any context manager. It can be used as:
# `map_vector, _ = brahmap.math.cg(A=A, b=b, M=precond_LO)`


### Producing output map masked for bad pixels
# `map_vector` obtained in the previous step contains the output maps in form
# [Q_1, U_1, Q_2, U_2, ...] only for the pixels that are being used in
# map-making. To produce the final output maps, we need to separate the Q and U
# maps, and mask them appropriately
output_maps = brahmap.separate_map_vectors(
    map_vector=map_vector, processed_samples=processed_samples
)


#########################################################
####### The output maps are produced successfully #######
#########################################################


##############################
### Validating the results ###
##############################

np.testing.assert_allclose(input_maps, output_maps)
