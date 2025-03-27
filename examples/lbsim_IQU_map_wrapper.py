import tempfile
import numpy as np

import litebird_sim as lbs
import brahmap


###############################################################################
####### Producing the input maps, pointings and TODs using litebird_sim #######
###############################################################################


### Mission parameters
telescope = "MFT"
channel = "M1-195"
detectors = [
    "001_002_030_00A_195_B",
    "001_002_029_45B_195_B",
    "001_002_015_15A_195_T",
    "001_002_047_00A_195_B",
]
start_time = 51
mission_time_days = 30
detector_sampling_freq = 1


### Simulation parameters
nside = 128
random_seed = 45
imo_version = "vPTEP"
imo = lbs.Imo(flatfile_location=lbs.PTEP_IMO_LOCATION)
dtype_float = np.float64
tmp_dir = tempfile.TemporaryDirectory()


### Getting the MPI communicator
comm = lbs.MPI_COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


### Initializing the simulation
sim = lbs.Simulation(
    random_seed=random_seed,
    base_path=tmp_dir.name,
    name="brahmap_example",
    mpi_comm=comm,
    start_time=start_time,
    duration_s=mission_time_days * 24 * 60 * 60.0,
    imo=imo,
)


### Instrument definition
sim.set_instrument(
    lbs.InstrumentInfo.from_imo(
        imo,
        f"/releases/{imo_version}/satellite/{telescope}/instrument_info",
    )
)


### Detector list
detector_list = []
for n_det in detectors:
    det = lbs.DetectorInfo.from_imo(
        url=f"/releases/{imo_version}/satellite/{telescope}/{channel}/{n_det}/detector_info",
        imo=imo,
    )
    det.sampling_rate_hz = detector_sampling_freq
    detector_list.append(det)


### Scanning strategy
sim.set_scanning_strategy(
    imo_url=f"/releases/{imo_version}/satellite/scanning_parameters/"
)


### Create observations
sim.create_observations(
    detectors=detector_list,
    num_of_obs_per_detector=3,
    n_blocks_det=1,
    n_blocks_time=comm_size,  # Non-zero number of time blocks for example
    split_list_over_processes=False,
    tod_dtype=dtype_float,
)


### Prepare pointings
sim.prepare_pointings()


### Compute pointings (optional)
sim.precompute_pointings(pointings_dtype=dtype_float)


### Channel info
ch_info = []
n_ch_info = lbs.FreqChannelInfo.from_imo(
    imo,
    f"/releases/{imo_version}/satellite/{telescope}/{channel}/channel_info",
)
ch_info.append(n_ch_info)


### Producing the input CMB maps
mbs_params = lbs.MbsParameters(
    make_cmb=True,
    make_fg=False,
    seed_cmb=1,
    gaussian_smooth=True,
    bandpass_int=False,
    nside=nside,
    units="uK_CMB",
    maps_in_ecliptic=False,
    output_string="mbs_cmb_lens",
)

mbs_obj = lbs.Mbs(
    simulation=sim,
    parameters=mbs_params,
    channel_list=ch_info,
)

if comm.rank == 0:
    input_maps = mbs_obj.run_all()
else:
    input_maps = None

# Distributing the maps to all MPI processes
input_maps = lbs.MPI_COMM_WORLD.bcast(input_maps, 0)


### Scanning the sky
lbs.scan_map_in_observations(
    sim.observations,
    maps=input_maps[0][channel],
)


#####################################
####### Producing the GLS map #######
#####################################


### Updating the BrahMap environment
# The MPI communicator used in map-making must be the one that contains
# exclusively all the data needed for map-making. In case of litebird_sim, the
# communicator `lbs.MPI_COMM_GRID.COMM_OBS_GRID` is a subset of
# `lbs.MPI_COMM_WORLD`, and it excludes the MPI processes that do not contain
# any detectors (and TODs). Therefore, it is a suitable communicator to be
# used in map-making. The default communicator (MPI.COMM_WORLD) used by
# BrahMap can be updated to a new one using a dedicated functions as shown below:
brahmap.MPI_UTILS.update_communicator(comm=lbs.MPI_COMM_GRID.COMM_OBS_GRID)


### Creating an inverse noise covariance operator (unit diagonal operator in this case)
inv_cov = brahmap.LBSim_InvNoiseCovLO_UnCorr(sim.observations)


### Defining the parameters used for GLS map-making
# Since we are solving only for I, Q, and U maps, it is necessary to set the solver type to IQU
gls_params = brahmap.LBSimGLSParameters(
    solver_type=brahmap.SolverType.IQU,  # default value
)


### Computing the GLS maps
gls_result = brahmap.LBSim_compute_GLS_maps(
    nside=nside,
    observations=sim.observations,
    component="tod",
    inv_noise_cov_operator=inv_cov,
    dtype_float=dtype_float,
    LBSim_gls_parameters=gls_params,
)

# The output `gls_result` in the previous cell is an instance of
# `LBSimGLSResult`. The output maps can be accessed from it with
# `gls_result.GLS_maps`.

#########################################################
####### The output maps are produced successfully #######
#########################################################


##############################
### Validating the results ###
##############################

np.testing.assert_allclose(input_maps[0][channel], gls_result.GLS_maps)
