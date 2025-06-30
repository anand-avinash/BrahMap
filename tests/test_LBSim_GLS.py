import os
import tempfile
import pytest
import numpy as np
import healpy as hp

import brahmap

from mpi4py import MPI

comm_size_global = MPI.COMM_WORLD.size

pytestmark = pytest.mark.skipif(
    comm_size_global > 4,
    reason="These tests are not meant for more than 4 MPI processes",
)

litebird_sim = pytest.importorskip(
    modname="litebird_sim",
    minversion="0.13.0",
    reason="Couldn't import `litebird_sim` module",
)
import litebird_sim as lbs  # noqa: E402


class lbsim_simulation:
    def __init__(self, nside, dtype_float):
        self.comm = lbs.MPI_COMM_WORLD

        try:
            nthreads = int(os.environ["OMP_NUM_THREADS"])
        except:  # noqa: E722
            nthreads = 1

        ### Mission params
        telescope = "MFT"
        channel = "M1-195"
        detector_list = [
            "001_002_030_00A_195_B",
            "001_002_029_45B_195_B",
            "001_002_015_15A_195_T",
            "001_002_047_00A_195_B",
        ]

        start_time = 51
        mission_time_days = 10
        detector_sampling_freq = 1

        ### Simulation params
        imo_version = "vPTEP"
        imo = lbs.Imo(flatfile_location=lbs.PTEP_IMO_LOCATION)
        sim_seed = 5132
        map_seed = 4664
        self.nside = nside
        self.dtype_float = dtype_float
        tmp_dir = tempfile.TemporaryDirectory()

        ### Initializing the Simulation
        self.sim = lbs.Simulation(
            base_path=tmp_dir.name,
            start_time=start_time,
            duration_s=mission_time_days * 24 * 60 * 60,
            random_seed=sim_seed,
            numba_threads=nthreads,
            numba_threading_layer="omp",
            mpi_comm=self.comm,
            imo=imo,
        )

        ### Instrument definition
        self.sim.set_instrument(
            lbs.InstrumentInfo.from_imo(
                imo,
                f"/releases/{imo_version}/satellite/{telescope}/instrument_info",
            )
        )

        ### Detector list
        dets = []
        for n_det in detector_list:
            det = lbs.DetectorInfo.from_imo(
                url=f"/releases/{imo_version}/satellite/{telescope}/{channel}/{n_det}/detector_info",
                imo=imo,
            )
            det.sampling_rate_hz = detector_sampling_freq
            dets.append(det)

        ### Scanning strategy
        self.sim.set_scanning_strategy(
            imo_url=f"/releases/{imo_version}/satellite/scanning_parameters/"
        )

        ### Create observations
        comm_size = self.comm.Get_size()
        if comm_size == 2:
            n_block_det = 2
            n_block_time = 1
        elif comm_size == 4:
            n_block_det = 2
            n_block_time = 2
        else:
            n_block_det = 1
            n_block_time = self.comm.Get_size()

        self.sim.create_observations(
            detectors=dets,
            num_of_obs_per_detector=3,
            n_blocks_det=n_block_det,
            n_blocks_time=n_block_time,
            split_list_over_processes=False,
            tod_dtype=self.dtype_float,
        )

        ### Ideal half-wave plate
        # self.sim.set_hwp(
        #     lbs.IdealHWP(
        #         self.sim.instrument.hwp_rpm * 2 * np.pi / 60,
        #     ),  # applies hwp rotation angle to the polarization angle
        # )

        # Grid communicator object
        self.MPI_COMM_GRID = lbs.MPI_COMM_GRID

        ### Compute pointings
        self.sim.prepare_pointings()
        # self.sim.precompute_pointings(pointings_dtype=self.dtype_float)

        ### Random maps
        np.random.seed(map_seed)
        self.npix = hp.nside2npix(self.nside)
        self.input_map = np.empty([3, self.npix], dtype=self.dtype_float)
        self.input_map[0] = np.random.uniform(low=-7.0, high=7.0, size=self.npix)
        self.input_map[1] = np.random.uniform(low=-5.0, high=5.0, size=self.npix)
        self.input_map[2] = np.random.uniform(low=-3.0, high=3.0, size=self.npix)


sim_float32 = lbsim_simulation(16, np.float32)
sim_float64 = lbsim_simulation(16, np.float64)


@pytest.mark.parametrize(
    "lbsim_obj, rtol, atol",
    [
        (sim_float32, 1.5e-4, 1.0e-5),
        (sim_float64, 1.5e-6, 1.0e-10),
    ],
)
class TestLBSimGLS:
    def test_LBSim_compute_GLS_maps_I(self, lbsim_obj, rtol, atol):
        brahmap.MPI_UTILS.update_communicator(lbsim_obj.MPI_COMM_GRID.COMM_OBS_GRID)

        ### Setting tod arrays zero
        for obs in lbsim_obj.sim.observations:
            obs.tod = np.zeros(obs.tod.shape, dtype=lbsim_obj.dtype_float)

        ### Scanning the sky
        lbs.scan_map_in_observations(
            lbsim_obj.sim.observations,
            maps=np.array(
                [
                    lbsim_obj.input_map[0],
                    np.zeros(lbsim_obj.npix),
                    np.zeros(lbsim_obj.npix),
                ],
                dtype=lbsim_obj.dtype_float,
            ),
            input_map_in_galactic=True,
        )

        GLSparams = brahmap.lbsim.LBSimGLSParameters(
            solver_type=brahmap.core.SolverType.I,
            output_coordinate_system=lbs.CoordinateSystem.Galactic,
            return_processed_samples=False,
        )

        for obs in lbsim_obj.sim.observations:
            obs.tod_new = obs.tod

        GLSresults = brahmap.lbsim.LBSim_compute_GLS_maps(
            nside=lbsim_obj.nside,
            observations=lbsim_obj.sim.observations,
            components=["tod", "tod_new"],
            dtype_float=lbsim_obj.dtype_float,
            LBSim_gls_parameters=GLSparams,
        )

        np.testing.assert_equal(GLSresults.convergence_status, True)
        np.testing.assert_equal(GLSresults.num_iterations, 1)

        input_map = np.ma.masked_array(
            lbsim_obj.input_map[0],
            GLSresults.GLS_maps.mask,
            fill_value=hp.UNSEEN,
            dtype=lbsim_obj.dtype_float,
        )

        np.testing.assert_allclose(
            GLSresults.GLS_maps[0],
            input_map * 2.0,
            rtol,
            atol,
        )

    def test_LBSim_compute_GLS_maps_QU(self, lbsim_obj, rtol, atol):
        brahmap.MPI_UTILS.update_communicator(lbsim_obj.MPI_COMM_GRID.COMM_OBS_GRID)

        ### Setting tod arrays zero
        for obs in lbsim_obj.sim.observations:
            obs.tod = np.zeros(obs.tod.shape, lbsim_obj.dtype_float)

        ### Scanning the sky
        lbs.scan_map_in_observations(
            lbsim_obj.sim.observations,
            maps=np.array(
                [
                    np.zeros(lbsim_obj.npix),
                    lbsim_obj.input_map[1],
                    lbsim_obj.input_map[2],
                ],
                dtype=lbsim_obj.dtype_float,
            ),
            input_map_in_galactic=True,
        )

        GLSparams = brahmap.lbsim.LBSimGLSParameters(
            solver_type=brahmap.core.SolverType.QU,
            output_coordinate_system=lbs.CoordinateSystem.Galactic,
            return_processed_samples=False,
        )

        GLSresults = brahmap.lbsim.LBSim_compute_GLS_maps(
            nside=lbsim_obj.nside,
            observations=lbsim_obj.sim.observations,
            dtype_float=lbsim_obj.dtype_float,
            LBSim_gls_parameters=GLSparams,
        )

        np.testing.assert_equal(GLSresults.convergence_status, True)
        np.testing.assert_equal(GLSresults.num_iterations, 1)

        input_map = np.ma.masked_array(
            lbsim_obj.input_map[1:], GLSresults.GLS_maps.mask, fill_value=hp.UNSEEN
        )

        np.testing.assert_allclose(GLSresults.GLS_maps, input_map, rtol, atol)

    def test_LBSim_compute_GLS_maps_IQU(self, lbsim_obj, rtol, atol):
        brahmap.MPI_UTILS.update_communicator(lbsim_obj.MPI_COMM_GRID.COMM_OBS_GRID)

        ### Setting tod arrays zero
        for obs in lbsim_obj.sim.observations:
            obs.tod = np.zeros(obs.tod.shape, lbsim_obj.dtype_float)

        ### Scanning the sky
        lbs.scan_map_in_observations(
            lbsim_obj.sim.observations,
            maps=lbsim_obj.input_map,
            input_map_in_galactic=True,
        )

        GLSparams = brahmap.lbsim.LBSimGLSParameters(
            solver_type=brahmap.core.SolverType.IQU,
            output_coordinate_system=lbs.CoordinateSystem.Galactic,
            return_processed_samples=False,
        )

        GLSresults = brahmap.lbsim.LBSim_compute_GLS_maps(
            nside=lbsim_obj.nside,
            observations=lbsim_obj.sim.observations,
            dtype_float=lbsim_obj.dtype_float,
            LBSim_gls_parameters=GLSparams,
        )

        np.testing.assert_equal(GLSresults.convergence_status, True)
        np.testing.assert_equal(GLSresults.num_iterations, 1)

        input_map = np.ma.masked_array(
            lbsim_obj.input_map, GLSresults.GLS_maps.mask, fill_value=hp.UNSEEN
        )

        np.testing.assert_allclose(GLSresults.GLS_maps, input_map, rtol, atol)


if __name__ == "__main__":
    pytest.main(
        [f"{__file__}::TestLBSimGLS::test_LBSim_compute_GLS_maps_I", "-v", "-s"]
    )
    pytest.main(
        [f"{__file__}::TestLBSimGLS::test_LBSim_compute_GLS_maps_QU", "-v", "-s"]
    )
    pytest.main(
        [f"{__file__}::TestLBSimGLS::test_LBSim_compute_GLS_maps_IQU", "-v", "-s"]
    )
