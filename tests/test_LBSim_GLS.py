import os
import tempfile
import pytest
import numpy as np
import healpy as hp

import brahmap

from mpi4py import MPI

comm_size_global = MPI.COMM_WORLD.size

litebird_sim = pytest.importorskip(
    modname="litebird_sim",
    minversion="0.13.0",
    reason="Couldn't import `litebird_sim` module",
)
import litebird_sim as lbs  # noqa: E402


@pytest.fixture(
    scope="module",
    params=[
        (np.float32, 1.5e-4, 1.0e-5),
        (np.float64, 1.5e-6, 1.0e-10),
    ],
    ids=["float32", "float64"],
)
def setup_lbsim(request):
    dtype_float, rtol, atol = request.param

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
                num_of_obs_per_detector = 1
            elif comm_size == 4:
                n_block_det = 2
                n_block_time = 2
                num_of_obs_per_detector = 3
            else:
                n_block_det = 1
                n_block_time = self.comm.Get_size()
                num_of_obs_per_detector = 2

            self.sim.create_observations(
                detectors=dets,
                num_of_obs_per_detector=num_of_obs_per_detector,
                n_blocks_det=n_block_det,
                n_blocks_time=n_block_time,
                split_list_over_processes=False,
                tod_dtype=self.dtype_float,
            )

            ### Compute pointings
            self.sim.prepare_pointings()

            ### Random maps
            np.random.seed(map_seed)
            self.npix = hp.nside2npix(self.nside)
            self.dummy_map = np.empty([3, self.npix], dtype=self.dtype_float)

            self.dummy_map[0] = np.random.uniform(low=-7.0, high=7.0, size=self.npix)
            self.dummy_map[1] = np.random.uniform(low=-5.0, high=5.0, size=self.npix)
            self.dummy_map[2] = np.random.uniform(low=-3.0, high=3.0, size=self.npix)

    lbsim_obj = lbsim_simulation(16, dtype_float)

    return lbsim_obj, rtol, atol


class TestLBSimGLS:
    def test_LBSim_compute_GLS_maps_I(self, setup_lbsim):
        lbsim_obj, rtol, atol = setup_lbsim

        ### Setting tod arrays zero
        for obs in lbsim_obj.sim.observations:
            obs.tod = np.zeros(obs.tod.shape, dtype=lbsim_obj.dtype_float)

        ### Scanning the sky
        lbs.scan_map_in_observations(
            lbsim_obj.sim.observations,
            maps=lbs.HealpixMap(
                values=np.array(
                    [
                        lbsim_obj.dummy_map[0],
                        np.zeros(lbsim_obj.npix, dtype=lbsim_obj.dtype_float),
                        np.zeros(lbsim_obj.npix, dtype=lbsim_obj.dtype_float),
                    ]
                ),
                nside=lbsim_obj.nside,
                coordinates=lbs.CoordinateSystem.Galactic,
            ),
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
        # np.testing.assert_equal(GLSresults.num_iterations, 1)

        input_map = np.ma.masked_array(
            lbsim_obj.dummy_map[0],
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

    def test_LBSim_compute_GLS_maps_QU(self, setup_lbsim):
        lbsim_obj, rtol, atol = setup_lbsim

        ### Setting tod arrays zero
        for obs in lbsim_obj.sim.observations:
            obs.tod = np.zeros(obs.tod.shape, lbsim_obj.dtype_float)

        ### Scanning the sky
        lbs.scan_map_in_observations(
            lbsim_obj.sim.observations,
            maps=lbs.HealpixMap(
                values=np.array(
                    [
                        np.zeros(lbsim_obj.npix, dtype=lbsim_obj.dtype_float),
                        lbsim_obj.dummy_map[1],
                        lbsim_obj.dummy_map[2],
                    ]
                ),
                nside=lbsim_obj.nside,
                coordinates=lbs.CoordinateSystem.Galactic,
            ),
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
        # np.testing.assert_equal(GLSresults.num_iterations, 1)

        input_map = np.ma.masked_array(
            lbsim_obj.dummy_map[1:], GLSresults.GLS_maps.mask, fill_value=hp.UNSEEN
        )

        np.testing.assert_allclose(GLSresults.GLS_maps, input_map, rtol, atol)

    def test_LBSim_compute_GLS_maps_IQU(self, setup_lbsim):
        lbsim_obj, rtol, atol = setup_lbsim

        ### Setting tod arrays zero
        for obs in lbsim_obj.sim.observations:
            obs.tod = np.zeros(obs.tod.shape, lbsim_obj.dtype_float)

        ### Scanning the sky
        lbs.scan_map_in_observations(
            lbsim_obj.sim.observations,
            maps=lbs.HealpixMap(
                values=lbsim_obj.dummy_map,
                nside=lbsim_obj.nside,
                coordinates=lbs.CoordinateSystem.Galactic,
            ),
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
        # np.testing.assert_equal(GLSresults.num_iterations, 1)

        input_map = np.ma.masked_array(
            lbsim_obj.dummy_map, GLSresults.GLS_maps.mask, fill_value=hp.UNSEEN
        )

        np.testing.assert_allclose(GLSresults.GLS_maps, input_map, rtol, atol)


class TestSharedMemLBSimGLS:
    def test_LBSim_compute_GLS_maps_shmem(self, setup_lbsim):
        lbsim_obj, rtol, atol = setup_lbsim

        # Setting tod arrays zero
        for obs in lbsim_obj.sim.observations:
            obs.tod = np.zeros(obs.tod.shape, lbsim_obj.dtype_float)

        # Scanning the sky
        lbs.scan_map_in_observations(
            lbsim_obj.sim.observations,
            maps=lbs.HealpixMap(
                values=lbsim_obj.dummy_map,
                nside=lbsim_obj.nside,
                coordinates=lbs.CoordinateSystem.Galactic,
            ),
        )

        # Run with standard
        GLSparams_std = brahmap.lbsim.LBSimGLSParameters(
            solver_type=brahmap.core.SolverType.IQU,
            output_coordinate_system=lbs.CoordinateSystem.Galactic,
            return_processed_samples=False,
            shmem_return_copy=True,
        )
        GLSresults_std = brahmap.lbsim.LBSim_compute_GLS_maps(
            nside=lbsim_obj.nside,
            observations=lbsim_obj.sim.observations,
            dtype_float=lbsim_obj.dtype_float,
            LBSim_gls_parameters=GLSparams_std,
            use_shared_memory=False,
        )

        # Run with shared memory (return_copy=True)
        GLSparams_shm = brahmap.lbsim.LBSimGLSParameters(
            solver_type=brahmap.core.SolverType.IQU,
            output_coordinate_system=lbs.CoordinateSystem.Galactic,
            return_processed_samples=False,
            shmem_return_copy=True,
        )
        GLSresults_shm = brahmap.lbsim.LBSim_compute_GLS_maps(
            nside=lbsim_obj.nside,
            observations=lbsim_obj.sim.observations,
            dtype_float=lbsim_obj.dtype_float,
            LBSim_gls_parameters=GLSparams_shm,
            use_shared_memory=True,
            nproc_reduce=2,
        )

        # Run with shared memory (return_copy=False)
        GLSparams_shm_nocopy = brahmap.lbsim.LBSimGLSParameters(
            solver_type=brahmap.core.SolverType.IQU,
            output_coordinate_system=lbs.CoordinateSystem.Galactic,
            return_processed_samples=False,
            shmem_return_copy=False,
        )
        GLSresults_shm_nocopy = brahmap.lbsim.LBSim_compute_GLS_maps(
            nside=lbsim_obj.nside,
            observations=lbsim_obj.sim.observations,
            dtype_float=lbsim_obj.dtype_float,
            LBSim_gls_parameters=GLSparams_shm_nocopy,
            use_shared_memory=True,
            nproc_reduce=2,
        )

        # Compare they are identical
        np.testing.assert_allclose(
            GLSresults_shm.GLS_maps, GLSresults_std.GLS_maps, rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(
            GLSresults_shm_nocopy.GLS_maps,
            GLSresults_std.GLS_maps,
            rtol=rtol,
            atol=atol,
        )


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
    pytest.main(
        [
            f"{__file__}::TestSharedMemLBSimGLS::test_LBSim_compute_GLS_maps_shmem",
            "-v",
            "-s",
        ]
    )
