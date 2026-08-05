import numpy as np
import pytest
import brahmap
from brahmap._extensions import (
    compute_weights,
    compute_weights_shared,
    repixelize,
    PointingLO_tools,
    BlkDiagPrecondLO_tools,
)
from brahmap.math import linalg_tools, unary_functions
from .benchmark_utils import setup_common_data


@pytest.fixture(scope="module")
def data(bench_params):
    return setup_common_data(params=bench_params)


# --- compute_weights.cpp ---
@pytest.mark.benchmark(group="extensions::compute_weights")
class TestComputeWeights:
    def test_bench_compute_weights_pol_I(self, mpi_benchmark, data):
        npix, nsamples, dtype_int, dtype_float = (
            data["npix"],
            data["nsamples"],
            data["dtype_int"],
            data["dtype_float"],
        )
        hit_counts = np.zeros(npix, dtype=dtype_int)
        weighted_counts = np.zeros(npix, dtype=dtype_float)
        observed_pixels = np.zeros(npix, dtype=dtype_int)
        old2new_pixel = np.zeros(npix, dtype=dtype_int)
        pixel_flag = np.zeros(npix, dtype=bool)

        def setup():
            hit_counts.fill(0)
            weighted_counts.fill(0)
            observed_pixels.fill(0)
            old2new_pixel.fill(0)
            pixel_flag.fill(False)
            return (
                npix,
                nsamples,
                data["pointings"],
                data["pointings_flag"],
                data["noise_weights"],
                hit_counts,
                weighted_counts,
                observed_pixels,
                old2new_pixel,
                pixel_flag,
                brahmap.MPI_UTILS.comm,
            ), {}

        mpi_benchmark(
            compute_weights.compute_weights_pol_I,
            setup=setup,
        )

    def test_bench_compute_weights_pol_QU(self, mpi_benchmark, data):
        npix, nsamples, dtype_int, dtype_float = (
            data["npix"],
            data["nsamples"],
            data["dtype_int"],
            data["dtype_float"],
        )
        hit_counts = np.zeros(npix, dtype=dtype_int)
        weighted_counts = np.zeros(npix, dtype=dtype_float)
        sin2phi = np.zeros(nsamples, dtype=dtype_float)
        cos2phi = np.zeros(nsamples, dtype=dtype_float)
        weighted_sin_sq = np.zeros(npix, dtype=dtype_float)
        weighted_cos_sq = np.zeros(npix, dtype=dtype_float)
        weighted_sincos = np.zeros(npix, dtype=dtype_float)
        one_over_determinant = np.zeros(npix, dtype=dtype_float)

        def setup():
            hit_counts.fill(0)
            weighted_counts.fill(0)
            weighted_sin_sq.fill(0)
            weighted_cos_sq.fill(0)
            weighted_sincos.fill(0)
            one_over_determinant.fill(0)
            return (
                npix,
                nsamples,
                data["pointings"],
                data["pointings_flag"],
                data["noise_weights"],
                data["pol_angles"],
                hit_counts,
                weighted_counts,
                sin2phi,
                cos2phi,
                weighted_sin_sq,
                weighted_cos_sq,
                weighted_sincos,
                one_over_determinant,
                brahmap.MPI_UTILS.comm,
            ), {}

        mpi_benchmark(
            compute_weights.compute_weights_pol_QU,
            setup=setup,
        )

    def test_bench_compute_weights_pol_IQU(self, mpi_benchmark, data):
        npix, nsamples, dtype_int, dtype_float = (
            data["npix"],
            data["nsamples"],
            data["dtype_int"],
            data["dtype_float"],
        )
        hit_counts = np.zeros(npix, dtype=dtype_int)
        weighted_counts = np.zeros(npix, dtype=dtype_float)
        sin2phi = np.zeros(nsamples, dtype=dtype_float)
        cos2phi = np.zeros(nsamples, dtype=dtype_float)
        weighted_sin_sq = np.zeros(npix, dtype=dtype_float)
        weighted_cos_sq = np.zeros(npix, dtype=dtype_float)
        weighted_sincos = np.zeros(npix, dtype=dtype_float)
        weighted_sin = np.zeros(npix, dtype=dtype_float)
        weighted_cos = np.zeros(npix, dtype=dtype_float)
        one_over_determinant = np.zeros(npix, dtype=dtype_float)

        def setup():
            hit_counts.fill(0)
            weighted_counts.fill(0)
            weighted_sin_sq.fill(0)
            weighted_cos_sq.fill(0)
            weighted_sincos.fill(0)
            weighted_sin.fill(0)
            weighted_cos.fill(0)
            one_over_determinant.fill(0)
            return (
                npix,
                nsamples,
                data["pointings"],
                data["pointings_flag"],
                data["noise_weights"],
                data["pol_angles"],
                hit_counts,
                weighted_counts,
                sin2phi,
                cos2phi,
                weighted_sin_sq,
                weighted_cos_sq,
                weighted_sincos,
                weighted_sin,
                weighted_cos,
                one_over_determinant,
                brahmap.MPI_UTILS.comm,
            ), {}

        mpi_benchmark(
            compute_weights.compute_weights_pol_IQU,
            setup=setup,
        )

    def test_bench_get_pixel_mask_pol(self, mpi_benchmark, data):
        npix, dtype_int, dtype_float, rng = (
            data["npix"],
            data["dtype_int"],
            data["dtype_float"],
            data["rng"],
        )
        hit_counts = rng.integers(0, 10, npix, dtype=dtype_int)
        one_over_determinant = rng.random(npix).astype(dtype=dtype_float)
        observed_pixels = np.zeros(npix, dtype=dtype_int)
        old2new_pixel = np.zeros(npix, dtype=dtype_int)
        pixel_flag = np.zeros(npix, dtype=bool)

        mpi_benchmark(
            compute_weights.get_pixel_mask_pol,
            3,
            npix,
            1e-5,
            hit_counts,
            one_over_determinant,
            observed_pixels,
            old2new_pixel,
            pixel_flag,
        )


# --- compute_weights_shared.cpp ---
@pytest.mark.benchmark(group="extensions::compute_weights_shared")
class TestComputeWeightsShared:
    def test_bench_compute_weights_shmem_pol_I(
        self,
        mpi_benchmark,
        data,
        nproc_reduce,
    ):
        npix, nsamples, dtype_int, dtype_float = (
            data["npix"],
            data["nsamples"],
            data["dtype_int"],
            data["dtype_float"],
        )
        mgr = brahmap.mpi.SharedMemoryManager(
            base_comm=brahmap.MPI_UTILS.comm,
            nproc_reduce=nproc_reduce,
            node_root=0,
        )
        hit_counts, win_hit_counts = mgr.alloc_shared_zeros_node(
            npix,
            dtype_int,
        )
        weighted_counts, win_weighted_counts = mgr.alloc_shared_zeros_node(
            npix,
            dtype_float,
        )
        observed_pixels, _ = mgr.alloc_shared_zeros_node(
            npix,
            dtype=dtype_int,
        )
        old2new_pixel, _ = mgr.alloc_shared_zeros_node(
            npix,
            dtype=dtype_int,
        )
        pixel_flag, _ = mgr.alloc_shared_zeros_node(
            npix,
            dtype=bool,
        )

        mgr.fence_comm_all(mgr.node_comm)

        def setup():
            hit_counts.fill(0)
            weighted_counts.fill(0)
            return (
                npix,
                nsamples,
                data["pointings"],
                data["pointings_flag"],
                data["noise_weights"],
                hit_counts,
                win_hit_counts,
                weighted_counts,
                win_weighted_counts,
                observed_pixels,
                old2new_pixel,
                pixel_flag,
                mgr.node_root,
                mgr.grp_reduce,
                mgr.tree_grp_comm,
                mgr.tree_grp_root_comm,
                mgr.node_comm,
                mgr.node_root_comm,
            ), {}

        mpi_benchmark(
            compute_weights_shared.compute_weights_shmem_pol_I,
            setup=setup,
        )
        mgr.free_shared_arrays_all()

    def test_bench_compute_weights_shmem_pol_QU(
        self,
        mpi_benchmark,
        data,
        nproc_reduce,
    ):
        npix, nsamples, dtype_int, dtype_float = (
            data["npix"],
            data["nsamples"],
            data["dtype_int"],
            data["dtype_float"],
        )
        mgr = brahmap.mpi.SharedMemoryManager(
            base_comm=brahmap.MPI_UTILS.comm,
            nproc_reduce=nproc_reduce,
            node_root=0,
        )
        hit_counts, win_hit_counts = mgr.alloc_shared_zeros_node(
            npix,
            dtype_int,
        )
        weighted_counts, win_weighted_counts = mgr.alloc_shared_zeros_node(
            npix,
            dtype_float,
        )
        sin2phi = np.zeros(nsamples, dtype=dtype_float)
        cos2phi = np.zeros(nsamples, dtype=dtype_float)
        weighted_sin_sq, win_weighted_sin_sq = mgr.alloc_shared_zeros_node(
            npix,
            dtype_float,
        )
        weighted_cos_sq, win_weighted_cos_sq = mgr.alloc_shared_zeros_node(
            npix,
            dtype_float,
        )
        weighted_sincos, win_weighted_sincos = mgr.alloc_shared_zeros_node(
            npix,
            dtype_float,
        )
        one_over_determinant, _ = mgr.alloc_shared_zeros_node(
            npix,
            dtype=dtype_float,
        )

        mgr.fence_comm_all(mgr.node_comm)

        def setup():
            hit_counts.fill(0)
            weighted_counts.fill(0)
            weighted_sin_sq.fill(0)
            weighted_cos_sq.fill(0)
            weighted_sincos.fill(0)
            one_over_determinant.fill(0)
            return (
                npix,
                nsamples,
                data["pointings"],
                data["pointings_flag"],
                data["noise_weights"],
                data["pol_angles"],
                hit_counts,
                win_hit_counts,
                weighted_counts,
                win_weighted_counts,
                sin2phi,
                cos2phi,
                weighted_sin_sq,
                win_weighted_sin_sq,
                weighted_cos_sq,
                win_weighted_cos_sq,
                weighted_sincos,
                win_weighted_sincos,
                one_over_determinant,
                mgr.node_root,
                mgr.grp_reduce,
                mgr.tree_grp_comm,
                mgr.tree_grp_root_comm,
                mgr.node_comm,
                mgr.node_root_comm,
            ), {}

        mpi_benchmark(
            compute_weights_shared.compute_weights_shmem_pol_QU,
            setup=setup,
        )
        mgr.free_shared_arrays_all()

    def test_bench_compute_weights_shmem_pol_IQU(
        self,
        mpi_benchmark,
        data,
        nproc_reduce,
    ):
        npix, nsamples, dtype_int, dtype_float = (
            data["npix"],
            data["nsamples"],
            data["dtype_int"],
            data["dtype_float"],
        )
        mgr = brahmap.mpi.SharedMemoryManager(
            base_comm=brahmap.MPI_UTILS.comm,
            nproc_reduce=nproc_reduce,
            node_root=0,
        )
        hit_counts, win_hit_counts = mgr.alloc_shared_zeros_node(
            npix,
            dtype_int,
        )
        weighted_counts, win_weighted_counts = mgr.alloc_shared_zeros_node(
            npix,
            dtype_float,
        )
        sin2phi = np.zeros(nsamples, dtype=dtype_float)
        cos2phi = np.zeros(nsamples, dtype=dtype_float)
        weighted_sin_sq, win_weighted_sin_sq = mgr.alloc_shared_zeros_node(
            npix,
            dtype_float,
        )
        weighted_cos_sq, win_weighted_cos_sq = mgr.alloc_shared_zeros_node(
            npix,
            dtype_float,
        )
        weighted_sincos, win_weighted_sincos = mgr.alloc_shared_zeros_node(
            npix,
            dtype_float,
        )
        weighted_sin, win_weighted_sin = mgr.alloc_shared_zeros_node(
            npix,
            dtype_float,
        )
        weighted_cos, win_weighted_cos = mgr.alloc_shared_zeros_node(
            npix,
            dtype_float,
        )
        one_over_determinant, _ = mgr.alloc_shared_zeros_node(
            npix,
            dtype=dtype_float,
        )

        mgr.fence_comm_all(mgr.node_comm)

        def setup():
            hit_counts.fill(0)
            weighted_counts.fill(0)
            weighted_sin_sq.fill(0)
            weighted_cos_sq.fill(0)
            weighted_sincos.fill(0)
            weighted_sin.fill(0)
            weighted_cos.fill(0)
            one_over_determinant.fill(0)
            return (
                npix,
                nsamples,
                data["pointings"],
                data["pointings_flag"],
                data["noise_weights"],
                data["pol_angles"],
                hit_counts,
                win_hit_counts,
                weighted_counts,
                win_weighted_counts,
                sin2phi,
                cos2phi,
                weighted_sin_sq,
                win_weighted_sin_sq,
                weighted_cos_sq,
                win_weighted_cos_sq,
                weighted_sincos,
                win_weighted_sincos,
                weighted_sin,
                win_weighted_sin,
                weighted_cos,
                win_weighted_cos,
                one_over_determinant,
                mgr.node_root,
                mgr.grp_reduce,
                mgr.tree_grp_comm,
                mgr.tree_grp_root_comm,
                mgr.node_comm,
                mgr.node_root_comm,
            ), {}

        mpi_benchmark(
            compute_weights_shared.compute_weights_shmem_pol_IQU,
            setup=setup,
        )
        mgr.free_shared_arrays_all()


# --- repixelization.cpp ---
@pytest.mark.benchmark(group="extensions::repixelize")
class TestRepixelize:
    def test_bench_repixelize_pol_I(self, mpi_benchmark, data):
        npix, rng, dtype_int, dtype_float = (
            data["npix"],
            data["rng"],
            data["dtype_int"],
            data["dtype_float"],
        )
        new_npix = npix * 2 // 3
        observed_pixels = np.sort(rng.choice(npix, new_npix, replace=False)).astype(
            dtype=dtype_int
        )
        hit_counts = rng.integers(0, 10, npix, dtype=dtype_int)
        weighted_counts = rng.random(npix).astype(dtype=dtype_float)

        mpi_benchmark(
            repixelize.repixelize_pol_I,
            new_npix,
            observed_pixels,
            hit_counts,
            weighted_counts,
        )

    def test_bench_repixelize_pol_QU(self, mpi_benchmark, data):
        npix, rng, dtype_int, dtype_float = (
            data["npix"],
            data["rng"],
            data["dtype_int"],
            data["dtype_float"],
        )
        new_npix = npix * 2 // 3
        observed_pixels = np.sort(rng.choice(npix, new_npix, replace=False)).astype(
            dtype=dtype_int
        )
        hit_counts = rng.integers(0, 10, npix, dtype=dtype_int)
        weighted_counts = rng.random(npix).astype(dtype=dtype_float)
        weighted_sin_sq = rng.random(npix).astype(dtype=dtype_float)
        weighted_cos_sq = rng.random(npix).astype(dtype=dtype_float)
        weighted_sincos = rng.random(npix).astype(dtype=dtype_float)
        one_over_determinant = rng.random(npix).astype(dtype=dtype_float)

        mpi_benchmark(
            repixelize.repixelize_pol_QU,
            new_npix,
            observed_pixels,
            hit_counts,
            weighted_counts,
            weighted_sin_sq,
            weighted_cos_sq,
            weighted_sincos,
            one_over_determinant,
        )

    def test_bench_repixelize_pol_IQU(self, mpi_benchmark, data):
        npix, rng, dtype_int, dtype_float = (
            data["npix"],
            data["rng"],
            data["dtype_int"],
            data["dtype_float"],
        )
        new_npix = npix * 2 // 3
        observed_pixels = np.sort(rng.choice(npix, new_npix, replace=False)).astype(
            dtype=dtype_int
        )
        hit_counts = rng.integers(0, 10, npix, dtype=dtype_int)
        weighted_counts = rng.random(npix).astype(dtype=dtype_float)
        weighted_sin_sq = rng.random(npix).astype(dtype=dtype_float)
        weighted_cos_sq = rng.random(npix).astype(dtype=dtype_float)
        weighted_sincos = rng.random(npix).astype(dtype=dtype_float)
        weighted_sin = rng.random(npix).astype(dtype=dtype_float)
        weighted_cos = rng.random(npix).astype(dtype=dtype_float)
        one_over_determinant = rng.random(npix).astype(dtype=dtype_float)

        mpi_benchmark(
            repixelize.repixelize_pol_IQU,
            new_npix,
            observed_pixels,
            hit_counts,
            weighted_counts,
            weighted_sin_sq,
            weighted_cos_sq,
            weighted_sincos,
            weighted_sin,
            weighted_cos,
            one_over_determinant,
        )

    def test_bench_flag_bad_pixel_samples(self, mpi_benchmark, data):
        pixel_flag = data["rng"].choice([True, False], data["npix"])
        old2new_pixel = np.arange(data["npix"], dtype=data["dtype_int"])
        pointings = data["pointings"].copy()
        pointings_flag = data["pointings_flag"].copy()

        mpi_benchmark(
            repixelize.flag_bad_pixel_samples,
            data["nsamples"],
            pixel_flag,
            old2new_pixel,
            pointings,
            pointings_flag,
        )


# --- PointingLO_tools ---
@pytest.mark.benchmark(group="extensions::PointingLO")
class TestPointingLO:
    def test_bench_PLO_mult_I(self, mpi_benchmark, data):
        nsamples, npix, dtype_float, rng = (
            data["nsamples"],
            data["npix"],
            data["dtype_float"],
            data["rng"],
        )
        vec = rng.random(npix).astype(dtype_float)
        prod = np.zeros(nsamples, dtype=dtype_float)

        mpi_benchmark(
            PointingLO_tools.PLO_mult_I,
            nsamples,
            data["pointings"],
            data["pointings_flag"],
            vec,
            prod,
        )

    def test_bench_PLO_rmult_I(self, mpi_benchmark, data):
        nsamples, npix, dtype_float, rng = (
            data["nsamples"],
            data["npix"],
            data["dtype_float"],
            data["rng"],
        )
        vec = rng.random(nsamples).astype(dtype_float)
        prod = np.zeros(npix, dtype=dtype_float)

        def setup():
            prod.fill(0)
            return (
                npix,
                nsamples,
                data["pointings"],
                data["pointings_flag"],
                vec,
                prod,
                brahmap.MPI_UTILS.comm,
            ), {}

        mpi_benchmark(
            PointingLO_tools.PLO_rmult_I,
            setup=setup,
        )

    def test_bench_PLO_mult_QU(self, mpi_benchmark, data):
        nsamples, npix, dtype_float, rng = (
            data["nsamples"],
            data["npix"],
            data["dtype_float"],
            data["rng"],
        )
        vec = rng.random(2 * npix).astype(dtype_float)
        prod = np.zeros(nsamples, dtype=dtype_float)
        sin2phi = rng.random(nsamples).astype(dtype_float)
        cos2phi = rng.random(nsamples).astype(dtype_float)

        mpi_benchmark(
            PointingLO_tools.PLO_mult_QU,
            nsamples,
            data["pointings"],
            data["pointings_flag"],
            sin2phi,
            cos2phi,
            vec,
            prod,
        )

    def test_bench_PLO_rmult_QU(self, mpi_benchmark, data):
        nsamples, npix, dtype_float, rng = (
            data["nsamples"],
            data["npix"],
            data["dtype_float"],
            data["rng"],
        )
        vec = rng.random(nsamples).astype(dtype_float)
        prod = np.zeros(2 * npix, dtype=dtype_float)
        sin2phi = rng.random(nsamples).astype(dtype_float)
        cos2phi = rng.random(nsamples).astype(dtype_float)

        def setup():
            prod.fill(0)
            return (
                npix,
                nsamples,
                data["pointings"],
                data["pointings_flag"],
                sin2phi,
                cos2phi,
                vec,
                prod,
                brahmap.MPI_UTILS.comm,
            ), {}

        mpi_benchmark(
            PointingLO_tools.PLO_rmult_QU,
            setup=setup,
        )

    def test_bench_PLO_mult_IQU(self, mpi_benchmark, data):
        nsamples, npix, dtype_float, rng = (
            data["nsamples"],
            data["npix"],
            data["dtype_float"],
            data["rng"],
        )
        vec = rng.random(3 * npix).astype(dtype_float)
        prod = np.zeros(nsamples, dtype=dtype_float)
        sin2phi = rng.random(nsamples).astype(dtype_float)
        cos2phi = rng.random(nsamples).astype(dtype_float)

        mpi_benchmark(
            PointingLO_tools.PLO_mult_IQU,
            nsamples,
            data["pointings"],
            data["pointings_flag"],
            sin2phi,
            cos2phi,
            vec,
            prod,
        )

    def test_bench_PLO_rmult_IQU(self, mpi_benchmark, data):
        nsamples, npix, dtype_float, rng = (
            data["nsamples"],
            data["npix"],
            data["dtype_float"],
            data["rng"],
        )
        vec = rng.random(nsamples).astype(dtype_float)
        prod = np.zeros(3 * npix, dtype=dtype_float)
        sin2phi = rng.random(nsamples).astype(dtype_float)
        cos2phi = rng.random(nsamples).astype(dtype_float)

        def setup():
            prod.fill(0)
            return (
                npix,
                nsamples,
                data["pointings"],
                data["pointings_flag"],
                sin2phi,
                cos2phi,
                vec,
                prod,
                brahmap.MPI_UTILS.comm,
            ), {}

        mpi_benchmark(
            PointingLO_tools.PLO_rmult_IQU,
            setup=setup,
        )

    def test_bench_shmem_PLO_rmult_I(
        self,
        mpi_benchmark,
        data,
        nproc_reduce,
    ):
        nsamples, npix, dtype_float, rng = (
            data["nsamples"],
            data["npix"],
            data["dtype_float"],
            data["rng"],
        )
        mgr = brahmap.mpi.SharedMemoryManager(
            base_comm=brahmap.MPI_UTILS.comm,
            nproc_reduce=nproc_reduce,
            node_root=0,
        )
        vec = rng.random(nsamples).astype(dtype_float)

        node_prod, win_node_prod = mgr.alloc_shared_zeros_node(
            npix,
            dtype_float,
        )

        if mgr.tree_grp_size == 1:
            grp_prod = node_prod
            win_grp_prod = win_node_prod
        else:
            grp_prod, win_grp_prod = mgr.alloc_shared_zeros_comm(
                npix,
                dtype_float,
                comm=mgr.tree_grp_comm,
                comm_root=0,
            )

        mgr.fence_comm_all(mgr.node_comm)

        def setup():
            grp_prod.fill(0)
            node_prod.fill(0)
            return (
                npix,
                nsamples,
                data["pointings"],
                data["pointings_flag"],
                vec,
                grp_prod,
                win_grp_prod,
                node_prod,
                win_node_prod,
                mgr.node_root,
                mgr.grp_reduce,
                mgr.tree_grp_comm,
                mgr.tree_grp_root_comm,
                mgr.node_comm,
                mgr.node_root_comm,
            ), {}

        mpi_benchmark(
            PointingLO_tools.shmem_PLO_rmult_I,
            setup=setup,
        )
        mgr.free_all_resources()

    def test_bench_shmem_PLO_rmult_QU(
        self,
        mpi_benchmark,
        data,
        nproc_reduce,
    ):
        nsamples, npix, dtype_float, rng = (
            data["nsamples"],
            data["npix"],
            data["dtype_float"],
            data["rng"],
        )
        mgr = brahmap.mpi.SharedMemoryManager(
            base_comm=brahmap.MPI_UTILS.comm,
            nproc_reduce=nproc_reduce,
            node_root=0,
        )
        vec = rng.random(nsamples).astype(dtype_float)
        sin2phi = rng.random(nsamples).astype(dtype_float)
        cos2phi = rng.random(nsamples).astype(dtype_float)

        node_prod, win_node_prod = mgr.alloc_shared_zeros_node(
            2 * npix,
            dtype_float,
        )

        if mgr.tree_grp_size == 1:
            grp_prod = node_prod
            win_grp_prod = win_node_prod
        else:
            grp_prod, win_grp_prod = mgr.alloc_shared_zeros_comm(
                2 * npix,
                dtype_float,
                comm=mgr.tree_grp_comm,
                comm_root=0,
            )

        mgr.fence_comm_all(mgr.node_comm)

        def setup():
            grp_prod.fill(0)
            node_prod.fill(0)
            return (
                npix,
                nsamples,
                data["pointings"],
                data["pointings_flag"],
                sin2phi,
                cos2phi,
                vec,
                grp_prod,
                win_grp_prod,
                node_prod,
                win_node_prod,
                mgr.node_root,
                mgr.grp_reduce,
                mgr.tree_grp_comm,
                mgr.tree_grp_root_comm,
                mgr.node_comm,
                mgr.node_root_comm,
            ), {}

        mpi_benchmark(
            PointingLO_tools.shmem_PLO_rmult_QU,
            setup=setup,
        )
        mgr.free_all_resources()

    def test_bench_shmem_PLO_rmult_IQU(
        self,
        mpi_benchmark,
        data,
        nproc_reduce,
    ):
        nsamples, npix, dtype_float, rng = (
            data["nsamples"],
            data["npix"],
            data["dtype_float"],
            data["rng"],
        )
        mgr = brahmap.mpi.SharedMemoryManager(
            base_comm=brahmap.MPI_UTILS.comm,
            nproc_reduce=nproc_reduce,
            node_root=0,
        )
        vec = rng.random(nsamples).astype(dtype_float)
        sin2phi = rng.random(nsamples).astype(dtype_float)
        cos2phi = rng.random(nsamples).astype(dtype_float)

        node_prod, win_node_prod = mgr.alloc_shared_zeros_node(
            3 * npix,
            dtype_float,
        )

        if mgr.tree_grp_size == 1:
            grp_prod = node_prod
            win_grp_prod = win_node_prod
        else:
            grp_prod, win_grp_prod = mgr.alloc_shared_zeros_comm(
                3 * npix,
                dtype_float,
                comm=mgr.tree_grp_comm,
                comm_root=0,
            )

        mgr.fence_comm_all(mgr.node_comm)

        def setup():
            grp_prod.fill(0)
            node_prod.fill(0)
            return (
                npix,
                nsamples,
                data["pointings"],
                data["pointings_flag"],
                sin2phi,
                cos2phi,
                vec,
                grp_prod,
                win_grp_prod,
                node_prod,
                win_node_prod,
                mgr.node_root,
                mgr.grp_reduce,
                mgr.tree_grp_comm,
                mgr.tree_grp_root_comm,
                mgr.node_comm,
                mgr.node_root_comm,
            ), {}

        mpi_benchmark(
            PointingLO_tools.shmem_PLO_rmult_IQU,
            setup=setup,
        )
        mgr.free_all_resources()


# --- BlkDiagPrecondLO_tools ---
@pytest.mark.benchmark(group="extensions::BlkDiagPrecondLO")
class TestBlkDiagPrecondLO:
    def test_bench_BDPLO_mult_QU(self, mpi_benchmark, data):
        npix, dtype_float, rng = (
            data["npix"],
            data["dtype_float"],
            data["rng"],
        )
        weighted_sin_sq = rng.random(npix).astype(dtype_float)
        weighted_cos_sq = rng.random(npix).astype(dtype_float)
        weighted_sincos = rng.random(npix).astype(dtype_float)
        one_over_determinant = rng.random(npix).astype(dtype_float)
        vec = rng.random(2 * npix).astype(dtype_float)
        prod = np.zeros(2 * npix, dtype=dtype_float)

        mpi_benchmark(
            BlkDiagPrecondLO_tools.BDPLO_mult_QU,
            npix,
            weighted_sin_sq,
            weighted_cos_sq,
            weighted_sincos,
            one_over_determinant,
            vec,
            prod,
        )

    def test_bench_BDPLO_mult_IQU(self, mpi_benchmark, data):
        npix, dtype_float, rng = data["npix"], data["dtype_float"], data["rng"]
        weighted_counts = rng.random(npix).astype(dtype_float)
        weighted_sin_sq = rng.random(npix).astype(dtype_float)
        weighted_cos_sq = rng.random(npix).astype(dtype_float)
        weighted_sincos = rng.random(npix).astype(dtype_float)
        weighted_sin = rng.random(npix).astype(dtype_float)
        weighted_cos = rng.random(npix).astype(dtype_float)
        one_over_determinant = rng.random(npix).astype(dtype_float)
        vec = rng.random(3 * npix).astype(dtype_float)
        prod = np.zeros(3 * npix, dtype=dtype_float)

        mpi_benchmark(
            BlkDiagPrecondLO_tools.BDPLO_mult_IQU,
            npix,
            weighted_counts,
            weighted_sin_sq,
            weighted_cos_sq,
            weighted_sincos,
            weighted_sin,
            weighted_cos,
            one_over_determinant,
            vec,
            prod,
        )


# --- math extensions ---
@pytest.mark.benchmark(group="extensions::math")
class TestMath:
    def test_bench_multiply_array(self, benchmark, data):
        nsamples, dtype_float, rng = data["nsamples"], data["dtype_float"], data["rng"]
        diag = rng.random(nsamples).astype(dtype_float)
        vec = rng.random(nsamples).astype(dtype_float)
        prod = np.zeros(nsamples, dtype=dtype_float)

        benchmark(linalg_tools.multiply_array, nsamples, diag, vec, prod)

    def test_bench_unary_sin(self, benchmark, data):
        nsamples, dtype_float, rng = data["nsamples"], data["dtype_float"], data["rng"]
        vec = rng.random(nsamples).astype(dtype_float)
        result = np.zeros(nsamples, dtype=dtype_float)

        benchmark(unary_functions.sin, nsamples, vec, result)
