############################ TEST DESCRIPTION ############################
#
# Test defined here are related to the functions defined in the extension
# module `compute_weights_shared`
#
# - class `TestComputeWeightsShared`: This test class implements the tests
#   for all the functions defined in the extension module
#   `compute_weights_shared`. It simply tests if the computations defined
#   in cpp functions produce the same result as their python analog.
#
#   -   `test_compute_weights_shmem_pol_{I,QU,IQU}()`: test the computations
#       of `compute_weights_shared.compute_weights_shmem_pol_{I,QU,IQU}`
#   -   `test_get_pix_mask_pol_{QU,IQU}`: test the computations of
#       `compute_weights_shared.get_pixel_mask_pol()`
#
###########################################################################

import pytest
import numpy as np

import brahmap
from brahmap._extensions import compute_weights_shared
from brahmap.mpi import SharedMemoryManager

import py_ComputeWeights as cw


TOLERANCES = {
    np.float32: {"rtol": 1.5e-3, "atol": 1.0e-5},
    np.float64: {"rtol": 1.5e-5, "atol": 1.0e-10},
}


class TestComputeWeightsShared:
    def test_compute_weights_shmem_pol_I(self, setup_scan):
        initint, initfloat = setup_scan

        tol = TOLERANCES[initfloat.dtype]
        rtol, atol = tol["rtol"], tol["atol"]
        mgr = SharedMemoryManager(
            base_comm=brahmap.MPI_UTILS.comm,
            nproc_reduce=1,
        )

        cpp_observed_pixels, _ = mgr.alloc_shared_zeros_node(
            initint.npix,
            initint.dtype,
        )
        cpp_old2new_pixel, _ = mgr.alloc_shared_zeros_node(
            initint.npix,
            initint.dtype,
        )
        cpp_pixel_flag, _ = mgr.alloc_shared_zeros_node(
            initint.npix,
            bool,
        )
        cpp_hit_counts, win_hit_counts = mgr.alloc_shared_zeros_node(
            initint.npix,
            initint.dtype,
        )
        cpp_weighted_counts, win_weighted_counts = mgr.alloc_shared_zeros_node(
            initint.npix,
            initfloat.dtype,
        )

        mgr.fence_comm_all(mgr.node_comm)

        cpp_new_npix = compute_weights_shared.compute_weights_shmem_pol_I(
            initint.npix,
            initint.nsamples,
            initint.pointings,
            initint.pointings_flag,
            initfloat.noise_weights,
            cpp_hit_counts,
            win_hit_counts,
            cpp_weighted_counts,
            win_weighted_counts,
            cpp_observed_pixels,
            cpp_old2new_pixel,
            cpp_pixel_flag,
            mgr.node_root,
            mgr.grp_reduce,
            mgr.tree_grp_comm,
            mgr.tree_grp_root_comm,
            mgr.node_comm,
            mgr.node_root_comm,
        )

        (
            py_new_npix,
            py_hit_counts,
            py_weighted_counts,
            py_observed_pixels,
            py_old2new_pixel,
            py_pixel_flag,
        ) = cw.computeweights_pol_I(
            initint.npix,
            initint.nsamples,
            initint.pointings,
            initint.pointings_flag,
            initfloat.noise_weights,
            dtype_float=initfloat.dtype,
            comm=brahmap.MPI_UTILS.comm,
        )

        mgr.fence_comm_all(mgr.node_comm)

        cpp_observed_pixels = cpp_observed_pixels[:cpp_new_npix]

        np.testing.assert_equal(cpp_new_npix, py_new_npix)
        np.testing.assert_array_equal(cpp_hit_counts, py_hit_counts)
        np.testing.assert_allclose(
            cpp_weighted_counts, py_weighted_counts, rtol=rtol, atol=atol
        )
        np.testing.assert_array_equal(cpp_observed_pixels, py_observed_pixels)
        np.testing.assert_array_equal(cpp_old2new_pixel, py_old2new_pixel)
        np.testing.assert_array_equal(cpp_pixel_flag, py_pixel_flag)

        mgr.free_shared_arrays_all()

    def test_compute_weights_shmem_pol_QU(self, setup_scan):
        initint, initfloat = setup_scan

        tol = TOLERANCES[initfloat.dtype]
        rtol, atol = tol["rtol"], tol["atol"]
        mgr = SharedMemoryManager(
            base_comm=brahmap.MPI_UTILS.comm,
            nproc_reduce=1,
        )

        cpp_sin2phi = np.zeros(initint.nsamples, dtype=initfloat.dtype)
        cpp_cos2phi = np.zeros(initint.nsamples, dtype=initfloat.dtype)

        cpp_hit_counts, win_hit_counts = mgr.alloc_shared_zeros_node(
            initint.npix,
            initint.dtype,
        )
        cpp_weighted_counts, win_weighted_counts = mgr.alloc_shared_zeros_node(
            initint.npix,
            initfloat.dtype,
        )
        cpp_weighted_sin_sq, win_weighted_sin_sq = mgr.alloc_shared_zeros_node(
            initint.npix,
            initfloat.dtype,
        )
        cpp_weighted_cos_sq, win_weighted_cos_sq = mgr.alloc_shared_zeros_node(
            initint.npix,
            initfloat.dtype,
        )
        cpp_weighted_sincos, win_weighted_sincos = mgr.alloc_shared_zeros_node(
            initint.npix,
            initfloat.dtype,
        )
        cpp_one_over_determinant, _ = mgr.alloc_shared_zeros_node(
            initint.npix,
            initfloat.dtype,
        )

        mgr.fence_comm_all(mgr.node_comm)

        compute_weights_shared.compute_weights_shmem_pol_QU(
            initint.npix,
            initint.nsamples,
            initint.pointings,
            initint.pointings_flag,
            initfloat.noise_weights,
            initfloat.pol_angles,
            cpp_hit_counts,
            win_hit_counts,
            cpp_weighted_counts,
            win_weighted_counts,
            cpp_sin2phi,
            cpp_cos2phi,
            cpp_weighted_sin_sq,
            win_weighted_sin_sq,
            cpp_weighted_cos_sq,
            win_weighted_cos_sq,
            cpp_weighted_sincos,
            win_weighted_sincos,
            cpp_one_over_determinant,
            mgr.node_root,
            mgr.grp_reduce,
            mgr.tree_grp_comm,
            mgr.tree_grp_root_comm,
            mgr.node_comm,
            mgr.node_root_comm,
        )

        (
            py_hit_counts,
            py_weighted_counts,
            py_sin2phi,
            py_cos2phi,
            py_weighted_sin_sq,
            py_weighted_cos_sq,
            py_weighted_sincos,
            __,
        ) = cw.computeweights_pol_QU(
            initint.npix,
            initint.nsamples,
            initint.pointings,
            initint.pointings_flag,
            initfloat.noise_weights,
            initfloat.pol_angles,
            dtype_float=initfloat.dtype,
            comm=brahmap.MPI_UTILS.comm,
        )

        mgr.fence_comm_all(mgr.node_comm)

        np.testing.assert_array_equal(cpp_hit_counts, py_hit_counts)
        np.testing.assert_allclose(
            cpp_weighted_counts, py_weighted_counts, rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(cpp_sin2phi, py_sin2phi, rtol=rtol, atol=atol)
        np.testing.assert_allclose(cpp_cos2phi, py_cos2phi, rtol=rtol, atol=atol)
        np.testing.assert_allclose(
            cpp_weighted_sin_sq, py_weighted_sin_sq, rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(
            cpp_weighted_cos_sq, py_weighted_cos_sq, rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(
            cpp_weighted_sincos, py_weighted_sincos, rtol=rtol, atol=atol
        )

        mgr.free_shared_arrays_all()

    def test_compute_weights_shmem_pol_IQU(self, setup_scan):
        initint, initfloat = setup_scan

        tol = TOLERANCES[initfloat.dtype]
        rtol, atol = tol["rtol"], tol["atol"]
        mgr = SharedMemoryManager(
            base_comm=brahmap.MPI_UTILS.comm,
            nproc_reduce=1,
        )

        cpp_sin2phi = np.zeros(initint.nsamples, dtype=initfloat.dtype)
        cpp_cos2phi = np.zeros(initint.nsamples, dtype=initfloat.dtype)

        cpp_hit_counts, win_hit_counts = mgr.alloc_shared_zeros_node(
            initint.npix,
            initint.dtype,
        )
        cpp_weighted_counts, win_weighted_counts = mgr.alloc_shared_zeros_node(
            initint.npix,
            initfloat.dtype,
        )
        cpp_weighted_sin_sq, win_weighted_sin_sq = mgr.alloc_shared_zeros_node(
            initint.npix,
            initfloat.dtype,
        )
        cpp_weighted_cos_sq, win_weighted_cos_sq = mgr.alloc_shared_zeros_node(
            initint.npix,
            initfloat.dtype,
        )
        cpp_weighted_sincos, win_weighted_sincos = mgr.alloc_shared_zeros_node(
            initint.npix,
            initfloat.dtype,
        )
        cpp_weighted_sin, win_weighted_sin = mgr.alloc_shared_zeros_node(
            initint.npix,
            initfloat.dtype,
        )
        cpp_weighted_cos, win_weighted_cos = mgr.alloc_shared_zeros_node(
            initint.npix,
            initfloat.dtype,
        )
        cpp_one_over_determinant, _ = mgr.alloc_shared_zeros_node(
            initint.npix,
            initfloat.dtype,
        )

        mgr.fence_comm_all(mgr.node_comm)

        compute_weights_shared.compute_weights_shmem_pol_IQU(
            initint.npix,
            initint.nsamples,
            initint.pointings,
            initint.pointings_flag,
            initfloat.noise_weights,
            initfloat.pol_angles,
            cpp_hit_counts,
            win_hit_counts,
            cpp_weighted_counts,
            win_weighted_counts,
            cpp_sin2phi,
            cpp_cos2phi,
            cpp_weighted_sin_sq,
            win_weighted_sin_sq,
            cpp_weighted_cos_sq,
            win_weighted_cos_sq,
            cpp_weighted_sincos,
            win_weighted_sincos,
            cpp_weighted_sin,
            win_weighted_sin,
            cpp_weighted_cos,
            win_weighted_cos,
            cpp_one_over_determinant,
            mgr.node_root,
            mgr.grp_reduce,
            mgr.tree_grp_comm,
            mgr.tree_grp_root_comm,
            mgr.node_comm,
            mgr.node_root_comm,
        )

        (
            py_hit_counts,
            py_weighted_counts,
            py_sin2phi,
            py_cos2phi,
            py_weighted_sin_sq,
            py_weighted_cos_sq,
            py_weighted_sincos,
            py_weighted_sin,
            py_weighted_cos,
            __,
        ) = cw.computeweights_pol_IQU(
            initint.npix,
            initint.nsamples,
            initint.pointings,
            initint.pointings_flag,
            initfloat.noise_weights,
            initfloat.pol_angles,
            dtype_float=initfloat.dtype,
            comm=brahmap.MPI_UTILS.comm,
        )

        mgr.fence_comm_all(mgr.node_comm)

        np.testing.assert_array_equal(cpp_hit_counts, py_hit_counts)
        np.testing.assert_allclose(
            cpp_weighted_counts, py_weighted_counts, rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(cpp_sin2phi, py_sin2phi, rtol=rtol, atol=atol)
        np.testing.assert_allclose(cpp_cos2phi, py_cos2phi, rtol=rtol, atol=atol)
        np.testing.assert_allclose(
            cpp_weighted_sin_sq, py_weighted_sin_sq, rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(
            cpp_weighted_cos_sq, py_weighted_cos_sq, rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(
            cpp_weighted_sincos, py_weighted_sincos, rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(
            cpp_weighted_sin, py_weighted_sin, rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(
            cpp_weighted_cos, py_weighted_cos, rtol=rtol, atol=atol
        )

        mgr.free_shared_arrays_all()

    def test_get_pix_mask_pol_QU(self, setup_scan):
        initint, initfloat = setup_scan
        (
            hit_counts,
            __,
            __,
            __,
            __,
            __,
            __,
            one_over_determinant,
        ) = cw.computeweights_pol_QU(
            initint.npix,
            initint.nsamples,
            initint.pointings,
            initint.pointings_flag,
            initfloat.noise_weights,
            initfloat.pol_angles,
            dtype_float=initfloat.dtype,
            comm=brahmap.MPI_UTILS.comm,
        )

        mgr = SharedMemoryManager(
            base_comm=brahmap.MPI_UTILS.comm,
            nproc_reduce=1,
        )

        cpp_observed_pixels, win_observed_pixels = mgr.alloc_shared_zeros_node(
            initint.npix,
            initint.dtype,
        )
        cpp_old2new_pixel, win_old2new_pixel = mgr.alloc_shared_zeros_node(
            initint.npix,
            initint.dtype,
        )
        cpp_pixel_flag, win_pixel_flag = mgr.alloc_shared_zeros_node(
            initint.npix,
            bool,
        )

        mgr.fence_comm_all(mgr.node_comm)

        hit_counts = hit_counts.astype(dtype=initint.dtype)

        if mgr.node_rank == mgr.node_root:
            cpp_new_npix = compute_weights_shared.get_pixel_mask_pol(
                2,
                initint.npix,
                1.0e3,
                hit_counts,
                one_over_determinant,
                cpp_observed_pixels,
                cpp_old2new_pixel,
                cpp_pixel_flag,
            )
        else:
            cpp_new_npix = 0

        mgr.fence_comm_all(mgr.node_comm)

        cpp_new_npix = mgr.node_comm.bcast(cpp_new_npix, root=mgr.node_root)

        cpp_observed_pixels_new = cpp_observed_pixels[:cpp_new_npix]

        (
            py_new_npix,
            py_observed_pixels,
            py_old2new_pixel,
            py_pixel_flag,
        ) = cw.get_pix_mask_pol(
            initint.npix,
            2,
            1.0e3,
            hit_counts,
            one_over_determinant,
            dtype_int=initint.dtype,
        )

        np.testing.assert_equal(cpp_new_npix, py_new_npix)
        np.testing.assert_array_equal(cpp_observed_pixels_new, py_observed_pixels)
        np.testing.assert_array_equal(cpp_old2new_pixel, py_old2new_pixel)
        np.testing.assert_array_equal(cpp_pixel_flag, py_pixel_flag)

        mgr.free_shared_arrays_all()

    def test_get_pix_mask_pol_IQU(self, setup_scan):
        initint, initfloat = setup_scan
        (
            hit_counts,
            __,
            __,
            __,
            __,
            __,
            __,
            __,
            __,
            one_over_determinant,
        ) = cw.computeweights_pol_IQU(
            initint.npix,
            initint.nsamples,
            initint.pointings,
            initint.pointings_flag,
            initfloat.noise_weights,
            initfloat.pol_angles,
            dtype_float=initfloat.dtype,
            comm=brahmap.MPI_UTILS.comm,
        )

        mgr = SharedMemoryManager(
            base_comm=brahmap.MPI_UTILS.comm,
            nproc_reduce=1,
        )

        cpp_observed_pixels, win_observed_pixels = mgr.alloc_shared_zeros_node(
            initint.npix,
            initint.dtype,
        )
        cpp_old2new_pixel, win_old2new_pixel = mgr.alloc_shared_zeros_node(
            initint.npix,
            initint.dtype,
        )
        cpp_pixel_flag, win_pixel_flag = mgr.alloc_shared_zeros_node(
            initint.npix,
            bool,
        )

        mgr.fence_comm_all(mgr.node_comm)

        hit_counts = hit_counts.astype(dtype=initint.dtype)

        if mgr.node_rank == mgr.node_root:
            cpp_new_npix = compute_weights_shared.get_pixel_mask_pol(
                3,
                initint.npix,
                1.0e3,
                hit_counts,
                one_over_determinant,
                cpp_observed_pixels,
                cpp_old2new_pixel,
                cpp_pixel_flag,
            )
        else:
            cpp_new_npix = 0

        mgr.fence_comm_all(mgr.node_comm)

        cpp_new_npix = mgr.node_comm.bcast(cpp_new_npix, root=mgr.node_root)

        cpp_observed_pixels_new = cpp_observed_pixels[:cpp_new_npix]

        (
            py_new_npix,
            py_observed_pixels,
            py_old2new_pixel,
            py_pixel_flag,
        ) = cw.get_pix_mask_pol(
            initint.npix,
            3,
            1.0e3,
            hit_counts,
            one_over_determinant,
            dtype_int=initint.dtype,
        )

        np.testing.assert_equal(cpp_new_npix, py_new_npix)
        np.testing.assert_array_equal(cpp_observed_pixels_new, py_observed_pixels)
        np.testing.assert_array_equal(cpp_old2new_pixel, py_old2new_pixel)
        np.testing.assert_array_equal(cpp_pixel_flag, py_pixel_flag)

        mgr.free_shared_arrays_all()


if __name__ == "__main__":
    pytest.main(
        [
            f"{__file__}::TestComputeWeightsShared::test_compute_weights_shmem_pol_I",
            "-v",
            "-s",
        ]
    )
    pytest.main(
        [
            f"{__file__}::TestComputeWeightsShared::test_compute_weights_shmem_pol_QU",
            "-v",
            "-s",
        ]
    )
    pytest.main(
        [
            f"{__file__}::TestComputeWeightsShared::test_compute_weights_shmem_pol_IQU",
            "-v",
            "-s",
        ]
    )
    pytest.main(
        [f"{__file__}::TestComputeWeightsShared::test_get_pix_mask_pol_QU", "-v", "-s"]
    )
    pytest.main(
        [f"{__file__}::TestComputeWeightsShared::test_get_pix_mask_pol_IQU", "-v", "-s"]
    )
