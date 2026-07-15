import numpy as np
import numpy.typing as npt
from mpi4py import MPI


from .._extensions import compute_weights_shared
from .._extensions import compute_weights
from .._extensions import repixelize

from ..mpi import MPI_UTILS, SharedMemoryManager

from ..math import DTypeFloat

from ..base import SolverType, BaseProcessTimeSamples


class ProcessTimeSamples(BaseProcessTimeSamples):
    """A data container to store pre-processed pointing information,
    pre-computed map-making weights and metadata.

    This class ingests raw pointing arrays, polarization angles, and
    noise weights, and computes the necessary pixel-space representations
    (such as hit counts and trigonometric weight sums) required for the
    iterative map-making process. It automatically drops unobserved or
    pathological pixels to minimize the memory footprint of the container.

    After pre-processing, the container object can be used to create
    pointing operators, block-diagonal preconditioners, etc. as required
    for map-making.
    """

    def __init__(
        self,
        npix: int,
        pointings: npt.NDArray[np.integer],
        pointings_flag: npt.NDArray[np.bool_] | None = None,
        solver_type: SolverType = SolverType.IQU,
        pol_angles: npt.NDArray[np.number] | None = None,
        noise_weights: npt.NDArray[np.number] | None = None,
        threshold: float = 1.0e-5,
        dtype_float: DTypeFloat | None = None,
        update_pointings_inplace: bool = False,
    ):
        super().__init__(
            npix=npix,
            pointings=pointings,
            pointings_flag=pointings_flag,
            solver_type=solver_type,
            pol_angles=pol_angles,
            noise_weights=noise_weights,
            threshold=threshold,
            dtype_float=dtype_float,
            update_pointings_inplace=update_pointings_inplace,
        )

    def _compute_weights(
        self,
        pol_angles: npt.NDArray[np.number],
        noise_weights: npt.NDArray[np.number],
    ):
        self._hit_counts = np.zeros(self.npix, dtype=self._pointings.dtype)
        self._weighted_counts = np.zeros(self.npix, dtype=self.dtype_float)
        self._observed_pixels = np.zeros(self.npix, dtype=self._pointings.dtype)
        self._old2new_pixel = np.zeros(self.npix, dtype=self._pointings.dtype)
        self._pixel_flag = np.zeros(self.npix, dtype=bool)

        if self.solver_type == SolverType.I:
            self._new_npix = compute_weights.compute_weights_pol_I(
                npix=self.npix,
                nsamples=self.nsamples,
                pointings=self._pointings,
                pointings_flag=self._pointings_flag,
                noise_weights=noise_weights,
                hit_counts=self._hit_counts,
                weighted_counts=self._weighted_counts,
                observed_pixels=self._observed_pixels,
                __old2new_pixel=self._old2new_pixel,  # type: ignore
                pixel_flag=self._pixel_flag,
                comm=MPI_UTILS.comm,
            )

        else:
            self._sin2phi = np.zeros(self.nsamples, dtype=self.dtype_float)
            self._cos2phi = np.zeros(self.nsamples, dtype=self.dtype_float)

            self._weighted_sin_sq = np.zeros(self.npix, dtype=self.dtype_float)
            self._weighted_cos_sq = np.zeros(self.npix, dtype=self.dtype_float)
            self._weighted_sincos = np.zeros(self.npix, dtype=self.dtype_float)

            self._one_over_determinant = np.zeros(self.npix, dtype=self.dtype_float)

            if self.solver_type == SolverType.QU:
                compute_weights.compute_weights_pol_QU(
                    npix=self.npix,
                    nsamples=self.nsamples,
                    pointings=self._pointings,
                    pointings_flag=self._pointings_flag,
                    noise_weights=noise_weights,
                    pol_angles=pol_angles,
                    hit_counts=self._hit_counts,
                    weighted_counts=self._weighted_counts,
                    sin2phi=self._sin2phi,
                    cos2phi=self._cos2phi,
                    weighted_sin_sq=self._weighted_sin_sq,
                    weighted_cos_sq=self._weighted_cos_sq,
                    weighted_sincos=self._weighted_sincos,
                    one_over_determinant=self._one_over_determinant,
                    comm=MPI_UTILS.comm,
                )

            elif self.solver_type == SolverType.IQU:
                self._weighted_sin = np.zeros(self.npix, dtype=self.dtype_float)
                self._weighted_cos = np.zeros(self.npix, dtype=self.dtype_float)

                compute_weights.compute_weights_pol_IQU(
                    npix=self.npix,
                    nsamples=self.nsamples,
                    pointings=self._pointings,
                    pointings_flag=self._pointings_flag,
                    noise_weights=noise_weights,
                    pol_angles=pol_angles,
                    hit_counts=self._hit_counts,
                    weighted_counts=self._weighted_counts,
                    sin2phi=self._sin2phi,
                    cos2phi=self._cos2phi,
                    weighted_sin_sq=self._weighted_sin_sq,
                    weighted_cos_sq=self._weighted_cos_sq,
                    weighted_sincos=self._weighted_sincos,
                    weighted_sin=self._weighted_sin,
                    weighted_cos=self._weighted_cos,
                    one_over_determinant=self._one_over_determinant,
                    comm=MPI_UTILS.comm,
                )

            self._new_npix = compute_weights.get_pixel_mask_pol(
                solver_type=self.solver_type,
                npix=self.npix,
                threshold=self.threshold,
                hit_counts=self._hit_counts,
                one_over_determinant=self._one_over_determinant,
                observed_pixels=self._observed_pixels,
                __old2new_pixel=self._old2new_pixel,  # type: ignore
                pixel_flag=self._pixel_flag,
            )

        self._observed_pixels.resize(self._new_npix, refcheck=False)

    def _repixelization(self):
        if self.solver_type == SolverType.I:
            repixelize.repixelize_pol_I(
                new_npix=self._new_npix,
                observed_pixels=self._observed_pixels,
                hit_counts=self._hit_counts,
                weighted_counts=self._weighted_counts,
            )

            self._hit_counts.resize(self._new_npix, refcheck=False)
            self._weighted_counts.resize(self._new_npix, refcheck=False)

        elif self.solver_type == SolverType.QU:
            repixelize.repixelize_pol_QU(
                new_npix=self._new_npix,
                observed_pixels=self._observed_pixels,
                hit_counts=self._hit_counts,
                weighted_counts=self._weighted_counts,
                weighted_sin_sq=self._weighted_sin_sq,
                weighted_cos_sq=self._weighted_cos_sq,
                weighted_sincos=self._weighted_sincos,
                one_over_determinant=self._one_over_determinant,
            )

            self._hit_counts.resize(self._new_npix, refcheck=False)
            self._weighted_counts.resize(self._new_npix, refcheck=False)
            self._weighted_sin_sq.resize(self._new_npix, refcheck=False)
            self._weighted_cos_sq.resize(self._new_npix, refcheck=False)
            self._weighted_sincos.resize(self._new_npix, refcheck=False)
            self._one_over_determinant.resize(self._new_npix, refcheck=False)

        elif self.solver_type == SolverType.IQU:
            repixelize.repixelize_pol_IQU(
                new_npix=self._new_npix,
                observed_pixels=self._observed_pixels,
                hit_counts=self._hit_counts,
                weighted_counts=self._weighted_counts,
                weighted_sin_sq=self._weighted_sin_sq,
                weighted_cos_sq=self._weighted_cos_sq,
                weighted_sincos=self._weighted_sincos,
                weighted_sin=self._weighted_sin,
                weighted_cos=self._weighted_cos,
                one_over_determinant=self._one_over_determinant,
            )

            self._hit_counts.resize(self._new_npix, refcheck=False)
            self._weighted_counts.resize(self._new_npix, refcheck=False)
            self._weighted_sin_sq.resize(self._new_npix, refcheck=False)
            self._weighted_cos_sq.resize(self._new_npix, refcheck=False)
            self._weighted_sincos.resize(self._new_npix, refcheck=False)
            self._weighted_sin.resize(self._new_npix, refcheck=False)
            self._weighted_cos.resize(self._new_npix, refcheck=False)
            self._one_over_determinant.resize(self._new_npix, refcheck=False)


class SharedMemProcessTimeSamples(BaseProcessTimeSamples):
    """A data container to store pre-processed pointing information,
    pre-computed map-making weights and metadata.

    This class ingests raw pointing arrays, polarization angles, and
    noise weights, and computes the necessary pixel-space representations
    (such as hit counts and trigonometric weight sums) required for the
    iterative map-making process. It automatically drops unobserved or
    pathological pixels to minimize the memory footprint of the container.

    After pre-processing, the container object can be used to create
    pointing operators, block-diagonal preconditioners, etc. as required
    for map-making.
    """

    def __init__(
        self,
        npix: int,
        pointings: npt.NDArray[np.integer],
        pointings_flag: npt.NDArray[np.bool_] | None = None,
        solver_type: SolverType = SolverType.IQU,
        pol_angles: npt.NDArray[np.number] | None = None,
        noise_weights: npt.NDArray[np.number] | None = None,
        threshold: float = 1.0e-5,
        dtype_float: DTypeFloat | None = None,
        update_pointings_inplace: bool = False,
        nproc_reduce: int = 1,
        shared_mem_root: int = 0,
    ):
        self.__nproc_reduce = nproc_reduce
        self.__shared_mem_manager = SharedMemoryManager(
            base_comm=MPI_UTILS.comm,
            nproc_reduce=nproc_reduce,
            node_root=shared_mem_root,
        )
        super().__init__(
            npix=npix,
            pointings=pointings,
            pointings_flag=pointings_flag,
            solver_type=solver_type,
            pol_angles=pol_angles,
            noise_weights=noise_weights,
            threshold=threshold,
            dtype_float=dtype_float,
            update_pointings_inplace=update_pointings_inplace,
        )

    def _allocate_shmem_arrays(
        self, mgr: SharedMemoryManager, dint, dfloat, comm, comm_root
    ):
        self._observed_pixels, self._win_observed_pixels = mgr.alloc_shared_array(
            self.npix, dint, mgr.node_comm, mgr.node_root
        )
        self._old2new_pixel, self._win_old2new_pixel = mgr.alloc_shared_array(
            self.npix, dint, mgr.node_comm, mgr.node_root
        )
        self._pixel_flag, self._win_pixel_flag = mgr.alloc_shared_array(
            self.npix, bool, mgr.node_comm, mgr.node_root
        )
        self._hit_counts, self._win_hit_counts = mgr.alloc_shared_array(
            self.npix, dint, comm, comm_root
        )
        self._weighted_counts, self._win_weighted_counts = mgr.alloc_shared_array(
            self.npix, dfloat, comm, comm_root
        )

        if self.solver_type != SolverType.I:
            self._weighted_sin_sq, self._win_weighted_sin_sq = mgr.alloc_shared_array(
                self.npix, dfloat, comm, comm_root
            )
            self._weighted_cos_sq, self._win_weighted_cos_sq = mgr.alloc_shared_array(
                self.npix, dfloat, comm, comm_root
            )
            self._weighted_sincos, self._win_weighted_sincos = mgr.alloc_shared_array(
                self.npix, dfloat, comm, comm_root
            )
            (
                self._one_over_determinant,
                self._win_one_over_determinant,
            ) = mgr.alloc_shared_array(self.npix, dfloat, mgr.node_comm, mgr.node_root)

        if self.solver_type == SolverType.IQU:
            self._weighted_sin, self._win_weighted_sin = mgr.alloc_shared_array(
                self.npix, dfloat, comm, comm_root
            )
            self._weighted_cos, self._win_weighted_cos = mgr.alloc_shared_array(
                self.npix, dfloat, comm, comm_root
            )

    def _compute_weights(
        self,
        pol_angles: npt.NDArray[np.number],
        noise_weights: npt.NDArray[np.number],
    ):
        mgr = self.__shared_mem_manager
        dint = self._pointings.dtype
        dfloat = self.dtype_float

        self._allocate_shmem_arrays(
            mgr,
            dint,
            dfloat,
            mgr.node_comm,
            mgr.node_root,
        )

        if self.solver_type != SolverType.I:
            self._sin2phi = np.zeros(self.nsamples, dtype=dfloat)
            self._cos2phi = np.zeros(self.nsamples, dtype=dfloat)

        if mgr.node_rank == 0:
            for array in mgr.list_arrays[mgr.node_comm]:
                array[:] = 0

        mgr.node_comm.Barrier()

        if self.solver_type == SolverType.I:
            self._new_npix = compute_weights_shared.compute_weights_shmem_pol_I(
                npix=self.npix,
                nsamples=self.nsamples,
                pointings=self._pointings,
                pointings_flag=self._pointings_flag,
                noise_weights=noise_weights,
                node_hit_counts=self._hit_counts,
                win_hit_counts=self._win_hit_counts,
                node_weighted_counts=self._weighted_counts,
                win_weighted_counts=self._win_weighted_counts,
                observed_pixels=self._observed_pixels,
                __old2new_pixel=self._old2new_pixel,  # type: ignore
                pixel_flag=self._pixel_flag,
                node_root=mgr.node_root,
                tree_grp_comm=mgr.tree_grp_comm,
                tree_grp_root_comm=mgr.tree_grp_root_comm,
                node_comm=mgr.node_comm,
                node_root_comm=mgr.node_root_comm,
            )

        else:
            if self.solver_type == SolverType.QU:
                compute_weights_shared.compute_weights_shmem_pol_QU(
                    npix=self.npix,
                    nsamples=self.nsamples,
                    pointings=self._pointings,
                    pointings_flag=self._pointings_flag,
                    noise_weights=noise_weights,
                    pol_angles=pol_angles,
                    node_hit_counts=self._hit_counts,
                    win_hit_counts=self._win_hit_counts,
                    node_weighted_counts=self._weighted_counts,
                    win_weighted_counts=self._win_weighted_counts,
                    sin2phi=self._sin2phi,
                    cos2phi=self._cos2phi,
                    node_weighted_sin_sq=self._weighted_sin_sq,
                    win_weighted_sin_sq=self._win_weighted_sin_sq,
                    node_weighted_cos_sq=self._weighted_cos_sq,
                    win_weighted_cos_sq=self._win_weighted_cos_sq,
                    node_weighted_sincos=self._weighted_sincos,
                    win_weighted_sincos=self._win_weighted_sincos,
                    one_over_determinant=self._one_over_determinant,
                    node_root=mgr.node_root,
                    tree_grp_comm=mgr.tree_grp_comm,
                    tree_grp_root_comm=mgr.tree_grp_root_comm,
                    node_comm=mgr.node_comm,
                    node_root_comm=mgr.node_root_comm,
                )

            elif self.solver_type == SolverType.IQU:
                compute_weights_shared.compute_weights_shmem_pol_IQU(
                    npix=self.npix,
                    nsamples=self.nsamples,
                    pointings=self._pointings,
                    pointings_flag=self._pointings_flag,
                    noise_weights=noise_weights,
                    pol_angles=pol_angles,
                    node_hit_counts=self._hit_counts,
                    win_hit_counts=self._win_hit_counts,
                    node_weighted_counts=self._weighted_counts,
                    win_weighted_counts=self._win_weighted_counts,
                    sin2phi=self._sin2phi,
                    cos2phi=self._cos2phi,
                    node_weighted_sin_sq=self._weighted_sin_sq,
                    win_weighted_sin_sq=self._win_weighted_sin_sq,
                    node_weighted_cos_sq=self._weighted_cos_sq,
                    win_weighted_cos_sq=self._win_weighted_cos_sq,
                    node_weighted_sincos=self._weighted_sincos,
                    win_weighted_sincos=self._win_weighted_sincos,
                    node_weighted_sin=self._weighted_sin,
                    win_weighted_sin=self._win_weighted_sin,
                    node_weighted_cos=self._weighted_cos,
                    win_weighted_cos=self._win_weighted_cos,
                    one_over_determinant=self._one_over_determinant,
                    node_root=mgr.node_root,
                    tree_grp_comm=mgr.tree_grp_comm,
                    tree_grp_root_comm=mgr.tree_grp_root_comm,
                    node_comm=mgr.node_comm,
                    node_root_comm=mgr.node_root_comm,
                )

            if mgr.node_rank == mgr.node_root:
                self._new_npix = compute_weights_shared.get_pixel_mask_pol(
                    solver_type=self.solver_type,
                    npix=self.npix,
                    threshold=self.threshold,
                    hit_counts=self._hit_counts,
                    one_over_determinant=self._one_over_determinant,
                    observed_pixels=self._observed_pixels,
                    __old2new_pixel=self._old2new_pixel,  # type: ignore
                    pixel_flag=self._pixel_flag,
                )
            else:
                self._new_npix = 0

            mgr.node_comm.Bcast([self._new_npix, MPI.INT], root=mgr.node_root)

    def _repixelization(self):
        mgr = self.__shared_mem_manager
        dint = self._pointings.dtype
        dfloat = self.dtype_float
        new_npix = self._new_npix

        if mgr.node_rank == mgr.node_root:
            if self.solver_type == SolverType.I:
                repixelize.repixelize_pol_I(
                    new_npix=new_npix,
                    observed_pixels=self._observed_pixels,
                    hit_counts=self._hit_counts,
                    weighted_counts=self._weighted_counts,
                )

            elif self.solver_type == SolverType.QU:
                repixelize.repixelize_pol_QU(
                    new_npix=new_npix,
                    observed_pixels=self._observed_pixels,
                    hit_counts=self._hit_counts,
                    weighted_counts=self._weighted_counts,
                    weighted_sin_sq=self._weighted_sin_sq,
                    weighted_cos_sq=self._weighted_cos_sq,
                    weighted_sincos=self._weighted_sincos,
                    one_over_determinant=self._one_over_determinant,
                )

            elif self.solver_type == SolverType.IQU:
                repixelize.repixelize_pol_IQU(
                    new_npix=new_npix,
                    observed_pixels=self._observed_pixels,
                    hit_counts=self._hit_counts,
                    weighted_counts=self._weighted_counts,
                    weighted_sin_sq=self._weighted_sin_sq,
                    weighted_cos_sq=self._weighted_cos_sq,
                    weighted_sincos=self._weighted_sincos,
                    weighted_sin=self._weighted_sin,
                    weighted_cos=self._weighted_cos,
                    one_over_determinant=self._one_over_determinant,
                )

        mgr.node_comm.Barrier()

        def _realloc_shared(old_arr, old_win, size, dtype):
            new_arr, new_win = mgr.alloc_shared_array(
                size, dtype, comm=mgr.node_comm, comm_root=mgr.node_root
            )
            if mgr.node_rank == 0:
                new_arr[:] = old_arr[:size]
                old_win.Free()
            return new_arr, new_win

        self._observed_pixels, self._win_observed_pixels = _realloc_shared(
            self._observed_pixels,
            self._win_observed_pixels,
            new_npix,
            dint,
        )
        self._hit_counts, self._win_hit_counts = _realloc_shared(
            self._hit_counts,
            self._win_hit_counts,
            new_npix,
            dint,
        )
        self._weighted_counts, self._win_weighted_counts = _realloc_shared(
            self._weighted_counts,
            self._win_weighted_counts,
            new_npix,
            dfloat,
        )

        if self.solver_type != SolverType.I:
            self._weighted_sin_sq, self._win_weighted_sin_sq = _realloc_shared(
                self._weighted_sin_sq,
                self._win_weighted_sin_sq,
                new_npix,
                dfloat,
            )
            self._weighted_cos_sq, self._win_weighted_cos_sq = _realloc_shared(
                self._weighted_cos_sq,
                self._win_weighted_cos_sq,
                new_npix,
                dfloat,
            )
            self._weighted_sincos, self._win_weighted_sincos = _realloc_shared(
                self._weighted_sincos,
                self._win_weighted_sincos,
                new_npix,
                dfloat,
            )
            (
                self._one_over_determinant,
                self._win_one_over_determinant,
            ) = _realloc_shared(
                self._one_over_determinant,
                self._win_one_over_determinant,
                new_npix,
                dfloat,
            )

        if self.solver_type == SolverType.IQU:
            self._weighted_sin, self._win_weighted_sin = _realloc_shared(
                self._weighted_sin,
                self._win_weighted_sin,
                new_npix,
                dfloat,
            )
            self._weighted_cos, self._win_weighted_cos = _realloc_shared(
                self._weighted_cos,
                self._win_weighted_cos,
                new_npix,
                dfloat,
            )

        mgr.node_comm.Barrier()
