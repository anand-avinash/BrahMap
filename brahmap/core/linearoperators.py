import numpy as np
import numpy.typing as npt
from ..base.linop import LinearOperator
from ..mpi import MPI_UTILS
from .process_time_samples import (
    SolverType,
    ProcessTimeSamples,
    SharedMemProcessTimeSamples,
)
from .._extensions import PointingLO_tools
from .._extensions import BlkDiagPrecondLO_tools


class PointingLO(LinearOperator):
    """A linear operator representing the pointing matrix (projection operator) $P$.

    This class encapsulates the highly sparse projection/de-projection
    operations (scatter/gather) required to map time samples onto sky
    pixels or vice versa, according to the given pointing information and
    map-making solver configuration. It excludes bad pointing samples and
    pathological pixels while performing projection/de-projection
    operations. The shape of the operator is `[nsamples, new_npix*ncomponents]`
    where `ncomponents` depends on the number of components being
    projected on the sky pixels. For instance, for `IQU` map-making,
    `ncomponents = 3`.

    Parameters
    ----------
    processed_samples : ProcessTimeSamples | SharedMemProcessTimeSamples
        The pre-processed time samples object containing pointing and
        map-making metadata
    solver_type : SolverType | None, optional
        The map-making solver configuration to use. If `None`, it falls
        back to the `solver_type` of `processed_samples`, by default `None`
    return_copy : bool, optional
        If `True`, the transposed operator (`rmatvec`) returns a copy of the
        shared memory buffer. If `False`, it returns the shared memory buffer
        directly. This argument is ignored if `processed_samples` is not a
        `SharedMemProcessTimeSamples` object, by default `True`

    Attributes
    ----------
    solver_type : SolverType
        The current map-making solver configuration
    return_copy : bool
        Whether the transposed operator returns a copy of the shared memory
        buffer
    """

    def __init__(
        self,
        processed_samples: ProcessTimeSamples | SharedMemProcessTimeSamples,
        solver_type: None | SolverType = None,
        return_copy: bool = True,
    ) -> None:
        ### Some of the functionalities of this class are implemented with C++
        ### extensions. A corresponding full Python implementation is provided in
        ### `tests/py_PointingLO.py` for reference.

        if solver_type is None:
            self.__solver_type = processed_samples.solver_type
        else:
            if int(processed_samples.solver_type) < int(solver_type):
                raise ValueError(
                    "`solver_type` must be lower than or equal to the "
                    "`solver_type` of `processed_samples` object"
                )
            self.__solver_type = solver_type

        self.return_copy = return_copy

        self.new_npix = processed_samples.new_npix
        self.ncols = processed_samples.new_npix * self.solver_type
        self.nrows = processed_samples.nsamples

        self.pointings = processed_samples.pointings
        self.pointings_flag = processed_samples.pointings_flag

        if self.solver_type > 1:
            self.sin2phi = processed_samples.sin2phi
            self.cos2phi = processed_samples.cos2phi

        self._is_shmem = hasattr(processed_samples, "shared_mem_manager")

        if self._is_shmem:
            assert isinstance(processed_samples, SharedMemProcessTimeSamples)
            self.__shared_mem_mgr = processed_samples.shared_mem_manager
            mgr = self.__shared_mem_mgr

            # Allocated node-level shared memory arrays for transposed product
            self._node_prod, self._win_node_prod = mgr.alloc_shared_array_node(
                self.ncols,
                processed_samples.dtype_float,
            )

            if mgr.tree_grp_size == 1:
                self._grp_prod = self._node_prod
                self._win_grp_prod = self._win_node_prod
            else:
                self._grp_prod, self._win_grp_prod = mgr.alloc_shared_array_comm(
                    self.ncols,
                    processed_samples.dtype_float,
                    comm=mgr.tree_grp_comm,
                    comm_root=0,
                )

            rmatvec_I = self._rmult_I_shmem
            rmatvec_QU = self._rmult_QU_shmem
            rmatvec_IQU = self._rmult_IQU_shmem
        else:
            rmatvec_I = self._rmult_I
            rmatvec_QU = self._rmult_QU
            rmatvec_IQU = self._rmult_IQU

        if self.solver_type == 1:
            super().__init__(
                nargin=self.ncols,
                nargout=self.nrows,
                symmetric=False,
                matvec=self._mult_I,
                rmatvec=rmatvec_I,
                dtype=processed_samples.dtype_float,
            )
        elif self.solver_type == 2:
            super().__init__(
                nargin=self.ncols,
                nargout=self.nrows,
                symmetric=False,
                matvec=self._mult_QU,
                rmatvec=rmatvec_QU,
                dtype=processed_samples.dtype_float,
            )
        else:
            super().__init__(
                nargin=self.ncols,
                nargout=self.nrows,
                symmetric=False,
                matvec=self._mult_IQU,
                rmatvec=rmatvec_IQU,
                dtype=processed_samples.dtype_float,
            )

    def _mult_I(self, vec: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
        r"""Performs the matrix-vector product $Pv$ for temperature-only ($I$)
        map-making.

        Parameters
        ----------
        vec : npt.NDArray[np.number]
            The input vector $v$ of size `new_npix`

        Returns
        -------
        npt.NDArray[np.number]
            The resulting vector of size `nsamples`
        """

        prod = np.zeros(self.nrows, dtype=self.dtype)

        PointingLO_tools.PLO_mult_I(
            nsamples=self.nrows,
            pointings=self.pointings,
            pointings_flag=self.pointings_flag,
            vec=vec,
            prod=prod,
        )

        return prod

    def _rmult_I(self, vec: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
        r"""Performs the transposed matrix-vector product $P^T v$ for
        temperature-only ($I$) map-making.

        Parameters
        ----------
        vec : npt.NDArray[np.number]
            The input vector of size `nsamples`

        Returns
        -------
        npt.NDArray[np.number]
            The resulting vector of size `new_npix`
        """

        prod = np.zeros(self.ncols, dtype=self.dtype)

        PointingLO_tools.PLO_rmult_I(
            new_npix=self.new_npix,
            nsamples=self.nrows,
            pointings=self.pointings,
            pointings_flag=self.pointings_flag,
            vec=vec,
            prod=prod,
            comm=MPI_UTILS.comm,
        )

        return prod

    def _rmult_I_shmem(self, vec: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
        r"""Performs the transposed matrix-vector product $P^T v$ for
        temperature-only ($I$) map-making.

        Parameters
        ----------
        vec : npt.NDArray[np.number]
            The input vector of size `nsamples`

        Returns
        -------
        npt.NDArray[np.number]
            The resulting vector of size `new_npix`
        """

        if self.__shared_mem_mgr.tree_grp_rank == 0:
            self._grp_prod[:] = 0
        if self.__shared_mem_mgr.node_rank == 0:
            self._node_prod[:] = 0

        self.__shared_mem_mgr.tree_grp_comm.Barrier()
        self.__shared_mem_mgr.node_comm.Barrier()

        PointingLO_tools.shmem_PLO_rmult_I(
            new_npix=self.new_npix,
            nsamples=self.nrows,
            pointings=self.pointings,
            pointings_flag=self.pointings_flag,
            vec=vec,
            grp_prod=self._grp_prod,
            win_grp_prod=self._win_grp_prod,
            node_prod=self._node_prod,
            win_node_prod=self._win_node_prod,
            node_root=self.__shared_mem_mgr.node_root,
            grp_reduce=self.__shared_mem_mgr.grp_reduce,
            tree_grp_comm=self.__shared_mem_mgr.tree_grp_comm,
            tree_grp_root_comm=self.__shared_mem_mgr.tree_grp_root_comm,
            node_comm=self.__shared_mem_mgr.node_comm,
            node_root_comm=self.__shared_mem_mgr.node_root_comm,
        )

        return self._node_prod.copy() if self.return_copy else self._node_prod

    def _mult_QU(self, vec: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
        r"""Performs the matrix-vector product $Pv$ for linear
        polarization ($QU$) map-making.

        Parameters
        ----------
        vec : npt.NDArray[np.number]
            The input vector $v$ of size `2*new_npix`

        Returns
        -------
        npt.NDArray[np.number]
            The resulting vector of size `nsamples`
        """

        prod = np.zeros(self.nrows, dtype=self.dtype)

        PointingLO_tools.PLO_mult_QU(
            nsamples=self.nrows,
            pointings=self.pointings,
            pointings_flag=self.pointings_flag,
            sin2phi=self.sin2phi,
            cos2phi=self.cos2phi,
            vec=vec,
            prod=prod,
        )

        return prod

    def _rmult_QU(self, vec: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
        r"""Performs the transposed matrix-vector product $P^T v$ for
        linear polarization ($QU$) map-making.

        Parameters
        ----------
        vec : npt.NDArray[np.number]
            The input vector of size `nsamples`

        Returns
        -------
        npt.NDArray[np.number]
            The resulting array of size `2*new_npix`
        """

        prod = np.zeros(self.ncols, dtype=self.dtype)

        PointingLO_tools.PLO_rmult_QU(
            new_npix=self.new_npix,
            nsamples=self.nrows,
            pointings=self.pointings,
            pointings_flag=self.pointings_flag,
            sin2phi=self.sin2phi,
            cos2phi=self.cos2phi,
            vec=vec,
            prod=prod,
            comm=MPI_UTILS.comm,
        )

        return prod

    def _rmult_QU_shmem(self, vec: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
        r"""Performs the transposed matrix-vector product $P^T v$ for
        linear polarization ($QU$) map-making.

        Parameters
        ----------
        vec : npt.NDArray[np.number]
            The input vector of size `nsamples`

        Returns
        -------
        npt.NDArray[np.number]
            The resulting array of size `2*new_npix`
        """

        if self.__shared_mem_mgr.tree_grp_rank == 0:
            self._grp_prod[:] = 0
        if self.__shared_mem_mgr.node_rank == 0:
            self._node_prod[:] = 0

        self.__shared_mem_mgr.tree_grp_comm.Barrier()
        self.__shared_mem_mgr.node_comm.Barrier()

        PointingLO_tools.shmem_PLO_rmult_QU(
            new_npix=self.new_npix,
            nsamples=self.nrows,
            pointings=self.pointings,
            pointings_flag=self.pointings_flag,
            sin2phi=self.sin2phi,
            cos2phi=self.cos2phi,
            vec=vec,
            grp_prod=self._grp_prod,
            win_grp_prod=self._win_grp_prod,
            node_prod=self._node_prod,
            win_node_prod=self._win_node_prod,
            node_root=self.__shared_mem_mgr.node_root,
            grp_reduce=self.__shared_mem_mgr.grp_reduce,
            tree_grp_comm=self.__shared_mem_mgr.tree_grp_comm,
            tree_grp_root_comm=self.__shared_mem_mgr.tree_grp_root_comm,
            node_comm=self.__shared_mem_mgr.node_comm,
            node_root_comm=self.__shared_mem_mgr.node_root_comm,
        )

        return self._node_prod.copy() if self.return_copy else self._node_prod

    def _mult_IQU(self, vec: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
        r"""Performs the matrix-vector product $Pv$ for temperature and
        linear polarization map-making.

        Parameters
        ----------
        vec : npt.NDArray[np.number]
            The input vector of size `3*new_npix`

        Returns
        -------
        npt.NDArray[np.number]
            The resulting vector of size `nsamples`
        """

        prod = np.zeros(self.nrows, dtype=self.dtype)

        PointingLO_tools.PLO_mult_IQU(
            nsamples=self.nrows,
            pointings=self.pointings,
            pointings_flag=self.pointings_flag,
            sin2phi=self.sin2phi,
            cos2phi=self.cos2phi,
            vec=vec,
            prod=prod,
        )

        return prod

    def _rmult_IQU(self, vec: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
        r"""Performs the transposed matrix-vector product $P^T v$ for
        temperature and linear polarization map-making.

        Parameters
        ----------
        vec : npt.NDArray[np.number]
            The input vector of size `nsamples`

        Returns
        -------
        npt.NDArray[np.number]
            The resulting array of size `3*new_npix`
        """

        prod = np.zeros(self.ncols, dtype=self.dtype)

        PointingLO_tools.PLO_rmult_IQU(
            new_npix=self.new_npix,
            nsamples=self.nrows,
            pointings=self.pointings,
            pointings_flag=self.pointings_flag,
            sin2phi=self.sin2phi,
            cos2phi=self.cos2phi,
            vec=vec,
            prod=prod,
            comm=MPI_UTILS.comm,
        )

        return prod

    def _rmult_IQU_shmem(self, vec: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
        r"""Performs the transposed matrix-vector product $P^T v$ for
        temperature and linear polarization map-making.

        Parameters
        ----------
        vec : npt.NDArray[np.number]
            The input vector of size `nsamples`

        Returns
        -------
        npt.NDArray[np.number]
            The resulting array of size `3*new_npix`
        """

        if self.__shared_mem_mgr.tree_grp_rank == 0:
            self._grp_prod[:] = 0
        if self.__shared_mem_mgr.node_rank == 0:
            self._node_prod[:] = 0

        self.__shared_mem_mgr.tree_grp_comm.Barrier()
        self.__shared_mem_mgr.node_comm.Barrier()

        PointingLO_tools.shmem_PLO_rmult_IQU(
            new_npix=self.new_npix,
            nsamples=self.nrows,
            pointings=self.pointings,
            pointings_flag=self.pointings_flag,
            sin2phi=self.sin2phi,
            cos2phi=self.cos2phi,
            vec=vec,
            grp_prod=self._grp_prod,
            win_grp_prod=self._win_grp_prod,
            node_prod=self._node_prod,
            win_node_prod=self._win_node_prod,
            node_root=self.__shared_mem_mgr.node_root,
            grp_reduce=self.__shared_mem_mgr.grp_reduce,
            tree_grp_comm=self.__shared_mem_mgr.tree_grp_comm,
            tree_grp_root_comm=self.__shared_mem_mgr.tree_grp_root_comm,
            node_comm=self.__shared_mem_mgr.node_comm,
            node_root_comm=self.__shared_mem_mgr.node_root_comm,
        )

        return self._node_prod.copy() if self.return_copy else self._node_prod

    @property
    def solver_type(self) -> SolverType:
        """The current map-making solver configuration.

        Returns
        -------
        SolverType
            The map-making solver type
        """
        return self.__solver_type


class BlockDiagonalPreconditionerLO(LinearOperator):
    r"""A block-diagonal preconditioner operator for iterative map-making solvers.

    Computes the standard map-making preconditioner defined as:

    $$M_{BD} = (P^T \text{diag}(N)^{-1} P)^{-1}$$

    where $P$ is the pointing matrix and $N$ is the
    noise covariance.

    Parameters
    ----------
    processed_samples : ProcessTimeSamples
        The pre-processed time samples object containing accumulated map-making weights
    solver_type : SolverType | None, optional
        The map-making solver configuration to use. If `None`, it falls
        back to the `solver_type` of `processed_samples`, by default None

    Attributes
    ----------
    solver_type : SolverType
        The active map-making solver configuration
    """

    def __init__(
        self,
        processed_samples: ProcessTimeSamples,
        solver_type: None | SolverType = None,
    ) -> None:
        ### Some of the functionalities of this class are implemented with C++
        ### extensions. A corresponding full Python implementation is provided in
        ### `tests/py_BlkDiagPrecondLO.py` for reference.

        if solver_type is None:
            self.__solver_type = processed_samples.solver_type
        else:
            if int(processed_samples.solver_type) < int(solver_type):
                raise ValueError(
                    "`solver_type` must be lower than or equal to the"
                    "`solver_type` of `processed_samples` object"
                )
            self.__solver_type = solver_type

        self.new_npix = processed_samples.new_npix
        self.size = processed_samples.new_npix * self.solver_type

        if self.solver_type == 1:
            self.weighted_counts = processed_samples.weighted_counts  # type: ignore
        else:
            self.weighted_sin_sq = processed_samples.weighted_sin_sq
            self.weighted_cos_sq = processed_samples.weighted_cos_sq
            self.weighted_sincos = processed_samples.weighted_sincos
            self.one_over_determinant = processed_samples.one_over_determinant
            if self.solver_type == 3:
                self.weighted_counts = processed_samples.weighted_counts  # type: ignore
                self.weighted_sin = processed_samples.weighted_sin  # type: ignore
                self.weighted_cos = processed_samples.weighted_cos  # type: ignore

        if self.solver_type == 1:
            super().__init__(
                nargin=self.size,
                nargout=self.size,
                symmetric=True,
                matvec=self._mult_I,
                dtype=processed_samples.dtype_float,
            )
        elif self.solver_type == 2:
            super().__init__(
                nargin=self.size,
                nargout=self.size,
                symmetric=True,
                matvec=self._mult_QU,
                dtype=processed_samples.dtype_float,
            )
        else:
            super().__init__(
                nargin=self.size,
                nargout=self.size,
                symmetric=True,
                matvec=self._mult_IQU,
                dtype=processed_samples.dtype_float,
            )

    def _mult_I(self, vec: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
        r"""Applies the block-diagonal preconditioner for temperature-only
        ($I$) map-making.

        Computes the action of $y = (P^T \text{diag}(N^{-1}) P)^{-1} v$.

        Parameters
        ----------
        vec : npt.NDArray[np.number]
            The input vector `new_npix`

        Returns
        -------
        npt.NDArray[np.number]
            The resulting array of size `new_npix`
        """

        prod = vec / self.weighted_counts

        return prod

    def _mult_QU(self, vec: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
        r"""Applies the block-diagonal preconditioner for linear
        polarization ($QU$) map-making.

        Computes the action of $y = (P^T \text{diag}(N^{-1}) P)^{-1} v$.

        Parameters
        ----------
        vec : npt.NDArray[np.number]
            The input vector `2*new_npix`

        Returns
        -------
        npt.NDArray[np.number]
            The resulting array of size `2*new_npix`
        """

        prod = np.zeros(self.size, dtype=self.dtype)

        BlkDiagPrecondLO_tools.BDPLO_mult_QU(
            new_npix=self.new_npix,
            weighted_sin_sq=self.weighted_sin_sq,
            weighted_cos_sq=self.weighted_cos_sq,
            weighted_sincos=self.weighted_sincos,
            one_over_determinant=self.one_over_determinant,
            vec=vec,
            prod=prod,
        )

        return prod

    def _mult_IQU(self, vec: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
        r"""Applies the block-diagonal preconditioner for temperature and
        linear polarization map-making.

        Computes the action of $y = (P^T \text{diag}(N^{-1}) P)^{-1} v$.

        Parameters
        ----------
        vec : npt.NDArray[np.number]
            The input vector `3*new_npix`

        Returns
        -------
        npt.NDArray[np.number]
            The resulting array of size `3*new_npix`
        """

        prod = np.zeros(self.size, dtype=self.dtype)

        BlkDiagPrecondLO_tools.BDPLO_mult_IQU(
            new_npix=self.new_npix,
            weighted_counts=self.weighted_counts,
            weighted_sin_sq=self.weighted_sin_sq,
            weighted_cos_sq=self.weighted_cos_sq,
            weighted_sincos=self.weighted_sincos,
            weighted_sin=self.weighted_sin,
            weighted_cos=self.weighted_cos,
            one_over_determinant=self.one_over_determinant,
            vec=vec,
            prod=prod,
        )

        return prod

    @property
    def solver_type(self) -> SolverType:
        """The current map-making solver configuration.

        Returns
        -------
        SolverType
            The map-making solver type
        """
        return self.__solver_type
