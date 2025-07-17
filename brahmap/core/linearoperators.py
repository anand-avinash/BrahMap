import numpy as np
import warnings
from typing import Union

from ..base import LinearOperator

from ..core import SolverType, ProcessTimeSamples

from ..base import TypeChangeWarning

from .._extensions import PointingLO_tools
from .._extensions import BlkDiagPrecondLO_tools

from ..mpi import MPI_RAISE_EXCEPTION

from brahmap import MPI_UTILS


class PointingLO(LinearOperator):
    """Derived class from the one from the  :class:`LinearOperator` in :mod:`linop`.
    It constitutes an interface for dealing with the projection operator
    (pointing matrix).

    Parameters
    ----------
    processed_samples : ProcessTimeSamples
        _description_
    solver_type : Union[None, SolverType], optional
        _description_, by default None
    """

    def __init__(
        self,
        processed_samples: ProcessTimeSamples,
        solver_type: Union[None, SolverType] = None,
    ):
        if solver_type is None:
            self.__solver_type = processed_samples.solver_type
        else:
            MPI_RAISE_EXCEPTION(
                condition=(int(processed_samples.solver_type) < int(solver_type)),
                exception=ValueError,
                message="`solver_type` must be lower than or equal to the"
                "`solver_type` of `processed_samples` object",
            )
            self.__solver_type = solver_type

        self.new_npix = processed_samples.new_npix
        self.ncols = processed_samples.new_npix * self.solver_type
        self.nrows = processed_samples.nsamples

        self.pointings = processed_samples.pointings
        self.pointings_flag = processed_samples.pointings_flag

        if self.solver_type > 1:
            self.sin2phi = processed_samples.sin2phi
            self.cos2phi = processed_samples.cos2phi

        if self.solver_type == 1:
            super(PointingLO, self).__init__(
                nargin=self.ncols,
                nargout=self.nrows,
                symmetric=False,
                matvec=self._mult_I,
                rmatvec=self._rmult_I,
                dtype=processed_samples.dtype_float,
            )
        elif self.solver_type == 2:
            super(PointingLO, self).__init__(
                nargin=self.ncols,
                nargout=self.nrows,
                symmetric=False,
                matvec=self._mult_QU,
                rmatvec=self._rmult_QU,
                dtype=processed_samples.dtype_float,
            )
        else:
            super(PointingLO, self).__init__(
                nargin=self.ncols,
                nargout=self.nrows,
                matvec=self._mult_IQU,
                symmetric=False,
                rmatvec=self._rmult_IQU,
                dtype=processed_samples.dtype_float,
            )

    def _mult_I(self, vec: np.ndarray):
        r"""
        Performs the product of a sparse matrix :math:`Av`,\
         with :math:`v` a  :mod:`numpy`  array (:math:`dim(v)=n_{pix}`)  .

        It extracts the components of :math:`v` corresponding  to the non-null \
        elements of the operator.

        """

        MPI_RAISE_EXCEPTION(
            condition=(len(vec) != self.ncols),
            exception=ValueError,
            message=f"Dimensions of `vec` is not compatible with the dimension of this `PointingLO` instance.\nShape of `PointingLO` instance: {self.shape}\nShape of `vec`: {vec.shape}",
        )

        if vec.dtype != self.dtype:
            if MPI_UTILS.rank == 0:
                warnings.warn(
                    f"dtype of `vec` will be changed to {self.dtype}",
                    TypeChangeWarning,
                )
            vec = vec.astype(dtype=self.dtype, copy=False)

        prod = np.zeros(self.nrows, dtype=self.dtype)

        PointingLO_tools.PLO_mult_I(
            nsamples=self.nrows,
            pointings=self.pointings,
            pointings_flag=self.pointings_flag,
            vec=vec,
            prod=prod,
        )

        return prod

    def _rmult_I(self, vec: np.ndarray):
        r"""
        Performs the product for the transpose operator :math:`A^T`.

        """

        MPI_RAISE_EXCEPTION(
            condition=(len(vec) != self.nrows),
            exception=ValueError,
            message=f"Dimensions of `vec` is not compatible with the dimension of this `PointingLO` instance.\nShape of `PointingLO` instance: {self.shape}\nShape of `vec`: {vec.shape}",
        )

        if vec.dtype != self.dtype:
            if MPI_UTILS.rank == 0:
                warnings.warn(
                    f"dtype of `vec` will be changed to {self.dtype}",
                    TypeChangeWarning,
                )
            vec = vec.astype(dtype=self.dtype, copy=False)

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

    def _mult_QU(self, vec: np.ndarray):
        r"""Performs :math:`A * v` with :math:`v` being a *polarization* vector.
        The output array will encode a linear combination of the two Stokes
        parameters,  (whose components are stored contiguously).

        .. math::
            d_t=  Q_p \cos(2\phi_t)+ U_p \sin(2\phi_t).
        """

        MPI_RAISE_EXCEPTION(
            condition=(len(vec) != self.ncols),
            exception=ValueError,
            message=f"Dimensions of `vec` is not compatible with the dimension of this `PointingLO` instance.\nShape of `PointingLO` instance: {self.shape}\nShape of `vec`: {vec.shape}",
        )

        if vec.dtype != self.dtype:
            if MPI_UTILS.rank == 0:
                warnings.warn(
                    f"dtype of `vec` will be changed to {self.dtype}",
                    TypeChangeWarning,
                )
            vec = vec.astype(dtype=self.dtype, copy=False)

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

    def _rmult_QU(self, vec: np.ndarray):
        r"""
        Performs :math:`A^T * v`. The output vector will be a QU-map-like array.
        """

        MPI_RAISE_EXCEPTION(
            condition=(len(vec) != self.nrows),
            exception=ValueError,
            message=f"Dimensions of `vec` is not compatible with the dimension of this `PointingLO` instance.\nShape of `PointingLO` instance: {self.shape}\nShape of `vec`: {vec.shape}",
        )

        if vec.dtype != self.dtype:
            if MPI_UTILS.rank == 0:
                warnings.warn(
                    f"dtype of `vec` will be changed to {self.dtype}",
                    TypeChangeWarning,
                )
            vec = vec.astype(dtype=self.dtype, copy=False)

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

    def _mult_IQU(self, vec: np.ndarray):
        r"""Performs the product of a sparse matrix :math:`Av`,
        with ``v`` a  :mod:`numpy` array containing the
        three Stokes parameters [IQU] .

        .. note::
            Compared to the operation ``mult`` this routine returns a
            :math:`n_t`-size vector defined as:

            .. math::
                d_t= I_p + Q_p \cos(2\phi_t)+ U_p \sin(2\phi_t).

            with :math:`p` is the pixel observed at time :math:`t` with polarization angle
            :math:`\phi_t`.
        """

        MPI_RAISE_EXCEPTION(
            condition=(len(vec) != self.ncols),
            exception=ValueError,
            message=f"Dimensions of `vec` is not compatible with the dimension of this `PointingLO` instance.\nShape of `PointingLO` instance: {self.shape}\nShape of `vec`: {vec.shape}",
        )

        if vec.dtype != self.dtype:
            if MPI_UTILS.rank == 0:
                warnings.warn(
                    f"dtype of `vec` will be changed to {self.dtype}",
                    TypeChangeWarning,
                )
            vec = vec.astype(dtype=self.dtype, copy=False)

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

    def _rmult_IQU(self, vec: np.ndarray):
        r"""
        Performs the product for the transpose operator :math:`A^T` to get a IQU map-like vector.
        Since this vector resembles the pixel of 3 maps it has 3 times the size ``Npix``.
        IQU values referring to the same pixel are  contiguously stored in the memory.

        """

        MPI_RAISE_EXCEPTION(
            condition=(len(vec) != self.nrows),
            exception=ValueError,
            message=f"Dimensions of `vec` is not compatible with the dimension of this `PointingLO` instance.\nShape of `PointingLO` instance: {self.shape}\nShape of `vec`: {vec.shape}",
        )

        if vec.dtype != self.dtype:
            if MPI_UTILS.rank == 0:
                warnings.warn(
                    f"dtype of `vec` will be changed to {self.dtype}",
                    TypeChangeWarning,
                )
            vec = vec.astype(dtype=self.dtype, copy=False)

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

    @property
    def solver_type(self):
        return self.__solver_type


class BlockDiagonalPreconditionerLO(LinearOperator):
    r"""
    Standard preconditioner defined as:

    $$M_{BD}=( P^T diag(N^{-1}) P)^{-1}$$

    where $P$ is the *pointing matrix* (see `PointingLO`).
    Such inverse operator  could be easily computed given the structure of the
    matrix $P$.

    Parameters
    ----------
    processed_samples : ProcessTimeSamples
        _description_
    solver_type : Union[None, SolverType], optional
        _description_, by default None
    """

    def __init__(
        self,
        processed_samples: ProcessTimeSamples,
        solver_type: Union[None, SolverType] = None,
    ):
        if solver_type is None:
            self.__solver_type = processed_samples.solver_type
        else:
            MPI_RAISE_EXCEPTION(
                condition=(int(processed_samples.solver_type) < int(solver_type)),
                exception=ValueError,
                message="`solver_type` must be lower than or equal to the"
                "`solver_type` of `processed_samples` object",
            )
            self.__solver_type = solver_type

        self.new_npix = processed_samples.new_npix
        self.size = processed_samples.new_npix * self.solver_type

        if self.solver_type == 1:
            self.weighted_counts = processed_samples.weighted_counts
        else:
            self.weighted_sin_sq = processed_samples.weighted_sin_sq
            self.weighted_cos_sq = processed_samples.weighted_cos_sq
            self.weighted_sincos = processed_samples.weighted_sincos
            self.one_over_determinant = processed_samples.one_over_determinant
            if self.solver_type == 3:
                self.weighted_counts = processed_samples.weighted_counts
                self.weighted_sin = processed_samples.weighted_sin
                self.weighted_cos = processed_samples.weighted_cos

        if self.solver_type == 1:
            super(BlockDiagonalPreconditionerLO, self).__init__(
                nargin=self.size,
                nargout=self.size,
                symmetric=True,
                matvec=self._mult_I,
                dtype=processed_samples.dtype_float,
            )
        elif self.solver_type == 2:
            super(BlockDiagonalPreconditionerLO, self).__init__(
                nargin=self.size,
                nargout=self.size,
                symmetric=True,
                matvec=self._mult_QU,
                dtype=processed_samples.dtype_float,
            )
        else:
            super(BlockDiagonalPreconditionerLO, self).__init__(
                nargin=self.size,
                nargout=self.size,
                symmetric=True,
                matvec=self._mult_IQU,
                dtype=processed_samples.dtype_float,
            )

    def _mult_I(self, vec: np.ndarray):
        r"""
        Action of :math:`y=( A  diag(N^{-1}) A^T)^{-1} x`,
        where :math:`x` is   an :math:`n_{pix}` array.
        """

        MPI_RAISE_EXCEPTION(
            condition=(len(vec) != self.size),
            exception=ValueError,
            message=f"Dimenstions of `vec` is not compatible with the dimension of this `BlockDiagonalPreconditionerLO` instance.\nShape of `BlockDiagonalPreconditionerLO` instance: {self.shape}\nShape of `vec`: {vec.shape}",
        )

        if vec.dtype != self.dtype:
            if MPI_UTILS.rank == 0:
                warnings.warn(
                    f"dtype of `vec` will be changed to {self.dtype}",
                    TypeChangeWarning,
                )
            vec = vec.astype(dtype=self.dtype, copy=False)

        prod = vec / self.weighted_counts

        return prod

    def _mult_QU(self, vec: np.ndarray):
        r"""
        Action of :math:`y=( A  diag(N^{-1}) A^T)^{-1} x`,
        where :math:`x` is   an :math:`n_{pix}` array.
        """

        MPI_RAISE_EXCEPTION(
            condition=(len(vec) != self.size),
            exception=ValueError,
            message=f"Dimenstions of `vec` is not compatible with the dimension of this `BlockDiagonalPreconditionerLO` instance.\nShape of `BlockDiagonalPreconditionerLO` instance: {self.shape}\nShape of `vec`: {vec.shape}",
        )

        if vec.dtype != self.dtype:
            if MPI_UTILS.rank == 0:
                warnings.warn(
                    f"dtype of `vec` will be changed to {self.dtype}",
                    TypeChangeWarning,
                )
            vec = vec.astype(dtype=self.dtype, copy=False)

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

    def _mult_IQU(self, vec: np.ndarray):
        r"""
        Action of :math:`y=( A  diag(N^{-1}) A^T)^{-1} x`,
        where :math:`x` is   an :math:`n_{pix}` array.
        """

        MPI_RAISE_EXCEPTION(
            condition=(len(vec) != self.size),
            exception=ValueError,
            message=f"Dimenstions of `vec` is not compatible with the dimension of this `BlockDiagonalPreconditionerLO` instance.\nShape of `BlockDiagonalPreconditionerLO` instance: {self.shape}\nShape of `vec`: {vec.shape}",
        )

        if vec.dtype != self.dtype:
            if MPI_UTILS.rank == 0:
                warnings.warn(
                    f"dtype of `vec` will be changed to {self.dtype}",
                    TypeChangeWarning,
                )
            vec = vec.astype(dtype=self.dtype, copy=False)

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
    def solver_type(self):
        return self.__solver_type
