import numpy as np
import warnings

from brahmap import (
    MPI_UTILS,
    MPI_RAISE_EXCEPTION,
    ProcessTimeSamples,
    TypeChangeWarning,
)

from brahmap.base import LinearOperator

import py_PointingLO_tools as hplo_tools


class PointingLO(LinearOperator):
    def __init__(
        self,
        processed_samples: ProcessTimeSamples,
    ):
        self.solver_type = processed_samples.solver_type

        self.ncols = processed_samples.new_npix * self.solver_type
        self.nrows = processed_samples.nsamples

        self.pointings = processed_samples.pointings
        self.pointings_flag = processed_samples.pointings_flag
        self.dtype_float = processed_samples.dtype_float

        if self.solver_type > 1:
            self.sin2phi = processed_samples.sin2phi
            self.cos2phi = processed_samples.cos2phi

        if self.solver_type == 1:
            self.__runcase = "I"
            super(PointingLO, self).__init__(
                nargin=self.ncols,
                nargout=self.nrows,
                symmetric=False,
                matvec=self._mult_I,
                rmatvec=self._rmult_I,
            )
        elif self.solver_type == 2:
            self.__runcase = "QU"
            super(PointingLO, self).__init__(
                nargin=self.ncols,
                nargout=self.nrows,
                symmetric=False,
                matvec=self._mult_QU,
                rmatvec=self._rmult_QU,
            )
        else:
            self.__runcase = "IQU"
            super(PointingLO, self).__init__(
                nargin=self.ncols,
                nargout=self.nrows,
                matvec=self._mult_IQU,
                symmetric=False,
                rmatvec=self._rmult_IQU,
            )

    def _mult_I(self, vec: np.ndarray):
        MPI_RAISE_EXCEPTION(
            condition=(len(vec) != self.ncols),
            exception=ValueError,
            message=f"Dimensions of `vec` is not compatible with the dimension of this `PointingLO` instance.\nShape of `PointingLO` instance: {self.shape}\nShape of `vec`: {vec.shape}",
        )

        if vec.dtype != self.dtype_float:
            if MPI_UTILS.rank == 0:
                warnings.warn(
                    f"dtype of `vec` will be changed to {self.dtype_float}",
                    TypeChangeWarning,
                )
            vec = vec.astype(dtype=self.dtype_float, copy=False)

        prod = hplo_tools.PLO_mult_I(
            nrows=self.nrows,
            pointings=self.pointings,
            pointings_flags=self.pointings_flag,
            vec=vec,
        )

        return prod

    def _rmult_I(self, vec: np.ndarray):
        MPI_RAISE_EXCEPTION(
            condition=(len(vec) != self.nrows),
            exception=ValueError,
            message=f"Dimensions of `vec` is not compatible with the dimension of this `PointingLO` instance.\nShape of `PointingLO` instance: {self.shape}\nShape of `vec`: {vec.shape}",
        )

        if vec.dtype != self.dtype_float:
            if MPI_UTILS.rank == 0:
                warnings.warn(
                    f"dtype of `vec` will be changed to {self.dtype_float}",
                    TypeChangeWarning,
                )
            vec = vec.astype(dtype=self.dtype_float, copy=False)

        prod = hplo_tools.PLO_rmult_I(
            nrows=self.nrows,
            ncols=self.ncols,
            pointings=self.pointings,
            pointings_flags=self.pointings_flag,
            vec=vec,
            comm=MPI_UTILS.comm,
        )

        return prod

    def _mult_QU(self, vec: np.ndarray):
        MPI_RAISE_EXCEPTION(
            condition=(len(vec) != self.ncols),
            exception=ValueError,
            message=f"Dimensions of `vec` is not compatible with the dimension of this `PointingLO` instance.\nShape of `PointingLO` instance: {self.shape}\nShape of `vec`: {vec.shape}",
        )

        if vec.dtype != self.dtype_float:
            if MPI_UTILS.rank == 0:
                warnings.warn(
                    f"dtype of `vec` will be changed to {self.dtype_float}",
                    TypeChangeWarning,
                )
            vec = vec.astype(dtype=self.dtype_float, copy=False)

        prod = hplo_tools.PLO_mult_QU(
            nrows=self.nrows,
            pointings=self.pointings,
            pointings_flags=self.pointings_flag,
            sin2phi=self.sin2phi,
            cos2phi=self.cos2phi,
            vec=vec,
        )

        return prod

    def _rmult_QU(self, vec: np.ndarray):
        MPI_RAISE_EXCEPTION(
            condition=(len(vec) != self.nrows),
            exception=ValueError,
            message=f"Dimensions of `vec` is not compatible with the dimension of this `PointingLO` instance.\nShape of `PointingLO` instance: {self.shape}\nShape of `vec`: {vec.shape}",
        )

        if vec.dtype != self.dtype_float:
            if MPI_UTILS.rank == 0:
                warnings.warn(
                    f"dtype of `vec` will be changed to {self.dtype_float}",
                    TypeChangeWarning,
                )
            vec = vec.astype(dtype=self.dtype_float, copy=False)

        prod = hplo_tools.PLO_rmult_QU(
            nrows=self.nrows,
            ncols=self.ncols,
            pointings=self.pointings,
            pointings_flags=self.pointings_flag,
            sin2phi=self.sin2phi,
            cos2phi=self.cos2phi,
            vec=vec,
            comm=MPI_UTILS.comm,
        )

        return prod

    def _mult_IQU(self, vec: np.ndarray):
        MPI_RAISE_EXCEPTION(
            condition=(len(vec) != self.ncols),
            exception=ValueError,
            message=f"Dimensions of `vec` is not compatible with the dimension of this `PointingLO` instance.\nShape of `PointingLO` instance: {self.shape}\nShape of `vec`: {vec.shape}",
        )

        if vec.dtype != self.dtype_float:
            if MPI_UTILS.rank == 0:
                warnings.warn(
                    f"dtype of `vec` will be changed to {self.dtype_float}",
                    TypeChangeWarning,
                )
            vec = vec.astype(dtype=self.dtype_float, copy=False)

        prod = hplo_tools.PLO_mult_IQU(
            nrows=self.nrows,
            pointings=self.pointings,
            pointings_flags=self.pointings_flag,
            sin2phi=self.sin2phi,
            cos2phi=self.cos2phi,
            vec=vec,
        )

        return prod

    def _rmult_IQU(self, vec: np.ndarray):
        MPI_RAISE_EXCEPTION(
            condition=(len(vec) != self.nrows),
            exception=ValueError,
            message=f"Dimensions of `vec` is not compatible with the dimension of this `PointingLO` instance.\nShape of `PointingLO` instance: {self.shape}\nShape of `vec`: {vec.shape}",
        )

        if vec.dtype != self.dtype_float:
            if MPI_UTILS.rank == 0:
                warnings.warn(
                    f"dtype of `vec` will be changed to {self.dtype_float}",
                    TypeChangeWarning,
                )
            vec = vec.astype(dtype=self.dtype_float, copy=False)

        prod = hplo_tools.PLO_rmult_IQU(
            nrows=self.nrows,
            ncols=self.ncols,
            pointings=self.pointings,
            pointings_flags=self.pointings_flag,
            sin2phi=self.sin2phi,
            cos2phi=self.cos2phi,
            vec=vec,
            comm=MPI_UTILS.comm,
        )

        return prod

    def solver_string(self):
        """
        Return a string depending on the map you are processing
        """
        if self.solver_type == 1:
            return "I"
        elif self.solver_type == 2:
            return "QU"
        else:
            return "IQU"
