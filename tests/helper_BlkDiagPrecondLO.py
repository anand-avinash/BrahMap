import numpy as np
import warnings
import brahmap.linop as lp
from brahmap.utilities import ProcessTimeSamples, TypeChangeWarning

import helper_BlkDiagPrecondLO_tools as bdplo_tools


class BlockDiagonalPreconditionerLO(lp.LinearOperator):
    def __init__(self, processed_samples: ProcessTimeSamples):
        self.solver_type = processed_samples.solver_type
        self.dtype_float = processed_samples.dtype_float
        self.new_npix = processed_samples.new_npix
        self.size = processed_samples.new_npix * self.solver_type

        if self.solver_type == 1:
            self.weighted_counts = processed_samples.weighted_counts
        else:
            self.weighted_sin_sq = processed_samples.weighted_sin_sq
            self.weighted_cos_sq = processed_samples.weighted_cos_sq
            self.weighted_sincos = processed_samples.weighted_sincos
            if self.solver_type == 3:
                self.weighted_counts = processed_samples.weighted_counts
                self.weighted_sin = processed_samples.weighted_sin
                self.weighted_cos = processed_samples.weighted_cos

        if self.solver_type == 1:
            super(BlockDiagonalPreconditionerLO, self).__init__(
                nargin=self.size, nargout=self.size, symmetric=True, matvec=self._mult_I
            )
        elif self.solver_type == 2:
            super(BlockDiagonalPreconditionerLO, self).__init__(
                nargin=self.size,
                nargout=self.size,
                symmetric=True,
                matvec=self._mult_QU,
            )
        else:
            super(BlockDiagonalPreconditionerLO, self).__init__(
                nargin=self.size,
                nargout=self.size,
                symmetric=True,
                matvec=self._mult_IQU,
            )

    def _mult_I(self, vec: np.ndarray):
        r"""
        Action of :math:`y=( A  diag(N^{-1}) A^T)^{-1} x`,
        where :math:`x` is   an :math:`n_{pix}` array.
        """

        if len(vec) != self.size:
            raise ValueError(
                f"Dimenstions of `vec` is not compatible with the dimension of this `BlockDiagonalPreconditionerLO` instance.\nShape of `BlockDiagonalPreconditionerLO` instance: {self.shape}\nShape of `vec`: {vec.shape}"
            )

        if vec.dtype != self.dtype_float:
            warnings.warn(
                f"dtype of `vec` will be changed to {self.dtype_float}",
                TypeChangeWarning,
            )
            vec = vec.astype(dtype=self.dtype_float, copy=False)

        prod = vec / self.weighted_counts

        return prod

    def _mult_QU(self, vec: np.ndarray):
        r"""
        Action of :math:`y=( A  diag(N^{-1}) A^T)^{-1} x`,
        where :math:`x` is   an :math:`n_{pix}` array.
        """

        if len(vec) != self.size:
            raise ValueError(
                f"Dimenstions of `vec` is not compatible with the dimension of this `BlockDiagonalPreconditionerLO` instance.\nShape of `BlockDiagonalPreconditionerLO` instance: {self.shape}\nShape of `vec`: {vec.shape}"
            )

        if vec.dtype != self.dtype_float:
            warnings.warn(
                f"dtype of `vec` will be changed to {self.dtype_float}",
                TypeChangeWarning,
            )
            vec = vec.astype(dtype=self.dtype_float, copy=False)

        prod = bdplo_tools.BDPLO_mult_QU(
            solver_type=self.solver_type,
            new_npix=self.new_npix,
            weighted_sin_sq=self.weighted_sin_sq,
            weighted_cos_sq=self.weighted_cos_sq,
            weighted_sincos=self.weighted_sincos,
            vec=vec,
        )

        return prod

    def _mult_IQU(self, vec: np.ndarray):
        r"""
        Action of :math:`y=( A  diag(N^{-1}) A^T)^{-1} x`,
        where :math:`x` is   an :math:`n_{pix}` array.
        """

        if len(vec) != self.size:
            raise ValueError(
                f"Dimenstions of `vec` is not compatible with the dimension of this `BlockDiagonalPreconditionerLO` instance.\nShape of `BlockDiagonalPreconditionerLO` instance: {self.shape}\nShape of `vec`: {vec.shape}"
            )

        if vec.dtype != self.dtype_float:
            warnings.warn(
                f"dtype of `vec` will be changed to {self.dtype_float}",
                TypeChangeWarning,
            )
            vec = vec.astype(dtype=self.dtype_float, copy=False)

        prod = bdplo_tools.BDPLO_mult_IQU(
            solver_type=self.solver_type,
            new_npix=self.new_npix,
            weighted_counts=self.weighted_counts,
            weighted_sin_sq=self.weighted_sin_sq,
            weighted_cos_sq=self.weighted_cos_sq,
            weighted_sincos=self.weighted_sincos,
            weighted_sin=self.weighted_sin,
            weighted_cos=self.weighted_cos,
            vec=vec,
        )

        return prod
