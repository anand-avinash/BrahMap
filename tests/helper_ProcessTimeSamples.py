from enum import IntEnum
import numpy as np
import warnings

import helper_ComputeWeights as cw
import helper_Repixelization as rp

import brahmap
from brahmap.utilities import TypeChangeWarning

from mpi4py import MPI


class SolverType(IntEnum):
    I = 1  # noqa: E741
    QU = 2
    IQU = 3


class ProcessTimeSamples(object):
    def __init__(
        self,
        npix: int,
        pointings: np.ndarray,
        pointings_flag: np.ndarray = None,
        solver_type: SolverType = SolverType.IQU,
        pol_angles: np.ndarray = None,
        noise_weights: np.ndarray = None,
        threshold: float = 1.0e-5,
        dtype_float=None,
        update_pointings_inplace: bool = True,
    ):
        if brahmap.bMPI is None:
            brahmap.Initialize()

        self.npix = npix
        self.nsamples = len(pointings)

        if update_pointings_inplace:
            self.pointings = pointings
            self.pointings_flag = pointings_flag
        else:
            self.pointings = pointings.copy()
            if pointings_flag is not None:
                self.pointings_flag = pointings_flag.copy()

        if self.pointings_flag is None:
            self.pointings_flag = np.ones(self.nsamples, dtype=bool)

        self.threshold = threshold
        self.solver_type = solver_type

        if dtype_float is not None:
            self.dtype_float = dtype_float
        elif noise_weights is not None and pol_angles is not None:
            self.dtype_float = np.promote_types(noise_weights.dtype, pol_angles.dtype)
        elif noise_weights is not None:
            self.dtype_float = noise_weights.dtype
        elif pol_angles is not None:
            self.dtype_float = pol_angles.dtype
        else:
            self.dtype_float = np.float64

        if noise_weights is None:
            noise_weights = np.ones(self.nsamples, dtype=self.dtype_float)

        noise_weights = noise_weights.astype(dtype=self.dtype_float, copy=False)

        if self.solver_type != 1:
            if len(pol_angles) != self.nsamples:
                raise AssertionError(
                    f"Size of `pol_angles` must be equal to the size of `pointings` array:\nlen(pol_angles) = {len(pol_angles)}\nlen(pointings) = {self.nsamples}"
                )

            if pol_angles.dtype != self.dtype_float:
                warnings.warn(
                    f"dtype of `pol_angles` will be changed to {self.dtype_float}",
                    TypeChangeWarning,
                )
                pol_angles = pol_angles.astype(dtype=self.dtype_float, copy=False)

        self._compute_weights(
            pol_angles,
            noise_weights,
        )

        self._repixelization()
        self._flag_bad_pixel_samples()

    def get_hit_counts(self, mask_fill_value=np.nan):
        """Returns hit counts of the pixel indices"""
        hit_counts_newidx = np.zeros(self.new_npix, dtype=int)
        for idx in range(self.nsamples):
            hit_counts_newidx[self.pointings[idx]] += self.pointings_flag[idx]

        brahmap.bMPI.comm.Allreduce(MPI.IN_PLACE, hit_counts_newidx, MPI.SUM)

        hit_counts = np.ma.masked_array(
            data=np.zeros(self.npix, dtype=int),
            mask=np.logical_not(self.pixel_flag),
        )

        hit_counts[~hit_counts.mask] = hit_counts_newidx
        return hit_counts

    @property
    def old2new_pixel(self):
        old2new_pixel = np.zeros(self.npix, dtype=self.pointings.dtype)
        for idx, flag in enumerate(self.pixel_flag):
            if flag:
                old2new_pixel[idx] = self.__old2new_pixel[idx]
            else:
                old2new_pixel[idx] = -1
        return old2new_pixel

    def _compute_weights(self, pol_angles, noise_weights):
        if self.solver_type == SolverType.I:
            (
                self.new_npix,
                self.weighted_counts,
                self.observed_pixels,
                self.__old2new_pixel,
                self.pixel_flag,
            ) = cw.computeweights_pol_I(
                npix=self.npix,
                nsamples=self.nsamples,
                pointings=self.pointings,
                pointings_flag=self.pointings_flag,
                noise_weights=noise_weights,
                dtype_float=self.dtype_float,
                comm=brahmap.bMPI.comm,
            )

        else:
            if self.solver_type == SolverType.QU:
                (
                    self.weighted_counts,
                    self.sin2phi,
                    self.cos2phi,
                    self.weighted_sin_sq,
                    self.weighted_cos_sq,
                    self.weighted_sincos,
                    self.one_over_determinant,
                ) = cw.computeweights_pol_QU(
                    npix=self.npix,
                    nsamples=self.nsamples,
                    pointings=self.pointings,
                    pointings_flag=self.pointings_flag,
                    noise_weights=noise_weights,
                    pol_angles=pol_angles,
                    dtype_float=self.dtype_float,
                    comm=brahmap.bMPI.comm,
                )

            elif self.solver_type == SolverType.IQU:
                (
                    self.weighted_counts,
                    self.sin2phi,
                    self.cos2phi,
                    self.weighted_sin_sq,
                    self.weighted_cos_sq,
                    self.weighted_sincos,
                    self.weighted_sin,
                    self.weighted_cos,
                    self.one_over_determinant,
                ) = cw.computeweights_pol_IQU(
                    npix=self.npix,
                    nsamples=self.nsamples,
                    pointings=self.pointings,
                    pointings_flag=self.pointings_flag,
                    noise_weights=noise_weights,
                    pol_angles=pol_angles,
                    dtype_float=self.dtype_float,
                    comm=brahmap.bMPI.comm,
                )

            (
                self.new_npix,
                self.observed_pixels,
                self.__old2new_pixel,
                self.pixel_flag,
            ) = cw.get_pix_mask_pol(
                npix=self.npix,
                solver_type=self.solver_type,
                threshold=self.threshold,
                weighted_counts=self.weighted_counts,
                one_over_determinant=self.one_over_determinant,
                dtype_int=self.pointings.dtype,
            )

    def _repixelization(self):
        if self.solver_type == SolverType.I:
            self.weighted_counts = rp.repixelize_pol_I(
                new_npix=self.new_npix,
                observed_pixels=self.observed_pixels,
                weighted_counts=self.weighted_counts,
            )

        elif self.solver_type == SolverType.QU:
            (
                self.weighted_counts,
                self.weighted_sin_sq,
                self.weighted_cos_sq,
                self.weighted_sincos,
                self.one_over_determinant,
            ) = rp.repixelize_pol_QU(
                new_npix=self.new_npix,
                observed_pixels=self.observed_pixels,
                weighted_counts=self.weighted_counts,
                weighted_sin_sq=self.weighted_sin_sq,
                weighted_cos_sq=self.weighted_cos_sq,
                weighted_sincos=self.weighted_sincos,
                one_over_determinant=self.one_over_determinant,
            )

        elif self.solver_type == SolverType.IQU:
            (
                self.weighted_counts,
                self.weighted_sin_sq,
                self.weighted_cos_sq,
                self.weighted_sincos,
                self.weighted_sin,
                self.weighted_cos,
                self.one_over_determinant,
            ) = rp.repixelize_pol_IQU(
                new_npix=self.new_npix,
                observed_pixels=self.observed_pixels,
                weighted_counts=self.weighted_counts,
                weighted_sin_sq=self.weighted_sin_sq,
                weighted_cos_sq=self.weighted_cos_sq,
                weighted_sincos=self.weighted_sincos,
                weighted_sin=self.weighted_sin,
                weighted_cos=self.weighted_cos,
                one_over_determinant=self.one_over_determinant,
            )

    def _flag_bad_pixel_samples(self):
        rp.flag_bad_pixel_samples(
            self.nsamples,
            self.pixel_flag,
            self.__old2new_pixel,
            self.pointings,
            self.pointings_flag,
        )
