from enum import IntEnum
import numpy as np

import helper_ComputeWeights as cw
import helper_Repixelization as rp


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
        threshold_cond: float = 1.0e3,
        dtype_float=None,
    ):
        self.npix = npix
        self.pointings = pointings
        self.nsamples = len(pointings)

        if dtype_float is None:
            dtype_float = np.float64
        self.dtype_float = dtype_float

        if pointings_flag is None:
            pointings_flag = np.ones(self.nsamples, dtype=bool)
        self.pointings_flag = pointings_flag

        self.solver_type = solver_type

        if noise_weights is None:
            noise_weights = np.ones(self.nsamples, dtype=self.dtype_float)

        self.threshold = threshold_cond

        self._compute_weights(
            pol_angles,
            noise_weights.astype(dtype=self.dtype_float),
        )

        self._repixelization()

    def get_hit_counts(self):
        """Returns hit counts of the pixel indices"""
        pass

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
                self.pixel_mask,
                self.__old2new_pixel,
                self.pixel_flag,
            ) = cw.computeweights_pol_I(
                npix=self.npix,
                nsamples=self.nsamples,
                pointings=self.pointings,
                pointings_flag=self.pointings_flag,
                noise_weights=noise_weights,
                dtype_float=self.dtype_float,
            )

        else:
            pol_angles = pol_angles.astype(dtype=self.dtype_float)

            if self.solver_type == SolverType.QU:
                (
                    self.weighted_counts,
                    self.sin2phi,
                    self.cos2phi,
                    self.weighted_sin_sq,
                    self.weighted_cos_sq,
                    self.weighted_sincos,
                ) = cw.computeweights_pol_QU(
                    npix=self.npix,
                    nsamples=self.nsamples,
                    pointings=self.pointings,
                    pointings_flag=self.pointings_flag,
                    noise_weights=noise_weights,
                    pol_angles=pol_angles,
                    dtype_float=self.dtype_float,
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
                ) = cw.computeweights_pol_IQU(
                    npix=self.npix,
                    nsamples=self.nsamples,
                    pointings=self.pointings,
                    pointings_flag=self.pointings_flag,
                    noise_weights=noise_weights,
                    pol_angles=pol_angles,
                    dtype_float=self.dtype_float,
                )

            (
                self.new_npix,
                self.pixel_mask,
                self.__old2new_pixel,
                self.pixel_flag,
            ) = cw.get_pix_mask_pol(
                npix=self.npix,
                solver_type=self.solver_type,
                threshold=self.threshold,
                weighted_counts=self.weighted_counts,
                weighted_sin_sq=self.weighted_sin_sq,
                weighted_cos_sq=self.weighted_cos_sq,
                weighted_sincos=self.weighted_sincos,
                dtype_int=self.pointings.dtype,
            )

    def _repixelization(self):
        if self.solver_type == SolverType.I:
            self.weighted_counts = rp.repixelize_pol_I(
                new_npix=self.new_npix,
                pixel_mask=self.pixel_mask,
                weighted_counts=self.weighted_counts,
            )

        elif self.solver_type == SolverType.QU:
            (
                self.weighted_counts,
                self.weighted_sin_sq,
                self.weighted_cos_sq,
                self.weighted_sincos,
            ) = rp.repixelize_pol_QU(
                new_npix=self.new_npix,
                pixel_mask=self.pixel_mask,
                weighted_counts=self.weighted_counts,
                weighted_sin_sq=self.weighted_sin_sq,
                weighted_cos_sq=self.weighted_cos_sq,
                weighted_sincos=self.weighted_sincos,
            )

        elif self.solver_type == SolverType.IQU:
            (
                self.weighted_counts,
                self.weighted_sin_sq,
                self.weighted_cos_sq,
                self.weighted_sincos,
                self.weighted_sin,
                self.weighted_cos,
            ) = rp.repixelize_pol_IQU(
                new_npix=self.new_npix,
                pixel_mask=self.pixel_mask,
                weighted_counts=self.weighted_counts,
                weighted_sin_sq=self.weighted_sin_sq,
                weighted_cos_sq=self.weighted_cos_sq,
                weighted_sincos=self.weighted_sincos,
                weighted_sin=self.weighted_sin,
                weighted_cos=self.weighted_cos,
            )
