from enum import IntEnum
import numpy as np

import compute_weights
import repixelize


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

        if len(self.pointings_flag) != self.nsamples:
            raise AssertionError(
                f"Size of `pointings_flag` must be equal to the size of `pointings` array:\nlen(pointings_flag) = {len(pointings_flag)}\nlen(pointings) = {self.nsamples}"
            )

        self.solver_type = solver_type

        if self.solver_type != 1:
            if len(pol_angles) != self.nsamples:
                raise AssertionError(
                    f"Size of `pol_angles` must be equal to the size of `pointings` array:\nlen(pol_angles) = {len(pol_angles)}\nlen(pointings) = {self.nsamples}"
                )

        if noise_weights is None:
            noise_weights = np.ones(self.nsamples, dtype=self.dtype_float)

        if len(noise_weights) != self.nsamples:
            raise AssertionError(
                f"Size of `noise_weights` must be equal to the size of `pointings` array:\nlen(noise_weigths) = {len(noise_weights)}\nlen(pointings) = {self.nsamples}"
            )

        self.threshold = threshold_cond

        self._compute_weights(
            pol_angles,
            noise_weights.astype(dtype=self.dtype_float),
        )

        self._repixelization()

    def get_hit_counts(self):
        """Returns hit counts of the pixel indices"""
        pass

    def _compute_weights(self, pol_angles, noise_weights):
        self.weighted_counts = np.zeros(self.npix, dtype=self.dtype_float)
        self.pixel_mask = np.zeros(self.npix, dtype=self.pointings.dtype)

        if self.solver_type == SolverType.I:
            self.new_npix = compute_weights.compute_weights_pol_I(
                npix=self.npix,
                nsamples=self.nsamples,
                pointings=self.pointings,
                pointings_flag=self.pointings_flag,
                noise_weights=noise_weights,
                weighted_counts=self.weighted_counts,
                pixel_mask=self.pixel_mask,
            )

        else:
            pol_angles = pol_angles.astype(dtype=self.dtype_float)

            self.sin2phi = np.zeros(self.nsamples, dtype=self.dtype_float)
            self.cos2phi = np.zeros(self.nsamples, dtype=self.dtype_float)

            self.weighted_sin_sq = np.zeros(self.npix, dtype=self.dtype_float)
            self.weighted_cos_sq = np.zeros(self.npix, dtype=self.dtype_float)
            self.weighted_sincos = np.zeros(self.npix, dtype=self.dtype_float)

            if self.solver_type == SolverType.QU:
                compute_weights.compute_weights_pol_QU(
                    nsamples=self.nsamples,
                    pointings=self.pointings,
                    pointings_flag=self.pointings_flag,
                    noise_weights=noise_weights,
                    pol_angles=pol_angles,
                    weighted_counts=self.weighted_counts,
                    sin2phi=self.sin2phi,
                    cos2phi=self.cos2phi,
                    weighted_sin_sq=self.weighted_sin_sq,
                    weighted_cos_sq=self.weighted_cos_sq,
                    weighted_sincos=self.weighted_sincos,
                )

            elif self.solver_type == SolverType.IQU:
                self.weighted_sin = np.zeros(self.npix, dtype=self.dtype_float)
                self.weighted_cos = np.zeros(self.npix, dtype=self.dtype_float)

                compute_weights.compute_weights_pol_IQU(
                    nsamples=self.nsamples,
                    pointings=self.pointings,
                    pointings_flag=self.pointings_flag,
                    noise_weights=noise_weights,
                    pol_angles=pol_angles,
                    weighted_counts=self.weighted_counts,
                    sin2phi=self.sin2phi,
                    cos2phi=self.cos2phi,
                    weighted_sin_sq=self.weighted_sin_sq,
                    weighted_cos_sq=self.weighted_cos_sq,
                    weighted_sincos=self.weighted_sincos,
                    weighted_sin=self.weighted_sin,
                    weighted_cos=self.weighted_cos,
                )

            self.new_npix = compute_weights.get_pixel_mask_pol(
                solver_type=self.solver_type,
                npix=self.npix,
                threshold=self.threshold,
                weighted_counts=self.weighted_counts,
                weighted_sin_sq=self.weighted_sin_sq,
                weighted_cos_sq=self.weighted_cos_sq,
                weighted_sincos=self.weighted_sincos,
                pixel_mask=self.pixel_mask,
            )

        self.pixel_mask.resize(self.new_npix, refcheck=False)

    def _repixelization(self):
        if self.solver_type == SolverType.I:
            repixelize.repixelize_pol_I(
                new_npix=self.new_npix,
                pixel_mask=self.pixel_mask,
                weighted_counts=self.weighted_counts,
            )

            self.weighted_counts.resize(self.new_npix, refcheck=False)

        elif self.solver_type == SolverType.QU:
            repixelize.repixelize_pol_QU(
                new_npix=self.new_npix,
                pixel_mask=self.pixel_mask,
                weighted_counts=self.weighted_counts,
                weighted_sin_sq=self.weighted_sin_sq,
                weighted_cos_sq=self.weighted_cos_sq,
                weighted_sincos=self.weighted_sincos,
            )

            self.weighted_counts.resize(self.new_npix, refcheck=False)
            self.weighted_sin_sq.resize(self.new_npix, refcheck=False)
            self.weighted_cos_sq.resize(self.new_npix, refcheck=False)
            self.weighted_sincos.resize(self.new_npix, refcheck=False)

        elif self.solver_type == SolverType.IQU:
            repixelize.repixelize_pol_IQU(
                new_npix=self.new_npix,
                pixel_mask=self.pixel_mask,
                weighted_counts=self.weighted_counts,
                weighted_sin_sq=self.weighted_sin_sq,
                weighted_cos_sq=self.weighted_cos_sq,
                weighted_sincos=self.weighted_sincos,
                weighted_sin=self.weighted_sin,
                weighted_cos=self.weighted_cos,
            )

            self.weighted_counts.resize(self.new_npix, refcheck=False)
            self.weighted_sin_sq.resize(self.new_npix, refcheck=False)
            self.weighted_cos_sq.resize(self.new_npix, refcheck=False)
            self.weighted_sincos.resize(self.new_npix, refcheck=False)
            self.weighted_sin.resize(self.new_npix, refcheck=False)
            self.weighted_cos.resize(self.new_npix, refcheck=False)
