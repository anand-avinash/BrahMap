from enum import IntEnum
import numpy as np
import warnings

from brahmap.utilities.tools import TypeChangeWarning
from brahmap.utilities import bash_colors

from brahmap._extensions import compute_weights
from brahmap._extensions import repixelize


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

        if len(self.pointings_flag) != self.nsamples:
            raise AssertionError(
                f"Size of `pointings_flag` must be equal to the size of `pointings` array:\nlen(pointings_flag) = {len(pointings_flag)}\nlen(pointings) = {self.nsamples}"
            )

        self.threshold = threshold
        self.solver_type = solver_type

        if self.solver_type not in [1, 2, 3]:
            raise ValueError(
                "Invalid `solver_type`!!!\n`solver_type` must be either SolverType.I, SolverType.QU or SolverType.IQU (equivalently 1, 2 or 3)."
            )

        # setting the dtype for the `float` arrays: if one or both of `noise_weights` and `pol_angles` are supplied, the `dtype_float` will be inferred from them. Otherwise, the it will be set to `np.float64`
        if dtype_float is not None:
            self.dtype_float = dtype_float
        elif noise_weights is not None and pol_angles is not None:
            # if both `noise_weights` and `pol_angles` are given, `dtype_float` will be assigned the higher `dtype`
            self.dtype_float = np.promote_types(noise_weights.dtype, pol_angles.dtype)
        elif noise_weights is not None:
            self.dtype_float = noise_weights.dtype
        elif pol_angles is not None:
            self.dtype_float = pol_angles.dtype
        else:
            self.dtype_float = np.float64

        if noise_weights is None:
            noise_weights = np.ones(self.nsamples, dtype=self.dtype_float)

        if len(noise_weights) != self.nsamples:
            raise AssertionError(
                f"Size of `noise_weights` must be equal to the size of `pointings` array:\nlen(noise_weigths) = {len(noise_weights)}\nlen(pointings) = {self.nsamples}"
            )

        if noise_weights.dtype != self.dtype_float:
            warnings.warn(
                f"dtype of `noise_weights` will be changed to {self.dtype_float}",
                TypeChangeWarning,
            )
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

        bc = bash_colors()
        print(bc.header(f"{bc.bold(' ProcessTimeSamples Summary '):-^60}"))
        print(
            bc.blue(bc.bold(f"Read {self.nsamples} time samples for npix={self.npix}"))
        )
        print(
            bc.blue(bc.bold(f"Found {self.npix - self.new_npix} pathological pixels"))
        )
        print(
            bc.blue(
                bc.bold(
                    f"Map-maker will take into account only {self.new_npix} pixels."
                )
            )
        )
        print(bc.header("---" * 20))

    def get_hit_counts(self):
        """Returns hit counts of the pixel indices"""
        hit_counts_newidx = np.zeros(self.new_npix, dtype=int)
        for idx in range(self.nsamples):
            hit_counts_newidx[self.pointings[idx]] += self.pointings_flag[idx]

        hit_counts = np.ma.masked_array(
            data=np.zeros(self.npix),
            mask=np.logical_not(self.pixel_flag),
            fill_value=-1.6375e30,
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

    def _compute_weights(self, pol_angles: np.ndarray, noise_weights: np.ndarray):
        self.weighted_counts = np.zeros(self.npix, dtype=self.dtype_float)
        self.pixel_mask = np.zeros(self.npix, dtype=self.pointings.dtype)
        self.__old2new_pixel = np.zeros(self.npix, dtype=self.pointings.dtype)
        self.pixel_flag = np.zeros(self.npix, dtype=bool)

        if self.solver_type == SolverType.I:
            self.new_npix = compute_weights.compute_weights_pol_I(
                npix=self.npix,
                nsamples=self.nsamples,
                pointings=self.pointings,
                pointings_flag=self.pointings_flag,
                noise_weights=noise_weights,
                weighted_counts=self.weighted_counts,
                pixel_mask=self.pixel_mask,
                __old2new_pixel=self.__old2new_pixel,
                pixel_flag=self.pixel_flag,
            )

        else:
            self.sin2phi = np.zeros(self.nsamples, dtype=self.dtype_float)
            self.cos2phi = np.zeros(self.nsamples, dtype=self.dtype_float)

            self.weighted_sin_sq = np.zeros(self.npix, dtype=self.dtype_float)
            self.weighted_cos_sq = np.zeros(self.npix, dtype=self.dtype_float)
            self.weighted_sincos = np.zeros(self.npix, dtype=self.dtype_float)

            self.one_over_determinant = np.zeros(self.npix, dtype=self.dtype_float)

            if self.solver_type == SolverType.QU:
                compute_weights.compute_weights_pol_QU(
                    npix=self.npix,
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
                    one_over_determinant=self.one_over_determinant,
                )

            elif self.solver_type == SolverType.IQU:
                self.weighted_sin = np.zeros(self.npix, dtype=self.dtype_float)
                self.weighted_cos = np.zeros(self.npix, dtype=self.dtype_float)

                compute_weights.compute_weights_pol_IQU(
                    npix=self.npix,
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
                    one_over_determinant=self.one_over_determinant,
                )

            self.new_npix = compute_weights.get_pixel_mask_pol(
                solver_type=self.solver_type,
                npix=self.npix,
                threshold=self.threshold,
                weighted_counts=self.weighted_counts,
                one_over_determinant=self.one_over_determinant,
                pixel_mask=self.pixel_mask,
                __old2new_pixel=self.__old2new_pixel,
                pixel_flag=self.pixel_flag,
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
                one_over_determinant=self.one_over_determinant,
            )

            self.weighted_counts.resize(self.new_npix, refcheck=False)
            self.weighted_sin_sq.resize(self.new_npix, refcheck=False)
            self.weighted_cos_sq.resize(self.new_npix, refcheck=False)
            self.weighted_sincos.resize(self.new_npix, refcheck=False)
            self.one_over_determinant.resize(self.new_npix, refcheck=False)

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
                one_over_determinant=self.one_over_determinant,
            )

            self.weighted_counts.resize(self.new_npix, refcheck=False)
            self.weighted_sin_sq.resize(self.new_npix, refcheck=False)
            self.weighted_cos_sq.resize(self.new_npix, refcheck=False)
            self.weighted_sincos.resize(self.new_npix, refcheck=False)
            self.weighted_sin.resize(self.new_npix, refcheck=False)
            self.weighted_cos.resize(self.new_npix, refcheck=False)
            self.one_over_determinant.resize(self.new_npix, refcheck=False)

    def _flag_bad_pixel_samples(self):
        repixelize.flag_bad_pixel_samples(
            nsamples=self.nsamples,
            pixel_flag=self.pixel_flag,
            old2new_pixel=self.__old2new_pixel,
            pointings=self.pointings,
            pointings_flag=self.pointings_flag,
        )
