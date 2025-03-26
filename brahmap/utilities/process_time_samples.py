from enum import IntEnum
import numpy as np
from typing import Union
from mpi4py import MPI


from brahmap.utilities import bash_colors

from brahmap._extensions import compute_weights
from brahmap._extensions import repixelize

from brahmap import Initialize, MPI_RAISE_EXCEPTION
import brahmap


class SolverType(IntEnum):
    I = 1  # noqa: E741
    QU = 2
    IQU = 3


class ProcessTimeSamples(object):
    """
    A class to store the pre-processed and pre-computed arrays that can be used later.

    Parameters
    ----------
    npix : int
        Number of pixels on which the map-making has to be done. Equal to `healpy.nside2npix(nside)` for a healpix map of given `nside`
    pointings : np.ndarray
        A 1-d array of pointing indices
    pointings_flag : np.ndarray
        A 1-d array of pointing flags. `True` means good pointing, `False` means bad pointing.
    solver_type : SolverType
        Map-making level: I or QU or IQU
    pol_angles : np.ndarray | None
        A 1-d array containing the orientation angles of the detectors
    noise_weights : np.ndarray | None
        A 1-d array of noise weights, or the diagonal elements of the inverse of noise covariance matrix
    threshold : float
        The threshold to be used to flag pixels in the sky
    dtype_float : boh
        `dtype` of the floating point arrays
    update_pointings_inplace : bool
        The class does some operations on the pointings array. Do you want to make these operations happen in-place? If yes, you will save a lot of memory. Not recommended if you are willing to use pointing arrays somewhere after doing map-making.

    Attributes
    ----------
    npix : int
        Number of pixels on which the map-making has to be done
    pointings : np.ndarray
        A 1-d array of pointing indices
    pointings_flag : np.ndarray
        A 1-d array of pointing flags
    nsamples : int
        Number of samples on present MPI rank
    nsamples_global : int
        Global number of samples
    solver_type : SolverType
        Level of map-making: I, QU, or IQU
    pol_angles : np.ndarray
        A 1-d array containing the orientation angles of detectors
    threshold : float
        Threshold to be used to flag the pixels in the sky
    dtype_float : boh
        `dtype` of the floating point arrays
    observed_pixels : np.ndarray
        Pixel indices that are considered for map-making
    pixel_flag : np.ndarray
        A 1-d array of size `npix`. `True` indicates that the corresponding pixel index will be dropped in map-making
    bad_pixels : np.ndarray
        A 1-d array that contains all the pixel indices that will be excluded in map-making
    weighted_counts : np.ndarray
        Weighted counts
    sin2phi : np.ndarray
        A 1-d array of `sin(2\phi)`
    cos2phi : np.ndarray
        A 1-d array of `cos(2\phi)`
    weighted_sin : np.ndarray
        Weighted `sin`
    weighted_cos : np.ndarray
        Weighted `cos`
    weighted_sin_sq : np.ndarray
        Weighted `sin^2`
    weighted_cos_sq : np.ndarray
        Weighted `cos^2`
    weighted_sincos : np.ndarray
        Weighted `sin.cos`
    one_over_determinant : np.ndarray
        Inverse of determinant for each valid pixels
    new_npix : int
        The number of pixels actually being used in map-making. Equal to `len(observed_pixels)`

    """

    def __init__(
        self,
        npix: int,
        pointings: np.ndarray,
        pointings_flag: Union[np.ndarray, None] = None,
        solver_type: SolverType = SolverType.IQU,
        pol_angles: Union[np.ndarray, None] = None,
        noise_weights: Union[np.ndarray, None] = None,
        threshold: float = 1.0e-5,
        dtype_float=None,
        update_pointings_inplace: bool = False,
    ):
        if brahmap.bMPI is None:
            Initialize()

        self.__npix = npix
        self.__nsamples = len(pointings)

        self.__nsamples_global = brahmap.bMPI.comm.allreduce(self.nsamples, MPI.SUM)

        if update_pointings_inplace:
            self.pointings = pointings
            self.pointings_flag = pointings_flag
        else:
            self.pointings = pointings.copy()
            if pointings_flag is not None:
                self.pointings_flag = pointings_flag.copy()

        if pointings_flag is None:
            self.pointings_flag = np.ones(self.nsamples, dtype=bool)

        MPI_RAISE_EXCEPTION(
            condition=(len(self.pointings_flag) != self.nsamples),
            exception=AssertionError,
            message=f"Size of `pointings_flag` must be equal to the size of `pointings` array:\nlen(pointings_flag) = {len(self.pointings_flag)}\nlen(pointings) = {self.nsamples}",
        )

        self.__solver_type = solver_type
        self.__threshold = threshold

        MPI_RAISE_EXCEPTION(
            condition=(self.solver_type not in [1, 2, 3]),
            exception=ValueError,
            message="Invalid `solver_type`!!!\n`solver_type` must be either SolverType.I, SolverType.QU or SolverType.IQU (equivalently 1, 2 or 3).",
        )

        # setting the dtype for the `float` arrays: if one or both of `noise_weights` and `pol_angles` are supplied, the `dtype_float` will be inferred from them. Otherwise, the it will be set to `np.float64`
        if dtype_float is not None:
            self.__dtype_float = dtype_float
        elif noise_weights is not None and pol_angles is not None:
            # if both `noise_weights` and `pol_angles` are given, `dtype_float` will be assigned the higher `dtype`
            self.__dtype_float = np.promote_types(noise_weights.dtype, pol_angles.dtype)
        elif noise_weights is not None:
            self.__dtype_float = noise_weights.dtype
        elif pol_angles is not None:
            self.__dtype_float = pol_angles.dtype
        else:
            self.__dtype_float = np.float64

        if noise_weights is None:
            noise_weights = np.ones(self.nsamples, dtype=self.dtype_float)

        MPI_RAISE_EXCEPTION(
            condition=(len(noise_weights) != self.nsamples),
            exception=AssertionError,
            message=f"Size of `noise_weights` must be equal to the size of `pointings` array:\nlen(noise_weigths) = {len(noise_weights)}\nlen(pointings) = {self.nsamples}",
        )

        try:
            noise_weights = noise_weights.astype(
                dtype=self.dtype_float, casting="safe", copy=False
            )
        except TypeError:
            raise TypeError(
                f"The `noise_weights` array has higher dtype than `self.dtype_float={self.dtype_float}`. Please called `ProcessTimeSamples` again with `dtype_float={noise_weights.dtype}`"
            )

        if self.solver_type != 1:
            MPI_RAISE_EXCEPTION(
                condition=(len(pol_angles) != self.nsamples),
                exception=AssertionError,
                message=f"Size of `pol_angles` must be equal to the size of `pointings` array:\nlen(pol_angles) = {len(pol_angles)}\nlen(pointings) = {self.nsamples}",
            )

            try:
                pol_angles = pol_angles.astype(
                    dtype=self.dtype_float, casting="safe", copy=False
                )
            except TypeError:
                raise TypeError(
                    f"The `pol_angles` array has higher dtype than `self.dtype_float={self.dtype_float}`. Please called `ProcessTimeSamples` again with `dtype_float={pol_angles.dtype}`"
                )

        self._compute_weights(
            pol_angles,
            noise_weights,
        )

        self._repixelization()
        self._flag_bad_pixel_samples()

        if brahmap.bMPI.rank == 0:
            bc = bash_colors()
            print(
                f"\n{bc.header('--' * 13)} {bc.header(bc.bold('ProcessTimeSamples Summary'))} {bc.header('--' * 13)}"
            )

            print(
                bc.blue(
                    bc.bold(
                        f"Processed {self.nsamples_global} time samples for npix={self.npix}"
                    )
                )
            )
            print(
                bc.blue(
                    bc.bold(
                        f"Found {self.npix - self.new_npix} pathological pixels on the map"
                    )
                )
            )
            print(
                bc.blue(
                    bc.bold(
                        f"Map-maker will take into account only {self.new_npix} pixels"
                    )
                )
            )
            print(bc.header(f"{'--' * 40}"))

    @property
    def npix(self):
        return self.__npix

    @property
    def nsamples(self):
        return self.__nsamples

    @property
    def nsamples_global(self):
        return self.__nsamples_global

    @property
    def solver_type(self):
        return self.__solver_type

    @property
    def threshold(self):
        return self.__threshold

    @property
    def dtype_float(self):
        return self.__dtype_float

    @property
    def old2new_pixel(self):
        old2new_pixel = np.zeros(self.npix, dtype=self.pointings.dtype)
        for idx, flag in enumerate(self.pixel_flag):
            if flag:
                old2new_pixel[idx] = self.__old2new_pixel[idx]
            else:
                old2new_pixel[idx] = -1
        return old2new_pixel

    @property
    def bad_pixels(self):
        return np.nonzero(~self.pixel_flag)[0]

    def get_hit_counts(self):
        """Returns hit counts of the pixel indices"""
        hit_counts_newidx = np.zeros(self.new_npix, dtype=int)
        for idx in range(self.nsamples):
            hit_counts_newidx[self.pointings[idx]] += self.pointings_flag[idx]

        brahmap.bMPI.comm.Allreduce(MPI.IN_PLACE, hit_counts_newidx, MPI.SUM)

        hit_counts = np.ma.masked_array(
            data=np.zeros(self.npix),
            mask=np.logical_not(self.pixel_flag),
            fill_value=-1.6375e30,
        )

        hit_counts[~hit_counts.mask] = hit_counts_newidx
        return hit_counts

    def _compute_weights(self, pol_angles: np.ndarray, noise_weights: np.ndarray):
        self.weighted_counts = np.zeros(self.npix, dtype=self.dtype_float)
        self.observed_pixels = np.zeros(self.npix, dtype=self.pointings.dtype)
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
                observed_pixels=self.observed_pixels,
                __old2new_pixel=self.__old2new_pixel,
                pixel_flag=self.pixel_flag,
                comm=brahmap.bMPI.comm,
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
                    comm=brahmap.bMPI.comm,
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
                    comm=brahmap.bMPI.comm,
                )

            self.new_npix = compute_weights.get_pixel_mask_pol(
                solver_type=self.solver_type,
                npix=self.npix,
                threshold=self.threshold,
                weighted_counts=self.weighted_counts,
                one_over_determinant=self.one_over_determinant,
                observed_pixels=self.observed_pixels,
                __old2new_pixel=self.__old2new_pixel,
                pixel_flag=self.pixel_flag,
            )

        self.observed_pixels.resize(self.new_npix, refcheck=False)

    def _repixelization(self):
        if self.solver_type == SolverType.I:
            repixelize.repixelize_pol_I(
                new_npix=self.new_npix,
                observed_pixels=self.observed_pixels,
                weighted_counts=self.weighted_counts,
            )

            self.weighted_counts.resize(self.new_npix, refcheck=False)

        elif self.solver_type == SolverType.QU:
            repixelize.repixelize_pol_QU(
                new_npix=self.new_npix,
                observed_pixels=self.observed_pixels,
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
                observed_pixels=self.observed_pixels,
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
