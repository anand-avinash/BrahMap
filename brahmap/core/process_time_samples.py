from enum import IntEnum
import numpy as np
import numpy.typing as npt
from mpi4py import MPI


from ..utilities import bash_colors

from .._extensions import compute_weights
from .._extensions import repixelize

from ..mpi import MPI_UTILS

from ..math import DTypeFloat


class SolverType(IntEnum):
    """An enumeration defining the map-making solver configuration.

    Attributes
    ----------
    I : int
        Temperature-only map-making (solves for Stokes $I$)
    QU : int
        Linear polarization-only map-making (solves for Stokes $Q$ and $U$)
    IQU : int
        Temperature and linear polarization map-making (solves for
        Stokes $I$, $Q$, and $U$)
    """

    I = 1  # noqa: E741
    QU = 2
    IQU = 3


class ProcessTimeSamples(object):
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

    Parameters
    ----------
    npix : int
        Number of pixels on which the map-making has to be done (e.g.
        `healpy.nside2npix(nside)`)
    pointings : npt.NDArray[np.integer]
        A 1-d array of pixel indices pointing to the sky map for each time sample
    pointings_flag : npt.NDArray[np.bool_] | None, optional
        A 1-d boolean array where `True` indicates a valid pointing and
        `False` flags a bad pointing, by default `None`. If set as `None`,
        all the pointings are considered valid
    solver_type : SolverType, optional
        The level of map-making solver to construct ($I$, $QU$, or
        $IQU$), by default `SolverType.IQU`
    pol_angles : npt.NDArray[np.number] | None, optional
        A 1-d array containing the polarization orientation angles of the
        detectors for each sample, by default `None`
    noise_weights : npt.NDArray[np.number] | None, optional
        A 1-d array containing the inverse noise variance for each time
        sample, by default `None`. If set as `None`, the inverse noise
        variance is set to 1 for each time sample
    threshold : float, optional
        The condition number threshold used to flag degenerate or
        under-sampled pixels, by default `1.0e-5`
    dtype_float : DTypeFloat | None, optional
        The data type to use for floating point arrays, by default
        `None`. If set as `None`, the data type is inferred from the input
        `noise_weights` or `pol_angles` array. If none of them are
        supplied, it will be set to `np.float64`
    update_pointings_inplace : bool, optional
        If `True`, the class will perform operations on the `pointings`
        array in-place to save memory. This can modify
        the input array. If `False`, the class will create a copy of the
        original array. By default `False`

    Attributes
    ----------
    npix : int
        The original number of pixels for the target map resolution
    pointings : npt.NDArray[np.integer]
        The 1-d array of pixel pointing indices for each time sample
    pointings_flag : npt.NDArray[np.bool_]
        The 1-d array of flags indicating valid (`True`) or discarded
        (`False`) time samples
    nsamples : int
        The number of time samples processed by the current MPI rank
    nsamples_global : int
        The total number of time samples across all MPI ranks
    solver_type : SolverType
        The current map-making solver configuration ($I$, $QU$, or $IQU$)
    threshold : float
        The condition number threshold used to flag bad pixels
    dtype_float : DTypeFloat
        The inferred or specified data type for floating point arrays
    observed_pixels : npt.NDArray[np.integer]
        A 1-d array containing the original indices of the pixels that
        are fully valid for map-making
    pixel_flag : npt.NDArray[np.bool_]
        A 1-d boolean array of size `npix` where `True` indicates a
        dropped or pathological pixel
    bad_pixels : npt.NDArray[np.integer]
        A 1-d array containing the indices of all pathological pixels
        excluded from the map-making
    old2new_pixel : npt.NDArray[np.integer]
        A 1-d array mapping original pixel indices to new pixel indices
    weighted_counts : npt.NDArray[np.number]
        A 1-d array accumulating the inverse noise weights per valid pixel
    sin2phi : npt.NDArray[np.number]
        A 1-d array containing $\\sin(2\\phi)$ evaluated at the valid time samples
    cos2phi : npt.NDArray[np.number]
        A 1-d array containing $\\cos(2\\phi)$ evaluated at the valid time samples
    weighted_sin : npt.NDArray[np.number]
        A 1-d array accumulating the noise-weighted $\\sin(2\\phi)$ sum
        per valid pixel
    weighted_cos : npt.NDArray[np.number]
        A 1-d array accumulating the noise-weighted $\\cos(2\\phi)$ sum
        per valid pixel
    weighted_sin_sq : npt.NDArray[np.number]
        A 1-d array accumulating the noise-weighted $\\sin^2(2\\phi)$ sum
        per valid pixel
    weighted_cos_sq : npt.NDArray[np.number]
        A 1-d array accumulating the noise-weighted $\\cos^2(2\\phi)$ sum
        per valid pixel
    weighted_sincos : npt.NDArray[np.number]
        A 1-d array accumulating the noise-weighted $\\sin(2\\phi)\\cos(2\\phi)$
        sum per valid pixel
    one_over_determinant : npt.NDArray[np.number]
        A 1-d array containing the inverse determinant of the
        block-diagonal operator $P^T diag(N)^{-1} P$
    new_npix : int
        The number of non-pathological pixels actually being solved for

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
        ### Some of the functionalities of this class are implemented with C++
        ### extensions. A corresponding full Python implementation is provided in
        ### `tests/py_ProcessTimeSamples.py` for reference.

        self.__npix = npix
        self.__nsamples = len(pointings)

        self.__nsamples_global = MPI_UTILS.comm.allreduce(self.nsamples, MPI.SUM)

        if update_pointings_inplace:
            self.__pointings = pointings
            self.__pointings_flag = pointings_flag
        else:
            self.__pointings = pointings.copy()
            if pointings_flag is not None:
                self.__pointings_flag = pointings_flag.copy()

        if pointings_flag is None:
            self.__pointings_flag = np.ones(self.nsamples, dtype=bool)

        assert self.__pointings_flag is not None

        if len(self.__pointings_flag) != self.nsamples:
            raise AssertionError(
                "Size of `pointings_flag` must be equal to the size of "
                "`pointings` array:\n"
                f"len(pointings_flag) = {len(self.__pointings_flag)}\n"
                f"len(pointings) = {self.nsamples}"
            )

        self.__solver_type = solver_type
        self.__threshold = threshold

        if self.solver_type not in [1, 2, 3]:
            raise ValueError(
                "Invalid `solver_type`!!!\n`solver_type` must be either "
                "SolverType.I, SolverType.QU or SolverType.IQU "
                "(equivalently 1, 2 or 3)."
            )

        # setting the dtype for the `float` arrays: if one or both of
        # `noise_weights` and `pol_angles` are supplied, the `dtype_float`
        # will be inferred from them. Otherwise, the it will be set to
        # `np.float64`
        if dtype_float is not None:
            self.__dtype_float = dtype_float
        elif noise_weights is not None and pol_angles is not None:
            # if both `noise_weights` and `pol_angles` are given,
            # `dtype_float` will be assigned the higher `dtype`
            self.__dtype_float = np.promote_types(
                noise_weights.dtype,
                pol_angles.dtype,
            )
        elif noise_weights is not None:
            self.__dtype_float = noise_weights.dtype  # type: ignore
        elif pol_angles is not None:
            self.__dtype_float = pol_angles.dtype  # type: ignore
        else:
            self.__dtype_float = np.float64

        if noise_weights is None:
            noise_weights = np.ones(self.nsamples, dtype=self.dtype_float)

        if len(noise_weights) != self.nsamples:
            raise AssertionError(
                "Size of `noise_weights` must be equal to the size of "
                "`pointings` array:\n"
                f"len(noise_weigths) = {len(noise_weights)}\n"
                f"len(pointings) = {self.nsamples}"
            )

        try:
            noise_weights = noise_weights.astype(
                dtype=self.dtype_float, casting="safe", copy=False
            )
        except TypeError:
            raise TypeError(
                "The `noise_weights` array has higher dtype than "
                f"`self.dtype_float={self.dtype_float}`. Please call "
                f"`ProcessTimeSamples` again with `dtype_float={noise_weights.dtype}`"
            )

        if self.solver_type != 1:
            assert pol_angles is not None
            if len(pol_angles) != self.nsamples:
                raise AssertionError(
                    "Size of `pol_angles` must be equal to the size of "
                    "`pointings` array:\n"
                    f"len(pol_angles) = {len(pol_angles)}\n"
                    f"len(pointings) = {self.nsamples}"
                )

            try:
                pol_angles = pol_angles.astype(
                    dtype=self.dtype_float, casting="safe", copy=False
                )
            except TypeError:
                raise TypeError(
                    "The `pol_angles` array has higher dtype than "
                    f"`self.dtype_float={self.dtype_float}`. Please call "
                    f"`ProcessTimeSamples` again with `dtype_float={pol_angles.dtype}`"
                )

        self._compute_weights(
            pol_angles,
            noise_weights,
        )

        if self.__new_npix == 0:
            raise ValueError(
                "All pixels were found to be pathological. The map-making "
                "cannot be done. Please ensure that the inputs are consistent!"
            )

        self._repixelization()
        self._flag_bad_pixel_samples()

        if MPI_UTILS.rank == 0:
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
                        f"Found {self.npix - self.__new_npix} pathological pixels on the map"
                    )
                )
            )
            print(
                bc.blue(
                    bc.bold(
                        f"Map-maker will take into account only {self.__new_npix} pixels"
                    )
                )
            )
            print(bc.header(f"{'--' * 40}"))

    @property
    def npix(self) -> int:
        """Number of pixels on which the map-making has to be done.

        Returns
        -------
        int
            Number of pixels on which the map-making has to be done
        """
        return self.__npix

    @property
    def pointings(self) -> npt.NDArray[np.integer]:
        """A 1-d array of pixel indices pointing to the observed sky pixel
        for each time sample

        Returns
        -------
        npt.NDArray[np.integer]
            A 1-d array of pixel pointing indices for each time sample
        """
        return self.__pointings

    @property
    def pointings_flag(self) -> npt.NDArray[np.bool_] | None:
        """A 1-d boolean array where `True` indicates a valid pointing and
        `False` flags a bad pointing

        Returns
        -------
        npt.NDArray[np.bool_]
            The 1-d array of flags indicating valid (`True`) or discarded
        (`False`) time samples
        """
        return self.__pointings_flag

    @property
    def nsamples(self) -> int:
        """The number of time samples processed by the current MPI rank

        Returns
        -------
        int
            Number of samples on current MPI rank
        """
        return self.__nsamples

    @property
    def nsamples_global(self) -> int:
        """The total number of time samples across all MPI ranks

        Returns
        -------
        int
            Global number of samples
        """
        return self.__nsamples_global

    @property
    def solver_type(self) -> SolverType:
        """The current map-making solver configuration ($I$, $QU$, or $IQU$)

        Returns
        -------
        SolverType
            Level of map-making: $I$, $QU$, or $IQU$
        """
        return self.__solver_type

    @property
    def threshold(self) -> float:
        """The condition number threshold used to flag bad pixels

        Returns
        -------
        float
            Threshold to used for flagging the pixels in the sky
        """
        return self.__threshold

    @property
    def dtype_float(self) -> DTypeFloat:
        """The inferred or specified data type for floating point arrays

        Returns
        -------
        DTypeFloat
            `dtype` of the floating point arrays
        """
        return self.__dtype_float

    @property
    def observed_pixels(self) -> npt.NDArray[np.integer]:
        """A 1-d array containing the original indices of the pixels that
        are fully valid for map-making

        Returns
        -------
        npt.NDArray[np.integer]
            A 1-d array that contains all the pixel indices that are
            considered valid for map-making
        """
        return self.__observed_pixels

    @property
    def pixel_flag(self) -> npt.NDArray[np.bool_]:
        """A 1-d boolean array of size `npix` where `True` indicates a bad
        pixel and `False` flags a valid pixel

        Returns
        -------
        npt.NDArray[np.bool_]
            A 1-d boolean array of size `npix` where `True` indicates a
            dropped or pathological pixel
        """
        return self.__pixel_flag

    @property
    def bad_pixels(self) -> npt.NDArray[np.integer]:
        """A 1-d array that contains all the pixel indices that will be excluded
        in map-making.

        Returns
        -------
        npt.NDArray[np.integer]
            A 1-d array that contains all the pixel indices that will be excluded
            in map-making
        """
        return np.nonzero(~self.__pixel_flag)[0]

    @property
    def old2new_pixel(self) -> npt.NDArray[np.integer]:
        """A 1-d array mapping old pixel indices to new pixel indices

        Returns
        -------
        npt.NDArray[np.integer]
            A 1-d array mapping old pixel indices to new pixel indices
        """
        old2new_pixel = np.where(self.__pixel_flag, self.__old2new_pixel, -1)
        return old2new_pixel.astype(self.__pointings.dtype, copy=False)

    @property
    def weighted_counts(self) -> npt.NDArray[np.number]:
        """A 1-d array accumulating the inverse noise weights per valid pixel

        Returns
        -------
        npt.NDArray[np.number]
            A 1-d array accumulating the inverse noise weights per valid pixel
        """
        return self.__weighted_counts

    @property
    def sin2phi(self) -> npt.NDArray[np.number]:
        """A 1-d array containing $\\sin(2\\phi)$ evaluated at the valid time samples

        Returns
        -------
        npt.NDArray[np.number]
            A 1-d array containing $\\sin(2\\phi)$ evaluated at the valid time samples
        """
        return self.__sin2phi

    @property
    def cos2phi(self) -> npt.NDArray[np.number]:
        """A 1-d array containing $\\cos(2\\phi)$ evaluated at the valid time samples

        Returns
        -------
        npt.NDArray[np.number]
            A 1-d array containing $\\cos(2\\phi)$ evaluated at the valid time samples
        """
        return self.__cos2phi

    @property
    def weighted_sin(self) -> npt.NDArray[np.number]:
        """A 1-d array accumulating the noise-weighted $\\sin(2\\phi)$ sum
        per valid pixel

        Returns
        -------
        npt.NDArray[np.number]
            A 1-d array accumulating the noise-weighted $\\sin(2\\phi)$ sum
            per valid pixel
        """
        return self.__weighted_sin

    @property
    def weighted_cos(self) -> npt.NDArray[np.number]:
        """A 1-d array accumulating the noise-weighted $\\cos(2\\phi)$ sum
        per valid pixel

        Returns
        -------
        npt.NDArray[np.number]
            A 1-d array accumulating the noise-weighted $\\cos(2\\phi)$ sum
            per valid pixel
        """
        return self.__weighted_cos

    @property
    def weighted_sin_sq(self) -> npt.NDArray[np.number]:
        """A 1-d array accumulating the noise-weighted $\\sin^2(2\\phi)$ sum
        per valid pixel

        Returns
        -------
        npt.NDArray[np.number]
            A 1-d array accumulating the noise-weighted $\\sin^2(2\\phi)$ sum
            per valid pixel
        """
        return self.__weighted_sin_sq

    @property
    def weighted_cos_sq(self) -> npt.NDArray[np.number]:
        """A 1-d array accumulating the noise-weighted $\\cos^2(2\\phi)$ sum
        per valid pixel

        Returns
        -------
        npt.NDArray[np.number]
            A 1-d array accumulating the noise-weighted $\\cos^2(2\\phi)$ sum
            per valid pixel
        """
        return self.__weighted_cos_sq

    @property
    def weighted_sincos(self) -> npt.NDArray[np.number]:
        """A 1-d array accumulating the noise-weighted $\\sin(2\\phi)\\cos(2\\phi)$
        sum per valid pixel

        Returns
        -------
        npt.NDArray[np.number]
            A 1-d array accumulating the noise-weighted $\\sin(2\\phi)\\cos(2\\phi)$
            sum per valid pixel
        """
        return self.__weighted_sincos

    @property
    def one_over_determinant(self) -> npt.NDArray[np.number]:
        """A 1-d array containing the inverse determinant of the
        block-diagonal operator $P^T diag(N)^{-1} P$

        Returns
        -------
        npt.NDArray[np.number]
            A 1-d array containing the inverse determinant of the
            block-diagonal operator $P^T diag(N)^{-1} P$
        """
        return self.__one_over_determinant

    @property
    def new_npix(self) -> int:
        """The number of pixels on which the map-making will be done

        Returns
        -------
        int
            Number of pixels on which the map-making will be done
        """
        return self.__new_npix

    def get_hit_counts(self) -> npt.NDArray[np.integer]:
        """Returns hit counts of the pixel indices.

        Returns
        -------
        npt.NDArray[np.integer]
            Hit counts of the pixel indices
        """
        hit_counts = np.ma.masked_array(
            data=np.zeros(self.npix),
            mask=np.logical_not(self.__pixel_flag),
            fill_value=-1.6375e30,
        )

        hit_counts[~hit_counts.mask] = self.__hit_counts
        return hit_counts

    def _compute_weights(
        self,
        pol_angles: npt.NDArray[np.number],
        noise_weights: npt.NDArray[np.number],
    ):
        """Computes the hit counts, observed pixels, and trigonometric weights for
        map-making.

        This method allocates internal arrays for weights and accumulates values over
        all local samples, dispatching to specific C++ extensions based on the
        value of `solver_type`.

        Parameters
        ----------
        pol_angles : npt.NDArray[np.number]
            The polarization angles for each time sample
        noise_weights : npt.NDArray[np.number]
            The inverse noise variance for each time sample
        """
        self.__hit_counts = np.zeros(self.npix, dtype=self.__pointings.dtype)
        self.__weighted_counts = np.zeros(self.npix, dtype=self.dtype_float)
        self.__observed_pixels = np.zeros(self.npix, dtype=self.__pointings.dtype)
        self.__old2new_pixel = np.zeros(self.npix, dtype=self.__pointings.dtype)
        self.__pixel_flag = np.zeros(self.npix, dtype=bool)

        if self.solver_type == SolverType.I:
            self.__new_npix = compute_weights.compute_weights_pol_I(
                npix=self.npix,
                nsamples=self.nsamples,
                pointings=self.__pointings,
                pointings_flag=self.__pointings_flag,
                noise_weights=noise_weights,
                hit_counts=self.__hit_counts,
                weighted_counts=self.__weighted_counts,
                observed_pixels=self.__observed_pixels,
                __old2new_pixel=self.__old2new_pixel,  # type: ignore
                pixel_flag=self.__pixel_flag,
                comm=MPI_UTILS.comm,
            )

        else:
            self.__sin2phi = np.zeros(self.nsamples, dtype=self.dtype_float)
            self.__cos2phi = np.zeros(self.nsamples, dtype=self.dtype_float)

            self.__weighted_sin_sq = np.zeros(self.npix, dtype=self.dtype_float)
            self.__weighted_cos_sq = np.zeros(self.npix, dtype=self.dtype_float)
            self.__weighted_sincos = np.zeros(self.npix, dtype=self.dtype_float)

            self.__one_over_determinant = np.zeros(self.npix, dtype=self.dtype_float)

            if self.solver_type == SolverType.QU:
                compute_weights.compute_weights_pol_QU(
                    npix=self.npix,
                    nsamples=self.nsamples,
                    pointings=self.__pointings,
                    pointings_flag=self.__pointings_flag,
                    noise_weights=noise_weights,
                    pol_angles=pol_angles,
                    hit_counts=self.__hit_counts,
                    weighted_counts=self.__weighted_counts,
                    sin2phi=self.__sin2phi,
                    cos2phi=self.__cos2phi,
                    weighted_sin_sq=self.__weighted_sin_sq,
                    weighted_cos_sq=self.__weighted_cos_sq,
                    weighted_sincos=self.__weighted_sincos,
                    one_over_determinant=self.__one_over_determinant,
                    comm=MPI_UTILS.comm,
                )

            elif self.solver_type == SolverType.IQU:
                self.__weighted_sin = np.zeros(self.npix, dtype=self.dtype_float)
                self.__weighted_cos = np.zeros(self.npix, dtype=self.dtype_float)

                compute_weights.compute_weights_pol_IQU(
                    npix=self.npix,
                    nsamples=self.nsamples,
                    pointings=self.__pointings,
                    pointings_flag=self.__pointings_flag,
                    noise_weights=noise_weights,
                    pol_angles=pol_angles,
                    hit_counts=self.__hit_counts,
                    weighted_counts=self.__weighted_counts,
                    sin2phi=self.__sin2phi,
                    cos2phi=self.__cos2phi,
                    weighted_sin_sq=self.__weighted_sin_sq,
                    weighted_cos_sq=self.__weighted_cos_sq,
                    weighted_sincos=self.__weighted_sincos,
                    weighted_sin=self.__weighted_sin,
                    weighted_cos=self.__weighted_cos,
                    one_over_determinant=self.__one_over_determinant,
                    comm=MPI_UTILS.comm,
                )

            self.__new_npix = compute_weights.get_pixel_mask_pol(
                solver_type=self.solver_type,
                npix=self.npix,
                threshold=self.threshold,
                hit_counts=self.__hit_counts,
                one_over_determinant=self.__one_over_determinant,
                observed_pixels=self.__observed_pixels,
                __old2new_pixel=self.__old2new_pixel,  # type: ignore
                pixel_flag=self.__pixel_flag,
            )

        self.__observed_pixels.resize(self.__new_npix, refcheck=False)

    def _repixelization(self):
        """Drops unobserved or pathological pixels to compress the memory footprint.

        This routine shrinks the allocated weight arrays to only include the `new_npix`
        observed pixels by mapping original pixel indices to contiguous block indices.
        """
        if self.solver_type == SolverType.I:
            repixelize.repixelize_pol_I(
                new_npix=self.__new_npix,
                observed_pixels=self.__observed_pixels,
                hit_counts=self.__hit_counts,
                weighted_counts=self.__weighted_counts,
            )

            self.__hit_counts.resize(self.__new_npix, refcheck=False)
            self.__weighted_counts.resize(self.__new_npix, refcheck=False)

        elif self.solver_type == SolverType.QU:
            repixelize.repixelize_pol_QU(
                new_npix=self.__new_npix,
                observed_pixels=self.__observed_pixels,
                hit_counts=self.__hit_counts,
                weighted_counts=self.__weighted_counts,
                weighted_sin_sq=self.__weighted_sin_sq,
                weighted_cos_sq=self.__weighted_cos_sq,
                weighted_sincos=self.__weighted_sincos,
                one_over_determinant=self.__one_over_determinant,
            )

            self.__hit_counts.resize(self.__new_npix, refcheck=False)
            self.__weighted_counts.resize(self.__new_npix, refcheck=False)
            self.__weighted_sin_sq.resize(self.__new_npix, refcheck=False)
            self.__weighted_cos_sq.resize(self.__new_npix, refcheck=False)
            self.__weighted_sincos.resize(self.__new_npix, refcheck=False)
            self.__one_over_determinant.resize(self.__new_npix, refcheck=False)

        elif self.solver_type == SolverType.IQU:
            repixelize.repixelize_pol_IQU(
                new_npix=self.__new_npix,
                observed_pixels=self.__observed_pixels,
                hit_counts=self.__hit_counts,
                weighted_counts=self.__weighted_counts,
                weighted_sin_sq=self.__weighted_sin_sq,
                weighted_cos_sq=self.__weighted_cos_sq,
                weighted_sincos=self.__weighted_sincos,
                weighted_sin=self.__weighted_sin,
                weighted_cos=self.__weighted_cos,
                one_over_determinant=self.__one_over_determinant,
            )

            self.__hit_counts.resize(self.__new_npix, refcheck=False)
            self.__weighted_counts.resize(self.__new_npix, refcheck=False)
            self.__weighted_sin_sq.resize(self.__new_npix, refcheck=False)
            self.__weighted_cos_sq.resize(self.__new_npix, refcheck=False)
            self.__weighted_sincos.resize(self.__new_npix, refcheck=False)
            self.__weighted_sin.resize(self.__new_npix, refcheck=False)
            self.__weighted_cos.resize(self.__new_npix, refcheck=False)
            self.__one_over_determinant.resize(self.__new_npix, refcheck=False)

    def _flag_bad_pixel_samples(self):
        """Flags individual time samples that correspond to unobserved or bad pixels.

        Updates `pointings_flag` such that any sample corresponding to a discarded
        pixel is flagged as invalid.
        """
        repixelize.flag_bad_pixel_samples(
            nsamples=self.nsamples,
            pixel_flag=self.__pixel_flag,
            old2new_pixel=self.__old2new_pixel,
            pointings=self.__pointings,
            pointings_flag=self.__pointings_flag,
        )
