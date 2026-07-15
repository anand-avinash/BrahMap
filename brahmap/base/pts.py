from enum import IntEnum
from typing import Any
import numpy as np
import numpy.typing as npt
from mpi4py import MPI

from ..utilities import bash_colors
from ..math import DTypeFloat
from ..mpi import MPI_UTILS


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


class BaseProcessTimeSamples(object):
    """Base class for processing pointing information and pre-computing
    map-making weights.

    This class provides the core container structure, properties, and
    basic verification logic for pointing arrays, polarization orientation
    angles, and noise weights.

    Specific weight accumulation and repixelization routines are implemented
    by its subclasses.

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
        self._npix = npix
        self._nsamples = len(pointings)
        self._nsamples_global = MPI_UTILS.comm.allreduce(self._nsamples, MPI.SUM)

        # Initialize attributes for static type checking
        self._new_npix: int = 0
        self._hit_counts: npt.NDArray[np.integer] = np.empty(0, dtype=int)
        self._weighted_counts: npt.NDArray[np.number] = np.empty(0)
        self._observed_pixels: npt.NDArray[np.integer] = np.empty(0, dtype=int)
        self._old2new_pixel: npt.NDArray[np.integer] = np.empty(0, dtype=int)
        self._pixel_flag: npt.NDArray[np.bool_] = np.empty(0, dtype=bool)
        self._sin2phi: npt.NDArray[np.number] = np.empty(0)
        self._cos2phi: npt.NDArray[np.number] = np.empty(0)
        self._weighted_sin: npt.NDArray[np.number] = np.empty(0)
        self._weighted_cos: npt.NDArray[np.number] = np.empty(0)
        self._weighted_sin_sq: npt.NDArray[np.number] = np.empty(0)
        self._weighted_cos_sq: npt.NDArray[np.number] = np.empty(0)
        self._weighted_sincos: npt.NDArray[np.number] = np.empty(0)
        self._one_over_determinant: npt.NDArray[np.number] = np.empty(0)
        self._dtype_float: npt.DTypeLike = np.float64

        if update_pointings_inplace:
            self._pointings = pointings
            self._pointings_flag = pointings_flag
        else:
            self._pointings = pointings.copy()
            if pointings_flag is not None:
                self._pointings_flag = pointings_flag.copy()

        if pointings_flag is None:
            self._pointings_flag = np.ones(self._nsamples, dtype=bool)

        assert self._pointings_flag is not None

        if len(self._pointings_flag) != self._nsamples:
            raise AssertionError(
                "Size of `pointings_flag` must be equal to the size of "
                "`pointings` array:\n"
                f"len(pointings_flag) = {len(self._pointings_flag)}\n"
                f"len(pointings) = {self._nsamples}"
            )

        self._solver_type = solver_type
        self._threshold = threshold

        if self.solver_type not in [1, 2, 3]:
            raise ValueError(
                "Invalid `solver_type`!!!\n`solver_type` must be either "
                "SolverType.I, SolverType.QU or SolverType.IQU "
                "(equivalently 1, 2 or 3)."
            )

        # setting the dtype for the `float` arrays
        if dtype_float is not None:
            self._dtype_float = dtype_float  # type: ignore
        elif noise_weights is not None and pol_angles is not None:
            self._dtype_float = np.promote_types(
                noise_weights.dtype,
                pol_angles.dtype,
            )
        elif noise_weights is not None:
            self._dtype_float = noise_weights.dtype
        elif pol_angles is not None:
            self._dtype_float = pol_angles.dtype
        else:
            self._dtype_float = np.float64

        if noise_weights is None:
            noise_weights = np.ones(self._nsamples, dtype=self._dtype_float)

        if len(noise_weights) != self._nsamples:
            raise AssertionError(
                "Size of `noise_weights` must be equal to the size of "
                "`pointings` array:\n"
                f"len(noise_weigths) = {len(noise_weights)}\n"
                f"len(pointings) = {self._nsamples}"
            )

        try:
            noise_weights = noise_weights.astype(
                dtype=self._dtype_float, casting="safe", copy=False
            )
        except TypeError:
            raise TypeError(
                "The `noise_weights` array has higher dtype than "
                f"`self._dtype_float={self._dtype_float}`. Please call "
                f"`ProcessTimeSamples` again with `dtype_float={noise_weights.dtype}`"
            )

        if self.solver_type != 1:
            assert pol_angles is not None
            if len(pol_angles) != self._nsamples:
                raise AssertionError(
                    "Size of `pol_angles` must be equal to the size of "
                    "`pointings` array:\n"
                    f"len(pol_angles) = {len(pol_angles)}\n"
                    f"len(pointings) = {self._nsamples}"
                )

            try:
                pol_angles = pol_angles.astype(
                    dtype=self._dtype_float, casting="safe", copy=False
                )
            except TypeError:
                raise TypeError(
                    "The `pol_angles` array has higher dtype than "
                    f"`self._dtype_float={self._dtype_float}`. Please call "
                    f"`ProcessTimeSamples` again with `dtype_float={pol_angles.dtype}`"
                )

        self._compute_weights(
            pol_angles,
            noise_weights,
        )

        if self._new_npix == 0:
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
                        f"Found {self.npix - self._new_npix} pathological pixels on the map"
                    )
                )
            )
            print(
                bc.blue(
                    bc.bold(
                        f"Map-maker will take into account only {self._new_npix} pixels"
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
        return self._npix

    @property
    def pointings(self) -> npt.NDArray[np.integer]:
        """A 1-d array of pixel indices pointing to the observed sky pixel
        for each time sample

        Returns
        -------
        npt.NDArray[np.integer]
            A 1-d array of pixel pointing indices for each time sample
        """
        return self._pointings

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
        return self._pointings_flag

    @property
    def nsamples(self) -> int:
        """The number of time samples processed by the current MPI rank

        Returns
        -------
        int
            Number of samples on current MPI rank
        """
        return self._nsamples

    @property
    def nsamples_global(self) -> int:
        """The total number of time samples across all MPI ranks

        Returns
        -------
        int
            Global number of samples
        """
        return self._nsamples_global

    @property
    def solver_type(self) -> SolverType:
        """The current map-making solver configuration ($I$, $QU$, or $IQU$)

        Returns
        -------
        SolverType
            Level of map-making: $I$, $QU$, or $IQU$
        """
        return self._solver_type

    @property
    def threshold(self) -> float:
        """The condition number threshold used to flag bad pixels

        Returns
        -------
        float
            Threshold to used for flagging the pixels in the sky
        """
        return self._threshold

    @property
    def dtype_float(self) -> Any:
        """The inferred or specified data type for floating point arrays

        Returns
        -------
        DTypeFloat
            `dtype` of the floating point arrays
        """
        return self._dtype_float

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
        return self._observed_pixels

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
        return self._pixel_flag

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
        return np.nonzero(~self._pixel_flag)[0]

    @property
    def old2new_pixel(self) -> npt.NDArray[np.integer]:
        """A 1-d array mapping old pixel indices to new pixel indices

        Returns
        -------
        npt.NDArray[np.integer]
            A 1-d array mapping old pixel indices to new pixel indices
        """
        old2new_pixel = np.where(self._pixel_flag, self._old2new_pixel, -1)
        return old2new_pixel.astype(self._pointings.dtype, copy=False)

    @property
    def weighted_counts(self) -> npt.NDArray[np.number]:
        """A 1-d array accumulating the inverse noise weights per valid pixel

        Returns
        -------
        npt.NDArray[np.number]
            A 1-d array accumulating the inverse noise weights per valid pixel
        """
        return self._weighted_counts

    @property
    def sin2phi(self) -> npt.NDArray[np.number]:
        """A 1-d array containing $\\sin(2\\phi)$ evaluated at the valid time samples

        Returns
        -------
        npt.NDArray[np.number]
            A 1-d array containing $\\sin(2\\phi)$ evaluated at the valid time samples
        """
        return self._sin2phi

    @property
    def cos2phi(self) -> npt.NDArray[np.number]:
        """A 1-d array containing $\\cos(2\\phi)$ evaluated at the valid time samples

        Returns
        -------
        npt.NDArray[np.number]
            A 1-d array containing $\\cos(2\\phi)$ evaluated at the valid time samples
        """
        return self._cos2phi

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
        return self._weighted_sin

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
        return self._weighted_cos

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
        return self._weighted_sin_sq

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
        return self._weighted_cos_sq

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
        return self._weighted_sincos

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
        return self._one_over_determinant

    @property
    def new_npix(self) -> int:
        """The number of pixels on which the map-making will be done

        Returns
        -------
        int
            Number of pixels on which the map-making will be done
        """
        return self._new_npix

    def get_hit_counts(self) -> np.ma.MaskedArray:
        """Returns hit counts of the pixel indices.

        Returns
        -------
        npt.NDArray[np.integer]
            Hit counts of the pixel indices
        """
        hit_counts = np.ma.masked_array(
            data=np.zeros(self.npix),
            mask=np.logical_not(self._pixel_flag),
            fill_value=-1.6375e30,
        )

        hit_counts[~hit_counts.mask] = self._hit_counts
        return hit_counts

    def _flag_bad_pixel_samples(self):
        from .._extensions import repixelize

        repixelize.flag_bad_pixel_samples(
            nsamples=self.nsamples,
            pixel_flag=self._pixel_flag,
            old2new_pixel=self._old2new_pixel,
            pointings=self._pointings,
            pointings_flag=self._pointings_flag,
        )

    def _compute_weights(self, pol_angles, noise_weights):
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
        raise NotImplementedError

    def _repixelization(self):
        """Drops unobserved or pathological pixels to compress the memory footprint.

        This routine shrinks the allocated weight arrays to only include the `new_npix`
        observed pixels by mapping original pixel indices to contiguous block indices.
        """
        raise NotImplementedError
