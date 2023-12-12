#
#   LINEAROPERATORS.PY
#   collects several LinearOperator classes inheritating attributes from linop package
#   date: 2016-12-02
#   author: GIUSEPPE PUGLISI
#
#   Copyright (C) 2016   Giuseppe Puglisi    giuspugl@sissa.it
#


import math as m
import numpy as np
from ..linop import linop as lp
from ..linop import blkop as blk
import random as rd

# import weave
# from weave import inline
# import scipy.sparse.linalg as spla
from ..utilities import *

# from multiprocessing import Pool, Queue, Process
import time
import multiprocessing


class GroundFilterLO(lp.LinearOperator):
    def counts_in_groundbins(self, g):
        counts = np.zeros(self.nbins)
        N = self.n

        # includes = r"""
        # #include <stdio.h>
        # #include <omp.h>
        # #include <math.h>
        # """
        # code = """
        # int i,groundbin;
        # for ( i=0;i<N;++i){
        #     groundbin=g(i);
        #     if (groundbin == -1) continue;
        #     counts(groundbin)+=1 ;
        #     }
        #     """
        # inline(
        #     code,
        #     ["g", "counts", "N"],
        #     verbose=1,
        #     extra_compile_args=["-march=native  -O3  -fopenmp "],
        #     support_code=includes,
        #     libraries=["gomp"],
        #     type_converters=weave.converters.blitz,
        # )

        for i in range(N):
            groundbin = g[i]
            if groundbin == -1:
                continue
            counts[groundbin] += 1

        return counts

    def mult(self, v):
        return v - self.Pg * v

    def __init__(self, ground):
        self.nbins = int(max(ground)) + 1
        self.n = len(ground)
        counts = self.counts_in_groundbins(ground)
        G = SparseLO(self.nbins, self.n, ground)
        G.counts = counts
        invGtG = BlockDiagonalPreconditionerLO(G, self.nbins)

        self.Pg = G * invGtG * G.T
        super(GroundFilterLO, self).__init__(
            nargin=self.n, nargout=self.n, matvec=self.mult, symmetric=True
        )


def calculate(func, args):
    result = func(*args)
    return "%s says that %s%s = %s" % (
        multiprocessing.current_process().name,
        func.__name__,
        args,
        result,
    )


def calculatestar(args):
    return calculate(*args)


def _pickle_method(method):
    func_name = method.__func__.__name__
    obj = method.__self__
    cls = method.__self__.__class__
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


class FilterLO(lp.LinearOperator):
    """
    When applied to :math:`n_t` vector, this  operator filters out
    *Legendre  Polynomial* components from it up to a certain order.
    In the simple case of a :math:`0-th` order polynomial the effect
     of filter consists of subtracting the offset from the samples.


    **Parameters**

    - ``size``: {int}
        the size of the input array;
    - ``subscan_nsample``: {list of 2 array}
        - ``subscan_nsample[0]``, contains the size of each chunk of the samples
            which has to be processed;
         - ``subscan_nsample[1]``, contains the starting sample index  of  each chunk;

    - ``samples_per_bolopair``:{list of int }
        Number of samples observed during one Constant Elevation Scan (CES) for
        any pair of detectors. If more CES are included it is a ``list of int``;
    - ``bolos_per_ces``:{list of int}
        Number of pairs of detectors that observed during a CES.
    - ``pix_samples``: {array}
        the same argument as in :class:`SparseLO`, encoding all the pixels observed
        during observations.
    - ``poly_order``: {int}
        maximum order of polynomials to be subtracted from the samples.

    .. note::

        To be consistent with tha analysis :class:`FilterLO` does not take into account
        all the flagged samples.


    """

    def mult(self, d):
        vec_out = d * 0.0
        pixs = self.pixels
        offset = 0
        mask = np.ma.masked_greater_equal(pixs, 0).mask
        for subsc, ts, ns, nb in zip(
            self.subscans, self.tstart, self.nsamples, self.nbolos
        ):
            n = nb * ns
            bolo_iter = 0
            while bolo_iter < nb:
                for i, j in zip(subsc, ts):
                    start = j + (ns * bolo_iter) + offset
                    end = start + i
                    # code = r"""
                    #     int j;
                    #     double mean=0.;
                    #     double counter=0.;
                    #     int tstart=start;
                    #     int tend=end;
                    #     for (j=tstart;j < tend;++j){
                    #         if (pixs(j) == -1){
                    #             continue;
                    #         }
                    #         mean+= d(j);
                    #         counter+=1.;
                    #     }
                    #     mean=mean/ counter;
                    #     return_val=mean;
                    #     """
                    # dmean = inline(
                    #     code,
                    #     ["pixs", "d", "start", "end"],
                    #     verbose=1,
                    #     extra_compile_args=[" -O3  -fopenmp"],
                    #     support_code=r"""
                    #                    #include <stdio.h>
                    #                    #include <math.h>""",
                    #     type_converters=weave.converters.blitz,
                    # )

                    mean = 0.0
                    counter = 0.0
                    tstart = start
                    tend = end
                    for j in range(tstart, tend):
                        if pixs[j] == -1:
                            continue
                        mean += d[j]
                        counter += 1.0

                    dmean = mean / counter

                    if np.isinf(dmean) or np.isnan(dmean):
                        continue
                    vec_out[start:end] = d[start:end] - dmean
                bolo_iter += 1
            offset += n
        return vec_out

    def polyfilter(self, d):
        vec_out = d * 0.0
        pixs = self.pixels
        offset = 0
        mask = np.ma.masked_greater_equal(pixs, 0).mask
        for subsc, ts, ns, nb in zip(
            self.subscans, self.tstart, self.nsamples, self.nbolos
        ):
            n = nb * ns
            bolo_iter = 0

            while bolo_iter < nb:
                for i, j in zip(subsc, ts):
                    start = j + (ns * bolo_iter) + offset
                    end = start + i
                    tmpmask = mask[start:end]
                    size = len(np.where(tmpmask == True)[0])
                    if size <= self.poly_order:
                        # too few samples to filter Legendre Polynomials
                        continue

                    legendres = self.legendres[i]
                    if size != i:
                        # orthonormalize the basis in the unflagged
                        # samples by a  QR decomposition
                        q, r = np.linalg.qr(legendres[tmpmask])
                        legendres = q

                    p = np.zeros(size)
                    for k in range(self.poly_order + 1):
                        # normalize Polynomial basis
                        filterbasis = legendres[:, k]
                        p += scalprod(filterbasis, d[start:end][tmpmask]) * filterbasis
                    vec_out[start:end][tmpmask] = d[start:end][tmpmask] - p
                bolo_iter += 1
            offset += n
        return vec_out

    def compute_legendres(self):
        subscan_sizes = []
        for array in self.subscans:
            for i in array:
                if not subscan_sizes.__contains__(i):
                    subscan_sizes.append(i)
                else:
                    continue
        self.legendres = {
            size: get_legendre_polynomials(self.poly_order, size)
            for size in subscan_sizes
        }

    def procsfilter(self, args):
        d, subsc, ts, ns, nb, mask = args
        vec_out = d * 0.0
        for bolo_iter in range(nb):
            for i, j in zip(subsc, ts):
                start = j + ns * bolo_iter
                end = start + i
                tmpmask = mask[start:end]
                size = len(np.where(tmpmask == True)[0])
                if size <= self.poly_order:
                    continue
                legendres = self.legendres[i]
                if size != i:
                    q, r = np.linalg.qr(legendres[tmpmask])
                    legendres = q

                    p = np.zeros(size)
                    for k in range(self.poly_order + 1):
                        filterbasis = legendres[:, k]
                        p += scalprod(filterbasis, d[start:end][tmpmask]) * filterbasis
                        vec_out[start:end][tmpmask] = d[start:end][tmpmask] - p
                else:
                    p = np.zeros(size)
                    for k in range(self.poly_order + 1):
                        filterbasis = legendres[:, k]
                        p += scalprod(filterbasis, d[start:end]) * filterbasis
                        vec_out[start:end] = d[start:end] - p

        return vec_out

    def polyfilter_multithreads(self, d):
        vec_out = d * 0.0
        pixs = self.pixels
        func = lambda ns, nb: ns * nb
        offsend = list(map(func, self.nsamples, self.nbolos))
        offstart = offsend[:-1]
        offstart.insert(0, 0)
        offsend = np.cumsum(offsend)
        offstart = np.cumsum(offstart)
        dces = [d[i:j] for i, j in zip(offstart, offsend)]
        mask = np.ma.masked_greater_equal(pixs, 0).mask
        maskces = [mask[i:j] for i, j in zip(offstart, offsend)]
        polyorder = [self.poly_order] * len(dces)
        leg = [self.legendres] * len(dces)
        vec = self.procs.map(
            globalprocsfilter,
            list(
                zip(
                    dces,
                    self.subscans,
                    self.tstart,
                    self.nsamples,
                    self.nbolos,
                    maskces,
                    polyorder,
                    leg,
                )
            ),
        )
        return np.concatenate(vec)

    def __init__(
        self,
        size,
        subscan_nsample,
        samples_per_bolopair,
        bolos_per_ces,
        pix_samples,
        poly_order=0,
        npool=4,
    ):
        self.n = size
        self.nsamples = samples_per_bolopair
        self.nbolos = bolos_per_ces
        self.subscans = subscan_nsample[0]
        self.tstart = subscan_nsample[1]

        if not (type(self.nsamples) is list):
            self.nsamples = [self.nsamples]
            self.nbolos = [self.nbolos]
            self.subscans = [self.subscans]
            self.tstart = [self.tstart]
        self.pixels = pix_samples
        self.poly_order = poly_order
        if poly_order == 0:
            super(FilterLO, self).__init__(
                nargin=size, nargout=size, matvec=self.mult, symmetric=False
            )
        elif poly_order > 0:
            self.procs = Pool(npool)
            self.compute_legendres()
            super(FilterLO, self).__init__(
                nargin=size,
                nargout=size,
                matvec=self.polyfilter_multithreads,
                symmetric=False,
            )


def globalprocsfilter(args):
    d, subsc, ts, ns, nb, mask, poly_order, legendredict = args
    # start = time.clock()
    # 		subscan_sizes=[]
    # 		for s in subsc :
    # 			if not subscan_sizes.__contains__(s):
    # 				subscan_sizes.append(s)
    # 			else : continue

    # 		legendredict={size: get_legendre_polynomials(poly_order,size) for size in subscan_sizes}
    vec_out = d * 0.0
    for bolo_iter in range(nb):
        for i, j in zip(subsc, ts):
            start = j + ns * bolo_iter
            end = start + i
            tmpmask = mask[start:end]
            size = len(np.where(tmpmask == True)[0])
            if size <= poly_order:
                continue
            legendres = legendredict[i]
            if size != i:
                q, r = np.linalg.qr(legendres[tmpmask])
                legendres = q

                p = np.zeros(size)
                for k in range(poly_order + 1):
                    filterbasis = legendres[:, k]
                    p += scalprod(filterbasis, d[start:end][tmpmask]) * filterbasis
                    vec_out[start:end][tmpmask] = d[start:end][tmpmask] - p
            else:
                p = np.zeros(size)
                for k in range(poly_order + 1):
                    filterbasis = legendres[:, k]
                    p += scalprod(filterbasis, d[start:end]) * filterbasis
                    vec_out[start:end] = d[start:end] - p
    return vec_out


class SparseLO(lp.LinearOperator):
    """
    Derived class from the one from the  :class:`LinearOperator` in :mod:`linop`.
    It constitutes an interface for dealing with the projection operator
    (pointing matrix).

    Since this can be represented as a sparse matrix, it is initialized \
    by an array of observed pixels which resembles the  ``(i,j)`` positions \
    of the non-null elements of  the matrix,``obs_pixs``.

    **Parameters**

    - ``n`` : {int}
        size of the pixel domain ;
    - ``m`` : {int}
        size of  time domain;
        (or the non-null elements in a row of :math:`A_{i,j}`);
    - ``pix_samples`` : {array}
        list of pixels observed in the time domain,
        (or the non-null elements in a row of :math:`A_{i,j}`);
    - ``pol`` : {int,[*default* `pol=1`]}
        process an intensity only (``pol=1``), polarization only ``pol=2``
        and intensity+polarization map (``pol=3``);
    - ``angle_processed``: {:class:`ProcessTimeSamples`}
        the class (instantiated before :class:`SparseLO`)has already computed
        the :math:`\cos 2\phi` and :math:`\sin 2\phi`, we link the ``cos`` and ``sin``
        attributes of this class to the  :class:`ProcessTimeSamples` ones ;

    """

    def mult(self, v):
        """
        Performs the product of a sparse matrix :math:`Av`,\
         with :math:`v` a  :mod:`numpy`  array (:math:`dim(v)=n_{pix}`)  .

        It extracts the components of :math:`v` corresponding  to the non-null \
        elements of the operator.

        """
        x = np.zeros(self.nrows)
        Nrows = self.nrows
        pixs = self.pairs
        # code = r"""
        # int i;

        # for (i=0;i<Nrows;++i){
        #     if (pixs(i) == -1) continue;
        #     x(i)+= v(pixs(i));
        #     }
        # """
        # res = inline(
        #     code,
        #     ["pixs", "v", "x", "Nrows"],
        #     verbose=1,
        #     extra_compile_args=["  -O3  -fopenmp "],
        #     support_code=r"""
        #                #include <stdio.h>
        #                #include <omp.h>
        #                #include <math.h>""",
        #     libraries=["gomp"],
        #     type_converters=weave.converters.blitz,
        # )

        for i in range(Nrows):
            if pixs[i] == -1:
                continue
            x[i] += v[pixs[i]]

        return x

    def rmult(self, v):
        """
        Performs the product for the transpose operator :math:`A^T`.

        """
        x = np.zeros(self.ncols)
        Nrows = self.nrows
        pixs = self.pairs

        # code = r"""
        #    int i ;
        #    for ( i=0;i<Nrows;++i){
        #     if (pixs(i) == -1) continue;

        #     x(pixs(i))+=v(i);
        #     }
        # """
        # inline(
        #     code,
        #     ["pixs", "v", "x", "Nrows"],
        #     verbose=1,
        #     extra_compile_args=["  -O3  -fopenmp "],
        #     support_code=r"""
        #            #include <stdio.h>
        #            #include <omp.h>
        #            #include <math.h>""",
        #     libraries=["gomp"],
        #     type_converters=weave.converters.blitz,
        # )

        for i in range(Nrows):
            if pixs[i] == -1:
                continue
            x[pixs[i]] += v[i]

        return x

    def mult_qu(self, v):
        """
        Performs :math:`A * v` with :math:`v` being a *polarization* vector.
        The output array will encode a linear combination of the two Stokes
        parameters,  (whose components are stored contiguously).

        .. math::
            d_t=  Q_p \cos(2\phi_t)+ U_p \sin(2\phi_t).
        """
        x = np.zeros(self.nrows)
        Nrows = self.nrows
        pixs = self.pairs
        cos, sin = self.cos, self.sin
        # code = """
        #    int i ;
        #    for ( i=0;i<Nrows;++i){
        #     if (pixs(i) == -1) continue;
        #     x(i)  +=  v(2*pixs(i)) *cos(i)
        #            +  v(2*pixs(i)+1) *sin(i);
        #     }
        # """
        # inline(
        #     code,
        #     ["pixs", "v", "x", "Nrows", "cos", "sin"],
        #     verbose=1,
        #     extra_compile_args=["  -O3  -fopenmp "],
        #     support_code=r"""
        #            #include <stdio.h>
        #            #include <omp.h>
        #            #include <math.h>""",
        #     libraries=["gomp"],
        #     type_converters=weave.converters.blitz,
        # )

        for i in range(Nrows):
            if pixs[i] == -1:
                continue
            x[i] += v[2 * pixs[i]] * cos[i]
            +v[2 * pixs[i] + 1] * sin[i]

        return x

    def rmult_qu(self, v):
        """
        Performs :math:`A^T * v`. The output vector will be a QU-map-like array.
        """
        vec_out = np.zeros(self.ncols * self.pol)
        Nrows = self.nrows
        pixs = self.pairs
        cos, sin = self.cos, self.sin
        # code = """
        #    int i;
        #    for ( i=0;i<Nrows;++i){
        #     if (pixs(i) == -1) continue;
        #     vec_out(2*pixs(i)) += v(i)*cos(i);
        #     vec_out(2*pixs(i)+1) += v(i)*sin(i);
        #     }
        # """
        # inline(
        #     code,
        #     ["pixs", "v", "vec_out", "Nrows", "cos", "sin"],
        #     verbose=1,
        #     extra_compile_args=["  -O3  -fopenmp "],
        #     support_code=r"""
        #            #include <stdio.h>
        #            #include <omp.h>
        #            #include <math.h>""",
        #     libraries=["gomp"],
        #     type_converters=weave.converters.blitz,
        # )

        for i in range(Nrows):
            if pixs[i] == -1:
                continue
            vec_out[2 * pixs[i]] += v[i] * cos[i]
            vec_out[2 * pixs[i] + 1] += v[i] * sin[i]

        return vec_out

    def mult_iqu(self, v):
        """
        Performs the product of a sparse matrix :math:`Av`,\
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
        x = np.zeros(self.nrows)
        Nrows = self.nrows
        pixs = self.pairs
        cos, sin = self.cos, self.sin
        # code = r"""
        #    int i ;
        #    for ( i=0;i<Nrows;++i){
        #     if (pixs(i) == -1) continue;
        #     x(i) +=  v(3*pixs(i))
        #           + v(3*pixs(i)+1) *cos(i)
        #           + v(3*pixs(i)+2) *sin(i);
        #     }
        # """
        # inline(
        #     code,
        #     ["pixs", "v", "x", "Nrows", "cos", "sin"],
        #     verbose=1,
        #     extra_compile_args=["  -O3  -fopenmp "],
        #     support_code=r"""
        #            #include <stdio.h>
        #            #include <omp.h>
        #            #include <math.h>""",
        #     libraries=["gomp"],
        #     type_converters=weave.converters.blitz,
        # )

        for i in range(Nrows):
            if pixs[i] == -1:
                continue
            x[i] += (
                v[3 * pixs[i]]
                + v[3 * pixs[i] + 1] * cos[i]
                + v[3 * pixs[i] + 2] * sin[i]
            )

        return x

    def rmult_iqu(self, v):
        """
        Performs the product for the transpose operator :math:`A^T` to get a IQU map-like vector.
        Since this vector resembles the pixel of 3 maps it has 3 times the size ``Npix``.
        IQU values referring to the same pixel are  contiguously stored in the memory.

        """
        x = np.zeros(self.ncols * self.pol)
        N = self.nrows
        pixs = self.pairs
        cos, sin = self.cos, self.sin
        # code = """
        #    int i;
        #    for ( i=0;i<N;++i){
        #     if (pixs(i) == -1) continue;
        #     x(3*pixs(i))   += v(i);
        #     x(3*pixs(i)+1) += v(i)*cos(i);
        #     x(3*pixs(i)+2) += v(i)*sin(i);
        #     }
        # """
        # inline(
        #     code,
        #     ["pixs", "v", "x", "N", "cos", "sin"],
        #     verbose=1,
        #     extra_compile_args=[" -O3  -fopenmp "],
        #     support_code=r"""
        #            #include <stdio.h>
        #            #include <omp.h>
        #            #include <math.h>""",
        #     libraries=["gomp"],
        #     type_converters=weave.converters.blitz,
        # )

        for i in range(N):
            if pixs[i] == -1:
                continue
            x[3 * pixs[i]] += v[i]
            x[3 * pixs[i] + 1] += v[i] * cos[i]
            x[3 * pixs[i] + 2] += v[i] * sin[i]

        return x

    def __init__(self, n, m, pix_samples, pol=1, angle_processed=None):
        self.ncols = n
        self.nrows = m
        self.pol = pol
        self.pairs = pix_samples
        if self.pol > 1:
            self.cos = angle_processed.cos
            self.sin = angle_processed.sin

        if pol == 3:
            self.__runcase = "IQU"
            super(SparseLO, self).__init__(
                nargin=self.pol * self.ncols,
                nargout=self.nrows,
                matvec=self.mult_iqu,
                symmetric=False,
                rmatvec=self.rmult_iqu,
            )
        elif pol == 1:
            self.__runcase = "I"
            super(SparseLO, self).__init__(
                nargin=self.pol * self.ncols,
                nargout=self.nrows,
                matvec=self.mult,
                symmetric=False,
                rmatvec=self.rmult,
            )
        elif pol == 2:
            self.__runcase = "QU"
            super(SparseLO, self).__init__(
                nargin=self.pol * self.ncols,
                nargout=self.nrows,
                matvec=self.mult_qu,
                symmetric=False,
                rmatvec=self.rmult_qu,
            )
        else:
            raise RuntimeError(
                "No valid polarization key set!\t=>\tpol=%d \n \
                                    Possible values are pol=%d(I),%d(QU), %d(IQU)."
                % (pol, 1, 2, 3)
            )

    @property
    def maptype(self):
        """
        Return a string depending on the map you are processing
        """
        return self.__runcase


class ToeplitzLO(lp.LinearOperator):
    """
    Derived Class from a LinearOperator. It exploit the symmetries of an ``dim x dim``
    Toeplitz matrix.
    This particular kind of matrices satisfy the following relation:

    .. math::

        A_{i,j}=A_{i+1,j+1}=a_{i-j}

    Therefore, it is enough to initialize ``A`` by mean of an array ``a`` of ``size = dim``.

    **Parameters**

    - ``a`` : {array, list}
        the array which resembles all the elements of the Toeplitz matrix;
    - ``size`` : {int}
        size of the block.

    """

    def mult(self, v):
        """
        Performs the product of a Toeplitz matrix with a vector ``x``.

        """
        val = self.array[0]
        y = val * v
        for i in range(1, len(self.array)):
            val = self.array[i]
            temp = val * v
            y[:-i] += temp[i:]
            y[i:] += temp[:-i]

        return y

    def __init__(self, a, size):
        super(ToeplitzLO, self).__init__(
            nargin=size, nargout=size, matvec=self.mult, symmetric=True
        )
        self.array = a


class WeightingLO(lp.LinearOperator):
    def mult(self, d):
        offset = 0
        oldb = 0
        for b, ns in zip(self.ndet_pairs, self.nsample_per_pair):
            for idx, w in np.ndenumerate(self.weights[oldb : b + oldb]):
                istart = idx[0] * ns + offset
                iend = (idx[0] + 1) * ns + offset
                d[istart:iend] = w * d[istart:iend]
            offset += b * ns
            oldb += b

        return d

    def __init__(self, bolos_per_ces, samples_per_bolopair, weights):
        self.ndet_pairs = bolos_per_ces
        self.nsample_per_pair = samples_per_bolopair
        self.size = np.sum([i * j for i, j in zip(samples_per_bolopair, bolos_per_ces)])
        self.weights = weights
        super(WeightingLO, self).__init__(
            nargin=self.size, nargout=self.size, matvec=self.mult, symmetric=True
        )


class BlockLO(blk.BlockDiagonalLinearOperator):
    """
    Derived class from  :mod:`blkop.BlockDiagonalLinearOperator`.
    It basically relies on the definition of a block diagonal operator,
    composed by ``nblocks``  diagonal operators with equal size .
    If it does not have any  off-diagonal terms (*default case* ), each block is a multiple  of
    the identity characterized by the  values listed in ``t`` and therefore is
    initialized by the :func:`BlockLO.build_blocks` as a :class:`linop.DiagonalOperator`.

    **Parameters**

    - ``blocksize`` : {int or list }
        size of each diagonal block, if `int` it is : :math:`blocksize= n/nblocks`.
    - ``t`` : {array}
        noise values for each block
    - ``offdiag`` : {bool, default ``False``}
        strictly  related to the way  the array ``t`` is passed (see notes ).

        .. note::

            - True : ``t`` is a list of array,
                    ``shape(t)= [nblocks,bandsize]``, to have a Toeplitz band diagonal operator,
                    :math:`bandsize != blocksize`
            - False : ``t`` is an array, ``shape(t)=[nblocks]``.
                    each block is identified by a scalar value in the diagonal.

    """

    def build_blocks(self):
        """
        Build each block of the operator either with or
        without off diagonal terms.
        Each block is initialized as a Toeplitz (either **band** or **diagonal**)
        linear operator.

        .. see also::

        ``self.diag``: {numpy array}
            the array resuming the :math:`diag(N^{-1})`.


        """

        tmplist = []
        self.blocklist = []

        for idx, elem in enumerate(self.covnoise):
            d = np.ones(self.blocksize[idx])

            # d = np.empty(self.blocksize)
            if self.isoffdiag:
                d.fill(elem[0])
                tmplist.append(d)
                self.blocklist.append(ToeplitzLO(elem, self.blocksize[idx]))
            else:
                d.fill(elem)
                tmplist.append(d)
                self.blocklist.append(lp.DiagonalOperator(d))

            # for j, i in enumerate(self.covnoise):
        self.diag = np.concatenate(tmplist)

    def __init__(self, blocksize, t, offdiag=False):
        self.__isoffdiag = offdiag
        self.blocksize = blocksize
        self.covnoise = t
        self.build_blocks()
        super(BlockLO, self).__init__(self.blocklist)

    @property
    def isoffdiag(self):
        """
        Property saying whether or not the operator has
        off-diagonal terms.
        """
        return self.__isoffdiag


class BlockDiagonalLO(lp.LinearOperator):
    """
    Explicit implementation of :math:`A \, diag(N^{-1}) A^T`, in order to save time
    in the application of the two matrices onto a vector (in this way the leading dimension  will be :math:`n_{pix}`
    instead of  :math:`n_t`).

    .. note::
        it is initialized as the  :class:`BlockDiagonalPreconditionerLO` since it involves
        computation with  the same matrices.
    """

    def __init__(self, CES, n, pol=1):
        self.size = pol * n
        self.pol = pol
        super(BlockDiagonalLO, self).__init__(
            nargin=self.size, nargout=self.size, matvec=self.mult, symmetric=True
        )
        self.pixels = np.arange(n)
        if pol == 1:
            self.counts = CES.counts
        elif pol > 1:
            self.sin2 = CES.sin2
            self.sincos = CES.sincos
            self.cos2 = CES.cos2
            if pol == 3:
                self.counts = CES.counts
                self.cos = CES.cosine
                self.sin = CES.sine

    def mult(self, x):
        """
        Multiplication of  :math:`A \, diag(N^{-1}) A^T` on to a vector math:`x`
        ( :math:`n_{pix}` array).
        """
        y = x * 0.0
        if self.pol == 1:
            y = x * self.counts
        elif self.pol == 3:
            for pix, s2, c2, cs, c, s, hits in zip(
                self.pixels,
                self.sin2,
                self.cos2,
                self.sincos,
                self.cos,
                self.sin,
                self.counts,
            ):
                y[3 * pix] = hits * x[3 * pix] + c * x[3 * pix + 1] + s * x[3 * pix + 2]
                y[3 * pix + 1] = (
                    c * x[3 * pix] + c2 * x[3 * pix + 1] + cs * x[3 * pix + 2]
                )
                y[3 * pix + 2] = (
                    s * x[3 * pix] + cs * x[3 * pix + 1] + s2 * x[3 * pix + 2]
                )
        elif self.pol == 2:
            for pix, s2, c2, cs in zip(self.pixels, self.sin2, self.cos2, self.sincos):
                y[pix * 2] = c2 * x[2 * pix] + cs * x[pix * 2 + 1]
                y[pix * 2 + 1] = cs * x[2 * pix] + s2 * x[pix * 2 + 1]
        return y


class BlockDiagonalPreconditionerLO(lp.LinearOperator):
    """
    Standard preconditioner defined as:

    .. math::

        M_{BD}=( A \, diag(N^{-1}) A^T)^{-1}

    where :math:`A` is the *pointing matrix* (see  :class:`SparseLO`).
    Such inverse operator  could be easily computed given the structure of the
    matrix :math:`A`. It could be  sparse in the case of Intensity only analysis (`pol=1`),
    block-sparse if polarization is included (`pol=3,2`).


    **Parameters**

    - ``n``:{int}
        the size of the problem, ``npix``;
    - ``CES``:{:class:`ProcessTimeSamples`}
        the linear operator related to the data sample processing. Its members (`counts`, `masks`,
        `sine`, `cosine`, etc... ) are  needed to explicitly compute the inverse of the
        :math:`n_{pix}` blocks of :math:`M_{BD}`.
    - ``pol``:{int}
        the size of each block of the matrix.
    """

    def mult(self, x):
        """
        Action of :math:`y=( A \, diag(N^{-1}) A^T)^{-1} x`,
        where :math:`x` is   an :math:`n_{pix}` array.
        """
        y = x * 0.0
        Npix = int(self.size / self.pol)
        includes = r"""
            #include <stdio.h>
            #include <ctype.h>
            #include <stdlib.h>
            #include <math.h>
            """
        if self.pol == 1:
            nan = np.ma.masked_greater(self.counts, 0)
            # print(
            #     "mbd shapes: self.counts =",
            #     self.counts.shape,
            #     "y.shape =",
            #     y.shape,
            #     "x.shape =",
            #     x.shape,
            #     "nan.shape =",
            #     nan.shape,
            # )

            # import pdb

            # pdb.set_trace()
            y[nan.mask] = x[nan.mask] / self.counts[nan.mask]
        elif self.pol == 3:
            determ = (
                self.counts * (self.cos2 * self.sin2 - self.sincos * self.sincos)
                - self.cos * self.cos * self.sin2
                - self.sin * self.sin * self.cos2
                + 2.0 * self.cos * self.sin * self.sincos
            )
            nan = np.ma.masked_greater(abs(determ), 1e-5)
            # code = """
            # for (int j=0; j<Npix; j++){
            #    if (mask(j)){
            #        y(3*j)   = (
            #        (c2(j)*s2(j)-cs(j)*cs(j) )*x(3*j)
            #        + ( s(j)*cs(j)-c(j)*s2(j) )*x(3*j+1)
            #        + ( c(j)*cs(j)-s(j)*c2(j) )*x(3*j+2)
            #        )/det(j);

            #        y(3*j+1) = (
            #        ( s(j)*cs(j)-c(j)*s2(j) )  *x(3*j)
            #        + ( hits(j)*s2(j)-s(j)*s(j) )*x(3*j+1)
            #        + ( s(j)*c(j)-hits(j)*cs(j) )*x(3*j+2)
            #        )/det(j);

            #        y(3*j+2) = (
            #        ( c(j)*cs(j) -s(j)*c2(j) ) *x(3*j)
            #        +( -hits(j)*cs(j)+c(j)*s(j) )*x(3*j+1)
            #        +( hits(j)*c2(j)-c(j)*c(j) ) *x(3*j+2)
            #        )/det(j);
            #    }
            #    else{
            #        continue;
            #    }
            # }
            # """
            mask = nan.mask
            det = determ
            c2 = self.cos2
            s2 = self.sin2
            cs = self.sincos
            hits = self.counts
            s = self.sin
            c = self.cos
            # listarrays = [
            #     "x",
            #     "y",
            #     "Npix",
            #     "mask",
            #     "c2",
            #     "s2",
            #     "cs",
            #     "s",
            #     "c",
            #     "det",
            #     "hits",
            # ]
            # inline(
            #     code,
            #     listarrays,
            #     extra_compile_args=["-march=native  -O3 "],
            #     verbose=1,
            #     support_code=includes,
            #     type_converters=weave.converters.blitz,
            # )

            for j in range(Npix):
                if mask[j]:
                    y[3 * j] = (
                        (c2[j] * s2[j] - cs[j] * cs[j]) * x[3 * j]
                        + (s[j] * cs[j] - c[j] * s2[j]) * x[3 * j + 1]
                        + (c[j] * cs[j] - s[j] * c2[j]) * x[3 * j + 2]
                    ) / det[j]
                    y[3 * j + 1] = (
                        (s[j] * cs[j] - c[j] * s2[j]) * x[3 * j]
                        + (hits[j] * s2[j] - s[j] * s[j]) * x[3 * j + 1]
                        + (s[j] * c[j] - hits[j] * cs[j]) * x[3 * j + 2]
                    ) / det[j]
                    y[3 * j + 2] = (
                        (c[j] * cs[j] - s[j] * c2[j]) * x[3 * j]
                        + (-hits[j] * cs[j] + c[j] * s[j]) * x[3 * j + 1]
                        + (hits[j] * c2[j] - c[j] * c[j]) * x[3 * j + 2]
                    ) / det[j]
                else:
                    continue

        elif self.pol == 2:
            determ = (self.cos2 * self.sin2) - (self.sincos * self.sincos)
            nan = np.ma.masked_greater(abs(determ), 1e-5)
            # code = """
            # for (int j=0; j<Npix; j++){
            #     if (mask(j)){
            #         y(2*j)  = (s2(j)*x(2*j) -cs(j)* x(2*j +1) )/det(j);
            #         y(2*j+1)=(-cs(j)*x(2*j) +c2(j)* x(2*j+ 1) )/det(j);
            #     }
            #     else{
            #         continue;
            #     }
            # }
            # """
            mask = nan.mask
            det = determ
            c2 = self.cos2
            s2 = self.sin2
            cs = self.sincos
            # listarrays = ["x", "y", "mask", "Npix", "c2", "s2", "cs", "det"]
            # inline(
            #     code,
            #     listarrays,
            #     extra_compile_args=["-march=native  -O3 "],
            #     verbose=1,
            #     support_code=includes,
            #     type_converters=weave.converters.blitz,
            # )

            # print("NAN.mask = ", nan.mask, "determinant = ", determ)
            for j in range(Npix):
                if mask[j]:
                    y[2 * j] = (s2[j] * x[2 * j] - cs[j] * x[2 * j + 1]) / det[j]
                    y[2 * j + 1] = (-cs[j] * x[2 * j] + c2[j] * x[2 * j + 1]) / det[j]
                else:
                    continue

        return y

    def __init__(self, CES, n, pol=1):
        self.size = pol * n
        self.pixels = np.arange(n)
        self.pol = pol
        if pol == 1:
            self.counts = CES.counts
        elif pol > 1:
            self.sin2 = CES.sin2
            self.cos2 = CES.cos2
            self.sincos = CES.sincos
            if pol == 3:
                self.counts = CES.counts
                self.cos = CES.cosine
                self.sin = CES.sine

        super(BlockDiagonalPreconditionerLO, self).__init__(
            nargin=self.size, nargout=self.size, matvec=self.mult, symmetric=True
        )


class InverseLO(lp.LinearOperator):
    """
    Construct the inverse operator of a matrix :math:`A`, as a linear operator.

    **Parameters**

    - ``A`` : {linear operator}
        the linear operator of the linear system to invert;
    - ``method`` : {function }
        the method to compute ``A^-1`` (see below);
    - ``P`` : {linear operator } (optional)
        the preconditioner for the computation of the inverse operator.

    """

    def mult(self, x):
        """
        It returns  :math:`y=A^{-1}x` by solving the linear system :math:`Ay=x`
        with a certain :mod:`scipy` routine (e.g. :func:`scipy.sparse.linalg.cg`)
        defined above as ``method``.
        """

        y, info = self.method(self.A, x, M=self.preconditioner)
        self.isconverged(info)
        return y

    def isconverged(self, info):
        """
        It returns a Boolean value  depending on the
        exit status of the solver.

        **Parameters**

        - ``info`` : {int}
            output of the solver method (usually :func:`scipy.sparse.cg`).



        """
        self.__converged = info
        if info == 0:
            return True
        else:
            return False

    def __init__(self, A, method=None, preconditioner=None):
        super(InverseLO, self).__init__(
            nargin=A.shape[0], nargout=A.shape[1], matvec=self.mult, symmetric=True
        )
        self.A = A
        self.__method = method
        self.__preconditioner = preconditioner
        self.__converged = None

    @property
    def method(self):
        """
        The method to compute the inverse of A. \
        It can be any :mod:`scipy.sparse.linalg` solver, namely :func:`scipy.sparse.linalg.cg`,
        :func:`scipy.sparse.linalg.bicg`, etc.

        """
        return self.__method

    @property
    def converged(self):
        """
        provides convergence information:

        - 0 : successful exit;
        - >0 : convergence to tolerance not achieved, number of iterations;
        - <0 : illegal input or breakdown.

        """
        return self.__converged

    @property
    def preconditioner(self):
        """
        Preconditioner for the solver.
        """
        return self.__preconditioner


from scipy.linalg import solve, lu, eigh


class CoarseLO(lp.LinearOperator):
    """
    This class contains all the operation involving the coarse operator :math:`E`.
    In this implementation :math:`E` is always applied to a vector wiht
    its inverse : :math:`E^{-1}`.
    When initialized it performs either an LU or an eigenvalue  decomposition
    to accelerate the performances of the inversion.

    **Parameters**

    - ``Z`` : {np.matrix}
            deflation matrix;
    - ``A`` : {SparseLO}
            to  compute vectors :math:`AZ_i`;
    - ``r`` :  {int}
            :math:`rank(Z)`, dimension of the deflation subspace;
    - ``apply``:{str}
            - ``LU``: performs LU decomposition,
            - ``eig``: compute the eigenvalues and eigenvectors of ``E``.


    """

    def mult(self, v):
        """
        Perform the multiplication of the inverse coarse operator :math:`x=E^{-1} v`.
        It exploits the LU decomposition of :math:`E` to solve the system :math:`Ex=v`.
        It first solves :math:`y=L^{-1} v` and then :math:`x=U^{-1}y`.
        """
        y = solve(self.L, v, lower=True, overwrite_b=False)
        x = solve(self.U, y, overwrite_b=True)
        return x

    def mult_eig(self, v):
        """
        Matrix vector multiplication with :math:`E^{-1}` computed via
        :func:`setting_inverse_w_eigenvalues`.
        """
        return self.invE.dot(v)

    def setting_inverse_w_eigenvalues(self, E):
        """
        This routine computes the inverse of ``E`` by a decomposition through an  eigenvalue
        decomposition. It further checks whether it has some degenerate eigenvalues,
        i.e. 0 to numerical precision (``1.e-15``), and eventually excludes these eigenvalues
        from the anaysis.
        """

        eigenvals, W = eigh(E)
        lambda_max = max(eigenvals)
        diags = eigenvals * 0.0
        threshold_to_degen = 1.0e-6
        nondegenerate = np.where(abs(eigenvals / lambda_max) > threshold_to_degen)[0]
        degenerate = np.where(abs(eigenvals / lambda_max) < threshold_to_degen)[0]
        c = bash_colors()
        if len(degenerate) != 0:
            print(c.header("===" * 30))
            print(
                c.warning(
                    "\t DISCARDING %d OUT OF %d EIGENVALUES\t"
                    % (len(degenerate), len(eigenvals))
                )
            )
            print(c.header("===" * 30))
        else:
            print(c.header("===" * 30))
            print(
                c.header(
                    "\t Matrix E is not singular, all its eigenvalues have been taken into account\t"
                )
            )
            print(c.header("===" * 30))

        for i in nondegenerate:
            diags[i] = 1.0 / eigenvals[i]
        D = np.diag(diags)
        tmp = dgemm(D.T, W)

        self.invE = dgemm(W.T, tmp.T)  # W.dot(D.dot(W.T))

    def __init__(self, Z, Az, r, apply="LU"):
        M = dgemm(Z, Az.T)
        if apply == "eig":
            self.setting_inverse_w_eigenvalues(M)
            super(CoarseLO, self).__init__(
                nargin=r, nargout=r, matvec=self.mult_eig, symmetric=True
            )
        elif apply == "LU":
            self.L, self.U = lu(M, permute_l=True, overwrite_a=True, check_finite=False)
            super(CoarseLO, self).__init__(
                nargin=r, nargout=r, matvec=self.mult, symmetric=True
            )


class DeflationLO(lp.LinearOperator):
    """
    This class builds the Deflation operator (and its transpose)
    from the columns of the matrix ``Z``.

    **Parameters**

    - ``z`` : {np.matrix}
            the deflation matrix. Its columns are read as arrays in a list ``self.z``.

    """

    def mult(self, x):
        """
        Performs the matrix vector multiplication   :math:`Z x`
        with  :math:`dim(x) = rank(Z)`.

        """
        y = np.zeros(self.nrows)
        for i in range(self.ncols):
            y += self.z[i] * x[i]  # .astype(x.dtype)
        return y

    def rmult(self, x):
        """
        Performs the product onto a ``N_pix`` vector with :math:`Z^T`.

        """
        return np.array([scalprod(i, x) for i in self.z])

    def __init__(self, z):
        self.z = []
        self.nrows, self.ncols = z.shape
        for j in range(self.ncols):
            self.z.append(z[:, j])
        super(DeflationLO, self).__init__(
            nargin=self.ncols,
            nargout=self.nrows,
            matvec=self.mult,
            symmetric=False,
            rmatvec=self.rmult,
        )
