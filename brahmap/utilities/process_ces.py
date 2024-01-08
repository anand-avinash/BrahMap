#
#   PROCESS_CES.PY
#   class to preprocess and flag bad data
#   date: 2016-12-02
#   author: GIUSEPPE PUGLISI
#
#   Copyright (C) 2016   Giuseppe Puglisi    giuspugl@sissa.it
#


import numpy as np

# from scipy import weave
# import weave
# from weave import inline
from .utilities_functions import is_sorted, bash_colors


class ProcessTimeSamples(object):
    """
    This class  precomputes quantities needed during the analysis once the input file have been read.
    During its initialization,  a private member function :func:`initializeweights`
    is called to precompute arrays needed for the explicit implementation of :class:`BlockDiagonalPreconditionerLO`
     and :class:`BlockDiagonalLO`.
    Moreover it masks all the unobserved or pathological pixels which won't be taken into account,
    via the functions  :func:`repixelization`  and   :func:`flagging_samples`.
    .. note::

        This the reason why the value ``npix`` has to be updated after the removal
         of the pathological pixels.

    **Parameters**

    - ``npix`` : {int}
        total number of pixels that could be observed;
    - ``pixs`` : {array}
        list of pixels observed in the time domain;
    - ``pol`` : {int,[*default* `pol=1`]}
        process an intensity only (``pol=1``), polarization only ``pol=2``
        and intensity+polarization map (``pol=3``);
    - ``phi``: {array, [*default* `None`]}
        array with polarization angles (needed if ``pol=3,2``);
    - ``w``: {array, [*default* `None`]}
        array with noise weights , :math:`w_t= N^{-1} _{tt}`, computed by
        :func:`BlockLO.build_blocks`.   If it is  not set :func:`ProcessTimeSamples.initializeweights`
        assumes it to be a :func:`numpy.ones` array;
    - ``obspix``:{array}
        Map from the internal pixelization to an external one, i.e. HEALPIX, it has to be modified when
        pathological pixels are not taken into account;
        Default is :func:`numpy.arange(npix)`;
    - ``threshold_cond``: {float}
        set the condition number threshold to mask bad conditioned pixels (it's used in polarization cases).
        Default is set to 1.e3.


    """

    def __init__(
        self,
        pixs,
        npix,
        obspix=None,
        pol=1,
        phi=None,
        w=None,
        ground=None,
        threshold_cond=1.0e3,
        obspix2=None,
    ):
        self.pixs = pixs

        self.oldnpix = npix
        self.nsamples = len(pixs)
        self.pol = pol
        self.bashc = bash_colors()
        if w is None:
            w = np.ones(self.nsamples)
        if obspix is None:
            obspix = np.arange(self.nsamples)
        self.obspix = obspix
        if ground is not None:
            neg_groundbins = np.ma.masked_less(ground, 0)
            ground[neg_groundbins.mask] = -1
            pixs[neg_groundbins.mask] = -1
        if obspix2 is None:
            self.threshold = threshold_cond
            self.initializeweights(phi, w)
            self.new_repixelization()
            # self.repixelization()
            self.flagging_samples()
        else:
            self.SetObspix(obspix2)
            self.flagging_samples()
            self.compute_arrays(phi, w)

        if ground is not None:
            # flag the ground array as the pixs one is
            flags = np.ma.masked_equal(pixs, -1)
            ground[flags.mask] = -1
            self.ground = ground

    @property
    def get_new_pixel(self):
        return self.__new_npix, self.obspix

    def SetObspix(self, new_obspix):
        self.old2new = np.full(self.oldnpix, -1, dtype=np.int32)
        if not (is_sorted(self.obspix) and is_sorted(new_obspix)):
            print(self.bashc.warning("Obspix isn't sorted, sorting it.."))
            indexsorted = np.argsort(self.obspix, kind="quicksort")
            self.obspix = self.obspix[indexsorted]

        index_of_new_in_old_obspix = np.searchsorted(self.obspix, new_obspix)
        self.old2new[index_of_new_in_old_obspix] = np.arange(
            len(index_of_new_in_old_obspix)
        )
        try:
            assert np.allclose(new_obspix, self.obspix[index_of_new_in_old_obspix])
        except Exception:
            print(
                self.bashc.fail(
                    "new_obspix contains pixels which aren't in the old obspix"
                )
            )

        self.obspix = new_obspix
        self.__new_npix = len(new_obspix)
        print(self.bashc.bold("NT=%d\tNPIX=%d" % (self.nsamples, self.__new_npix)))

    def compute_arrays(self, phi, w):
        npix = self.__new_npix
        N = self.nsamples
        pixs = self.pixs
        if self.pol == 1:
            self.counts = np.zeros(npix)
            counts = self.counts
            # includes=r"""
            # #include <stdio.h>
            # #include <omp.h>
            # #include <math.h>
            # """
            # code ="""
            # int i,pixel;
            # for ( i=0;i<N;++i){
            #     pixel=pixs(i);
            #     if (pixel == -1) continue;
            #     counts(pixel)+=w(i);
            #     }
            #     """
            # inline(code,['pixs','w','counts','N'],verbose=1,
            # extra_compile_args=['-march=native  -O3  -fopenmp ' ],
            # 	    support_code = includes,libraries=['gomp'],type_converters=weave.converters.blitz)

            for i in range(N):
                pixel = pixs[i]
                if pixel == -1:
                    continue
                counts[pixel] += w[i]
        else:
            self.cos = np.cos(2.0 * phi)
            self.sin = np.sin(2.0 * phi)
            self.cos2 = np.zeros(npix)
            self.sin2 = np.zeros(npix)
            self.sincos = np.zeros(npix)
            cos, sin = self.cos, self.sin
            cos2, sin2, sincos = self.cos2, self.sin2, self.sincos
            if self.pol == 2:
                # includes=r"""
                # #include <stdio.h>
                # #include <omp.h>
                # #include <math.h>
                # """
                # code = """
                # int i,pixel;
                # for ( i=0;i<N;++i){
                #     pixel=pixs(i);
                #     if (pixel == -1) continue;
                #     cos2(pixel)     +=  w(i)*cos(i)*cos(i);
                #     sin2(pixel)     +=  w(i)*sin(i)*sin(i);
                #     sincos(pixel)   +=  w(i)*sin(i)*cos(i);
                #     }
                # """
                # inline(code,['pixs','w','cos','sin','cos2','sin2','sincos','N'],verbose=1,
                # extra_compile_args=['-march=native  -O3  -fopenmp ' ],
                # support_code = includes,libraries=['gomp'],type_converters=weave.converters.blitz)

                for i in range(N):
                    pixel = pixs[i]
                    if pixel == -1:
                        continue
                    cos2[pixel] += w[i] * cos[i] * cos[i]
                    sin2[pixel] += w[i] * sin[i] * sin[i]
                    sincos[pixel] += w[i] * sin[i] * cos[i]

            elif self.pol == 3:
                self.counts = np.zeros(npix)
                self.cosine = np.zeros(npix)
                self.sine = np.zeros(npix)
                counts, cosine, sine = self.counts, self.cosine, self.sine
                # includes=r"""
                # #include <stdio.h>
                # #include <omp.h>
                # #include <math.h>
                # """
                # code = """
                # int i,pixel;
                # for ( i=0;i<N;++i){
                #     pixel=pixs(i);
                #     if (pixel == -1) continue;
                #     counts(pixel)   +=  w(i);
                #     cosine(pixel)   +=  w(i)*cos(i);
                #     sine(pixel)     +=  w(i)*sin(i);
                #     cos2(pixel)     +=  w(i)*cos(i)*cos(i);
                #     sin2(pixel)     +=  w(i)*sin(i)*sin(i);
                #     sincos(pixel)   +=  w(i)*sin(i)*cos(i);
                # }
                # """
                # inline(code,['pixs','w','cos','sin','counts','cosine','sine','cos2','sin2','sincos','N'],
                # extra_compile_args=['-march=native  -O3  -fopenmp ' ],verbose=1,
                # support_code = includes,libraries=['gomp'],type_converters=weave.converters.blitz)

                for i in range(N):
                    pixel = pixs[i]
                    if pixel == -1:
                        continue
                    counts[pixel] += w[i]
                    cosine[pixel] += w[i] * cos[i]
                    sine[pixel] += w[i] * sin[i]
                    cos2[pixel] += w[i] * cos[i] * cos[i]
                    sin2[pixel] += w[i] * sin[i] * sin[i]
                    sincos[pixel] += w[i] * sin[i] * cos[i]

    def new_repixelization(self):
        includes = r"""
            #include <stdio.h>
            #include <ctype.h>
            #include <stdlib.h>
            #include <math.h>
            """  # noqa: F841
        code_I = """
            int Nnew=0;
            int Nrem=0;

            int boolval;
            for (int jpix=0; jpix<Nold; jpix++){
            	boolval=0;
            	for (int i = 0; i< Nm; ++i){
            		if (mask(i)==jpix){
               			boolval=1;
            			break;
               		}
               		else{
               			continue;
               		}
               	}
               if (boolval==1) {
               		old2new(jpix)=Nnew;
               		counts(Nnew)= counts(jpix);
               		obspix(Nnew)= obspix(jpix);
               		Nnew++;
               }
               else{
               		old2new(jpix)=-1;
               		Nrem++;
               }
            }
            newN(0)=Nnew;
            newN(1)=Nrem;
            """  # noqa: F841
        code_QU = """
            int Nnew=0;
            int Nrem=0;

            int boolval;
            for (int jpix=0; jpix<Nold; jpix++){
            	boolval=0;
            	for (int i = 0; i< Nm; ++i){
            		if (mask(i)==jpix){
               			boolval=1;
            			break;
               		}
               		else{
               			continue;
               		}
               	}
               if (boolval==1) {
               		old2new(jpix)=Nnew;
               		cos2(Nnew)= cos2(jpix);
               		sin2(Nnew)= sin2(jpix);
               		sincos(Nnew)= sincos(jpix);
               		obspix(Nnew)= obspix(jpix);
               		Nnew++;
               }
               else{
               		old2new(jpix)=-1;
               		Nrem++;
               }
            }
            newN(0)=Nnew;
            newN(1)=Nrem;
            """  # noqa: F841
        code_IQU = """
            int Nnew=0;
            int Nrem=0;

            int boolval;
            for (int jpix=0; jpix<Nold; jpix++){
            	boolval=0;
            	for (int i = 0; i< Nm; ++i){
            		if (mask(i)==jpix){
               			boolval=1;
            			break;
               		}
               		else{
               			continue;
               		}
               	}
               if (boolval==1) {
               		old2new(jpix)=Nnew;
                    cos2(Nnew)= cos2(jpix);
               		sin2(Nnew)= sin2(jpix);
               		sincos(Nnew)= sincos(jpix);
               		cosine(Nnew)= cosine(jpix);
               		sine(Nnew)= sine(jpix);
               		counts(Nnew)= counts(jpix);
               		obspix(Nnew)= obspix(jpix);
               		Nnew++;
               }
               else{
               		old2new(jpix)=-1;
               		Nrem++;
               }
            }
            newN(0)=Nnew;
            newN(1)=Nrem;
            """  # noqa: F841
        Nold = self.oldnpix
        newN = np.zeros(2)
        old2new = np.zeros(Nold, dtype=int)
        mask = self.mask
        obspix = self.obspix
        Nm = len(mask)
        # print(Nold, newN, obspix.shape, Nm)
        if self.pol == 1:
            counts = self.counts
            # listarrays = ["obspix", "mask", "Nold", "old2new", "newN", "Nm", "counts"]
            # inline(
            #     code_I,
            #     listarrays,
            #     extra_compile_args=["-march=native  -O3 "],
            #     verbose=1,
            #     support_code=includes,
            #     type_converters=weave.converters.blitz,
            # )

            Nnew = 0
            Nrem = 0

            for jpix in range(Nold):
                boolval = 0
                for i in range(Nm):
                    if mask[i] == jpix:
                        boolval = 1
                        break
                    else:
                        continue

                if boolval == 1:
                    old2new[jpix] = Nnew
                    counts[Nnew] = counts[jpix]
                    obspix[Nnew] = obspix[jpix]
                    Nnew += 1
                else:
                    old2new[jpix] = -1
                    Nrem += 1

            newN[0] = Nnew
            newN[1] = Nrem

            n_new_pix, n_removed_pix = int(newN[0]), int(newN[1])

            # print("SELF.COUNTS = ", self.counts.shape)
            # resize arrays
            # self.counts =
            self.counts = np.delete(self.counts, range(n_new_pix, self.oldnpix))
            # import pdb

            # pdb.set_trace()
            # print("SELF.COUNTS = ", self.counts.shape)

        elif self.pol == 2:
            cos2 = self.cos2
            sin2 = self.sin2
            sincos = self.sincos
            # listarrays = [
            #     "obspix",
            #     "mask",
            #     "Nold",
            #     "old2new",
            #     "newN",
            #     "Nm",
            #     "cos2",
            #     "sin2",
            #     "sincos",
            # ]
            # inline(
            #     code_QU,
            #     listarrays,
            #     extra_compile_args=["-march=native  -O3 "],
            #     verbose=1,
            #     support_code=includes,
            #     type_converters=weave.converters.blitz,
            # )

            Nnew = 0
            Nrem = 0

            for jpix in range(Nold):
                boolval = 0
                for i in range(Nm):
                    if mask[i] == jpix:
                        boolval = 1
                        break
                    else:
                        continue

                if boolval == 1:
                    old2new[jpix] = Nnew
                    cos2[Nnew] = cos2[jpix]
                    sin2[Nnew] = sin2[jpix]
                    sincos[Nnew] = sincos[jpix]
                    obspix[Nnew] = obspix[jpix]
                    Nnew += 1
                else:
                    old2new[jpix] = -1
                    Nrem += 1

            newN[0] = Nnew
            newN[1] = Nrem

            n_new_pix, n_removed_pix = int(newN[0]), int(newN[1])
            # resize arrays
            self.cos2 = np.delete(self.cos2, range(n_new_pix, self.oldnpix))
            self.sin2 = np.delete(self.sin2, range(n_new_pix, self.oldnpix))
            self.sincos = np.delete(self.sincos, range(n_new_pix, self.oldnpix))
        elif self.pol == 3:
            cos2 = self.cos2
            sin2 = self.sin2
            sincos = self.sincos
            cosine = self.cosine
            sine = self.sine
            counts = self.counts
            # listarrays = [
            #     "obspix",
            #     "mask",
            #     "Nold",
            #     "old2new",
            #     "newN",
            #     "Nm",
            #     "cos2",
            #     "sin2",
            #     "sincos",
            #     "sine",
            #     "cosine",
            #     "counts",
            # ]
            # inline(
            #     code_IQU,
            #     listarrays,
            #     extra_compile_args=["-march=native  -O3 "],
            #     verbose=1,
            #     support_code=includes,
            #     type_converters=weave.converters.blitz,
            # )

            Nnew = 0
            Nrem = 0

            for jpix in range(Nold):
                boolval = 0
                for i in range(Nm):
                    if mask[i] == jpix:
                        boolval = 1
                        break
                    else:
                        continue

                if boolval == 1:
                    old2new[jpix] = Nnew
                    cos2[Nnew] = cos2[jpix]
                    sin2[Nnew] = sin2[jpix]
                    sincos[Nnew] = sincos[jpix]
                    cosine[Nnew] = cosine[jpix]
                    sine[Nnew] = sine[jpix]
                    counts[Nnew] = counts[jpix]
                    obspix[Nnew] = obspix[jpix]
                    Nnew += 1
                else:
                    old2new[jpix] = -1
                    Nrem += 1

            newN[0] = Nnew
            newN[1] = Nrem

            n_new_pix, n_removed_pix = int(newN[0]), int(newN[1])
            # resize arrays
            self.cos2 = np.delete(self.cos2, range(n_new_pix, self.oldnpix))
            self.sin2 = np.delete(self.sin2, range(n_new_pix, self.oldnpix))
            self.sincos = np.delete(self.sincos, range(n_new_pix, self.oldnpix))
            self.sine = np.delete(self.sine, range(n_new_pix, self.oldnpix))
            self.cosine = np.delete(self.cosine, range(n_new_pix, self.oldnpix))
            self.counts = np.delete(self.counts, range(n_new_pix, self.oldnpix))

        print(self.bashc.header("___" * 30))
        print(
            self.bashc.blue(
                "Found %d pathological pixels\nRepixelizing  w/ %d pixels."
                % (n_removed_pix, n_new_pix)
            )
        )
        print(self.bashc.header("___" * 30))
        # resizing all the arrays
        self.obspix = np.delete(self.obspix, range(n_new_pix, self.oldnpix))
        self.old2new = old2new
        self.__new_npix = n_new_pix
        print(self.bashc.bold("NT=%d\tNPIX=%d" % (self.nsamples, self.__new_npix)))

    def repixelization(self):
        """
        Performs pixel reordering by excluding all the unbserved or
        pathological pixels.
        """
        n_new_pix = 0
        n_removed_pix = 0
        self.old2new = np.zeros(self.oldnpix, dtype=int)
        if self.pol == 1:
            for jpix in range(self.oldnpix):
                if jpix in self.mask:
                    self.old2new[jpix] = n_new_pix
                    self.counts[n_new_pix] = self.counts[jpix]
                    self.obspix[n_new_pix] = self.obspix[jpix]
                    n_new_pix += 1
                else:
                    self.old2new[jpix] = -1
                    n_removed_pix += 1
            # resize array
            self.counts = np.delete(self.counts, range(n_new_pix, self.oldnpix))
        else:
            for jpix in range(self.oldnpix):
                if jpix in self.mask:
                    self.old2new[jpix] = n_new_pix
                    self.obspix[n_new_pix] = self.obspix[jpix]
                    self.cos2[n_new_pix] = self.cos2[jpix]
                    self.sin2[n_new_pix] = self.sin2[jpix]
                    self.sincos[n_new_pix] = self.sincos[jpix]
                    if self.pol == 3:
                        self.counts[n_new_pix] = self.counts[jpix]
                        self.sine[n_new_pix] = self.sine[jpix]
                        self.cosine[n_new_pix] = self.cosine[jpix]
                    n_new_pix += 1
                else:
                    self.old2new[jpix] = -1
                    n_removed_pix += 1
            # resize
            self.cos2 = np.delete(self.cos2, range(n_new_pix, self.oldnpix))
            self.sin2 = np.delete(self.sin2, range(n_new_pix, self.oldnpix))
            self.sincos = np.delete(self.sincos, range(n_new_pix, self.oldnpix))
            if self.pol == 3:
                self.counts = np.delete(self.counts, range(n_new_pix, self.oldnpix))
                self.sine = np.delete(self.sine, range(n_new_pix, self.oldnpix))
                self.cosine = np.delete(self.cosine, range(n_new_pix, self.oldnpix))
        print(self.bashc.header("___" * 30))
        print(
            self.bashc.blue(
                "Found %d pathological pixels\nRepixelizing  w/ %d pixels."
                % (n_removed_pix, n_new_pix)
            )
        )
        print(self.bashc.header("___" * 30))
        # resizing all the arrays
        self.obspix = np.delete(self.obspix, range(n_new_pix, self.oldnpix))
        self.__new_npix = n_new_pix
        print(self.bashc.bold("NT=%d\tNPIX=%d" % (self.nsamples, self.__new_npix)))

    def flagging_samples(self):
        """
        Flags the time samples related to bad pixels to -1.
        """
        N = self.nsamples
        o2n = self.old2new

        pixs = self.pixs
        # code = """
        #   int i,pixel;
        #   for ( i=0;i<N;++i){
        #     pixel=pixs(i);
        #     if (pixel == -1) continue;
        #     pixs(i)=o2n(pixel);
        #     }
        # """
        # inline(
        #     code,
        #     ["pixs", "o2n", "N"],
        #     verbose=1,
        #     extra_compile_args=["  -O3  -fopenmp "],
        #     support_code=r"""
        #            #include <stdio.h>
        #            #include <omp.h>
        #            #include <math.h>""",
        #     libraries=["gomp"],
        #     type_converters=weave.converters.blitz,
        # )

        for i in range(N):
            pixel = pixs[i]
            if pixel == -1:
                continue
            pixs[i] = o2n[pixel]

    def initializeweights(self, phi, w):
        """
        Pre-compute the quantitities needed for the implementation of :math:`(A^T A)`
        and to masks bad pixels.

        **Parameters**

        - ``counts`` :
            how many times a given pixel is observed in the timestream;
        - ``mask``:
            mask  either unobserved  (``counts=0``)  or   bad constrained pixels
            (see the ``pol=3,2`` following cases) ;
        - *If* ``pol=2``:
            the matrix :math:`(A^T A)`  is  symmetric and block-diagonal, each block
            can be written as :

            .. csv-table::

                ":math:`\sum_t cos^2 2 \phi_t`", ":math:`\sum_t sin 2\phi_t cos 2 \phi_t`"
                ":math:`\sum_t sin2 \phi_t cos 2 \phi_t`",   ":math:`\sum_t sin^2 2 \phi_t`"

            the determinant, the trace are therefore needed to compute the  eigenvalues
            of each block via the formula:

            .. math::
                \lambda_{min,max}= Tr(M)/2 \pm \sqrt{Tr^2(M)/4 - det(M)}

            being :math:`M` a ``2x2`` matrix.
            The eigenvalues are needed to define the mask of bad constrained pixels whose
            condition number is :math:`\gg 1`.

        - *If*  ``pol=3``*:
            each block of the matrix :math:`(A^T A)`  is a ``3 x 3`` matrix:

            .. csv-table::

                ":math:`n_{hits}`", ":math:`\sum_t cos 2 \phi_t`", ":math:`\sum_t sin 2 \phi_t`"
                ":math:`\sum_t cos 2 \phi_t`", ":math:`\sum_t cos^2 2 \phi_t`", ":math:`\sum_t sin 2\phi_t cos 2 \phi_t`"
                ":math:`\sum_t sin 2 \phi_t`",  ":math:`\sum_t sin2 \phi_t cos 2 \phi_t`",   ":math:`\sum_t sin^2 2 \phi_t`"

            We then define the mask of bad constrained pixels by both  considering
            the condition number similarly as in the ``pol=2`` case and the pixels
            whose count is :math:`\geq 3`.

        """
        N = self.nsamples
        pixs = self.pixs
        if self.pol == 1:
            self.counts = np.zeros(self.oldnpix)
            counts = self.counts
            # includes = r"""
            #         #include <stdio.h>
            #         #include <omp.h>
            #         #include <math.h>"""
            # code = """
            #       int i,pixel;
            #       for ( i=0;i<N;++i){
            #         pixel=pixs(i);
            #         if (pixel == -1) continue;
            #         counts(pixel)+=w(i);
            #         }
            #         """
            # inline(
            #     code,
            #     ["pixs", "w", "counts", "N"],
            #     verbose=1,
            #     extra_compile_args=["  -O3  -fopenmp "],
            #     support_code=includes,
            #     libraries=["gomp"],
            #     type_converters=weave.converters.blitz,
            # )

            for i in range(N):
                pixel = pixs[i]
                if pixel == -1:
                    continue
                # print(N, pixel, counts, w)
                counts[pixel] += w[i]

            self.mask = np.where(self.counts > 0)[0]
        else:
            self.cos = np.cos(2.0 * phi)
            self.sin = np.sin(2.0 * phi)
            self.cos2 = np.zeros(self.oldnpix)
            self.sin2 = np.zeros(self.oldnpix)
            self.sincos = np.zeros(self.oldnpix)
            cos, sin = self.cos, self.sin
            cos2, sin2, sincos = self.cos2, self.sin2, self.sincos
            if self.pol == 2:
                # includes = r"""
                #         #include <stdio.h>
                #         #include <omp.h>
                #         #include <math.h>"""
                # code = """
                #       int i,pixel;
                #       for ( i=0;i<N;++i){
                #         pixel=pixs(i);
                #         if (pixel == -1) continue;
                #         cos2(pixel)     +=  w(i)*cos(i)*cos(i);
                #         sin2(pixel)     +=  w(i)*sin(i)*sin(i);
                #         sincos(pixel)   +=  w(i)*sin(i)*cos(i);
                #         }
                #         """
                # inline(
                #     code,
                #     ["pixs", "w", "cos", "sin", "cos2", "sin2", "sincos", "N"],
                #     verbose=1,
                #     extra_compile_args=["  -O3  -fopenmp "],
                #     support_code=includes,
                #     libraries=["gomp"],
                #     type_converters=weave.converters.blitz,
                # )

                for i in range(N):
                    pixel = pixs[i]
                    if pixel == -1:
                        continue
                    cos2[pixel] += w[i] * cos[i] * cos[i]
                    sin2[pixel] += w[i] * sin[i] * sin[i]
                    sincos[pixel] += w[i] * sin[i] * cos[i]

            elif self.pol == 3:
                self.counts = np.zeros(self.oldnpix)
                self.cosine = np.zeros(self.oldnpix)
                self.sine = np.zeros(self.oldnpix)
                counts, cosine, sine = self.counts, self.cosine, self.sine
                # includes = r"""
                #         #include <stdio.h>
                #         #include <omp.h>
                #         #include <math.h>"""
                # code = """
                #       int i,pixel;
                #       for ( i=0;i<N;++i){
                #         pixel=pixs(i);
                #         if (pixel == -1) continue;
                #         counts(pixel)+=w(i);
                #         cosine(pixel)     +=  w(i)*cos(i);
                #         sine(pixel)       +=  w(i)*sin(i);
                #         cos2(pixel)     +=  w(i)*cos(i)*cos(i);
                #         sin2(pixel)     +=  w(i)*sin(i)*sin(i);
                #         sincos(pixel)   +=  w(i)*sin(i)*cos(i);
                #         }
                #         """
                # inline(
                #     code,
                #     [
                #         "pixs",
                #         "w",
                #         "cos",
                #         "sin",
                #         "counts",
                #         "cosine",
                #         "sine",
                #         "cos2",
                #         "sin2",
                #         "sincos",
                #         "N",
                #     ],
                #     extra_compile_args=["  -O3  -fopenmp "],
                #     verbose=1,
                #     support_code=includes,
                #     libraries=["gomp"],
                #     type_converters=weave.converters.blitz,
                # )

                for i in range(N):
                    pixel = pixs[i]
                    if pixel == -1:
                        continue
                    counts[pixel] += w[i]
                    cosine[pixel] += w[i] * cos[i]
                    sine[pixel] += w[i] * sin[i]
                    cos2[pixel] += w[i] * cos[i] * cos[i]
                    sin2[pixel] += w[i] * sin[i] * sin[i]
                    sincos[pixel] += w[i] * sin[i] * cos[i]

            det = (self.cos2 * self.sin2) - (self.sincos * self.sincos)
            tr = self.cos2 + self.sin2
            sqrt = np.sqrt(tr * tr / 4.0 - det)
            lambda_max = tr / 2.0 + sqrt
            lambda_min = tr / 2.0 - sqrt
            cond_num = np.abs(lambda_max / lambda_min)
            mask = np.where(cond_num <= self.threshold)[0]
            if self.pol == 2:
                self.mask = mask
            elif self.pol == 3:
                mask2 = np.where(self.counts > 2)[0]
                self.mask = np.intersect1d(mask2, mask)
        # import pdb

        # pdb.set_trace()
