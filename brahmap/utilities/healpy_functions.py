#
#   HEALPY_FUNCTIONS.PY
#   interfaces to the output function of healpy package
#
#   date: 2016-12-02
#   author: GIUSEPPE PUGLISI
#
#   Copyright (C) 2016   Giuseppe Puglisi    giuspugl@sissa.it
#


import healpy as hp
import numpy as np
import matplotlib
import os

if "DISPLAY" not in os.environ:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


def obspix2mask(obspix, nside, fname=None):
    """
    From the observed pixels to a binary mask, (``mask[obspix]=1 , 0 elsewhere``)

    **Parameters**

    - ``osbpix``:{array}
        pixels observed during the scanning of the telescope and considered
        as not pathological (ordering in the  HEALPIX pixelization).
    - ``nside``: {int}
        Healpix parameter to define the pixelization grid of the map
    - ``fname``:{str}
        path to the fits file to write the map, if set it writes onto the file

    **Returns**

    - mask :{array}


    """
    mask = np.zeros(hp.nside2npix(nside))
    mask[obspix] = 1
    if not fname is None:
        hp.write_map(fname, mask)

    return mask


def reorganize_map(mapin, obspix, npix, nside, pol, fname=None):
    """
    From the solution map of the preconditioner to a Healpix map.
    It specially splits the input array ``mapin`` which is a IQU
    for a polarization analysis in to 3 arrays ``i,q,u``.

    **Parameters**

    - ``mapin``:{array}
        solution array map (``size=npix*pol``);
    - ``obspix``:{array}
        array containing the observed pixels in the Healpix ordering;
    - ``npix``:{int}
    - ``nside``: {int}
        the same as in ``obspix2mask``;
    - ``pol``:{int}
    - ``fname``:{str}

    **Returns**

    - healpix_map:{list of arrays}
         pixelized map  with Healpix.

    """

    healpix_npix = hp.nside2npix(nside)

    if pol == 3:
        healpix_map = np.zeros(healpix_npix * pol).reshape((healpix_npix, pol))
        i = mapin[::3]
        q, u = mapin[1::3], mapin[2::3]

        m = np.where(q != 0.0)[0]
        healpix_map[obspix, 0] = i
        healpix_map[obspix, 1] = q
        healpix_map[obspix, 2] = u
        hp_list = [healpix_map[:, 0], healpix_map[:, 1], healpix_map[:, 2]]
    if pol == 2:
        healpix_map = np.zeros(healpix_npix * pol).reshape((healpix_npix, pol))

        q, u = mapin[::2], mapin[1::2]

        healpix_map[obspix, 0] = q
        healpix_map[obspix, 1] = u
        hp_list = [healpix_map[:, 0], healpix_map[:, 1]]

    elif pol == 1:
        healpix_map = np.zeros(healpix_npix)

        healpix_map[obspix] = mapin
        hp_list = [healpix_map]
    if not fname is None:
        hp.write_map(fname, hp_list)

    return hp_list


def show_map(outm, pol, patch, figname=None, title="", **kwargs):
    """
    Output the map `outm` to screen or to a file.

    **Parameters**

    - ``outm`` :
        map in the fullsky format;
    - ``pol`` : {int}
    -  ``patch``: {str}
        Key to a dictionary to get the equatorial coordinates given a name patch (Polarbear collaboration
        is now observing in 3 patches: `ra23`, `ra12`, `lst4p5`);
    - ``figname`` : {str}
        If unset, outputs on screen;
    - ``norm`` : {str}
        key to the normalization of the color scale, ( `None`, `hist`, `log`)
    - ``kwargs`` : {dict}
        gnomview arguments
    """
    coord_dict = {"ra23": [-14.7, -33.09], "LP": [2.5, -53.5]}
    runcase = {1: "T", 2: "QU", 3: "TQU"}
    keys = runcase[pol]

    if not "rot" in kwargs:
        kwargs["rot"] = coord_dict[patch]

    for k, i in zip(list(keys), range(pol)):
        unseen = np.where(outm[i] == 0)[0]
        outm[i][unseen] = hp.UNSEEN
        nplots = 100 + pol * 10 + i + 1
        plt.suptitle(title, fontsize=20)
        hp.gnomview(outm[i], sub=nplots, title=k, **kwargs)

    if figname is None:
        plt.show()
    else:
        plt.savefig(figname)
        plt.close()


def subtract_offset(mapp, obspix, pol):
    """
    remove the average from the observed pixels  of ``mapp``.
    """
    if pol == 1:
        average = np.mean(mapp[obspix])
        mapp[obspix] -= average
    else:
        for i in range(len(mapp)):
            average = np.mean(mapp[i][obspix])
            mapp[i][obspix] -= average

    return mapp


def compare_maps(outm, inm, pol, patch, figname=None, remove_offset=True, **kwargs):
    """
    Output on device or in file the input map,  the output one processed from datastream
    and their difference.

    **Parameters**

    - ``outm`` :{array,list}
        map in the `.fits` format;
    - ``inm``:{array,list}
        input `.fits` map to be compared  with `outm`;
    - ``pol`` : {int}
        see :func:`show_map`;
    -  ``patch``: {str}
        Key to a dictionary to get the equatorial coordinates given a name patch, see :func:`show_map`;
    -  ``mask``:{array}
        binary map (0=unobserved, 1=observed pixels);
    - ``figname`` : {str}
        If unset, outputs on screen;
    - ``remove_offset``:{bool}
        If True removes the monopole from the input map,`inm`, in the observed region;
    - ``norm`` : {str}
        key to the normalization of the color scale, ( `None`, `hist`, `log`)

    """
    coord_dict = {"ra23": [-14.7, -33.09], "LP": [2.5, -53.5]}

    if not "rot" in kwargs:
        kwargs["rot"] = coord_dict[patch]
    if pol == 1:
        unseen = np.ma.masked_equal(outm, 0).mask
        observ = ~unseen
    else:
        unseen = np.ma.masked_equal(outm, 0).mask
        observ = ~unseen

    if remove_offset:
        inm = subtract_offset(inm, observ, pol)
        outm = subtract_offset(outm, observ, pol)
    if pol == 1:
        inm[unseen] = hp.UNSEEN
        outm[unseen] = hp.UNSEEN
        hp.gnomview(inm, title="T input map", sub=131, **kwargs)
        # hp.graticule(dpar=5,dmer=5,local=True)
        hp.gnomview(outm, title="T reconstructed map", sub=132, **kwargs)
        # hp.graticule(dpar=5,dmer=5,local=True)
        diff = inm - outm
        diff[unseen] = hp.UNSEEN
        del kwargs["min"]
        del kwargs["max"]
        hp.gnomview(diff, title="T diff", sub=133, **kwargs)
        # hp.graticule(dpar=5,dmer=5,local=True)

    elif pol == 3:
        strnmap = ["T", "Q", "U"]
        figcount = 231
        for i in [1, 2]:
            inm[i][unseen] = hp.UNSEEN
            outm[i][unseen] = hp.UNSEEN
            hp.gnomview(inm[i], title=strnmap[i] + " input map", sub=figcount, **kwargs)
            # hp.graticule(dpar=5,dmer=5,local=True)
            figcount += 1
            hp.gnomview(
                outm[i], title=strnmap[i] + " output map", sub=figcount, **kwargs
            )
            # hp.graticule(dpar=5,dmer=5,local=True)
            figcount += 1
            diff = inm[i] - outm[i]
            diff[unseen] = hp.UNSEEN
            try:
                del kwargs["min"]
                del kwargs["max"]
            except KeyError:
                pass
            hp.gnomview(diff, title=strnmap[i] + " diff", sub=figcount, **kwargs)
            # hp.graticule(dpar=5,dmer=5,local=True)
            figcount += 1

    elif pol == 2:
        strnmap = ["Q", "U"]
        figcount = 231
        for i in range(2):
            inm[i][unseen] = hp.UNSEEN
            outm[i][unseen] = hp.UNSEEN
            hp.gnomview(inm[i], title=strnmap[i] + " input map", sub=figcount, **kwargs)
            # hp.graticule(dpar=5,dmer=5,local=True)
            figcount += 1
            hp.gnomview(
                outm[i], title=strnmap[i] + " reconstructed map", sub=figcount, **kwargs
            )
            # hp.graticule(dpar=5,dmer=5,local=True)
            figcount += 1
            diff = inm[i] - outm[i]
            diff[unseen] = hp.UNSEEN
            try:
                del kwargs["min"]
                del kwargs["max"]
            except KeyError:
                pass
            hp.gnomview((diff), title=strnmap[i] + " diff", sub=figcount, **kwargs)
            # hp.graticule(dpar=5,dmer=5,local=True)
            figcount += 1

    if figname is None:
        plt.show()
    else:
        plt.savefig(figname)
        plt.close()
    pass
