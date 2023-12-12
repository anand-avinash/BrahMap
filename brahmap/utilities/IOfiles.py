#
#   IOFILES.PY
#   input output routines for hdf5 data
#   date: 2016-12-02
#   author: GIUSEPPE PUGLISI
#
#   Copyright (C) 2016   Giuseppe Puglisi    giuspugl@sissa.it
#


import numpy as np
import h5py as h5
from .utilities_functions import *
from .healpy_functions import *
from functools import reduce


def read_from_data(filename, pol, npairs=None):
    """
    Read a hdf5 file of one Constant Elevation Scan preprocessed by the AnalysisBackend
    of the Polarbear Collaboration.

    **Parameters**

    - ``filename``:{str}
        path to the hdf5 file
    - ``pol``:{int}
      - ``1``: read data for intensity  data;
      - ``2``: read data for polarization data;
      - ``3``: read  for both intensity and  polarization data;

    - ``npairs``:{int}
        set how many bolo_pairs to read, default is ``None``.

    """

    f = h5.File(filename, "r")
    hp_pixs = f["obspix"][...]
    n_bolo_pair = f["n_bolo_pair"][...]
    n_ces = f["n_sample_ces"][...]
    print("Bolo Pairs: %d \t CES: %d " % (n_bolo_pair, n_ces))

    pixs_pair = []
    polang_pair = []
    d_pair = []
    weight_pair = []
    ground_pair = []
    if npairs is None:
        n_to_read = n_bolo_pair
    else:
        n_to_read = npairs

    for i in range(n_to_read):
        group = f["bolo_pair_" + str(i)]
        pixs_pair.append(group["pixel"][...])
        polang_pair.append(group["pol_angle"][...])
        ground_pair.append(group["ground"][...].astype("int"))
        if pol == 1:
            d_pair.append(group["sum"][...])
            weight_pair.append(group["weight_sum"][...])
        elif pol == 3 or pol == 2:
            d_pair.append(group["dif"][...])
            weight_pair.append(group["weight_dif"][...])
    f.close()

    d = np.concatenate(d_pair)
    weight = np.array(weight_pair)
    polang = np.concatenate(polang_pair)
    pixs = np.concatenate(pixs_pair)
    ground = np.concatenate(ground_pair)

    return d, weight, polang, pixs, hp_pixs, ground, n_ces


def read_multiple_ces(filelist, pol, npairs=None, filtersubscan=True):
    """
    Read a list of  hdf5 files of multiple CES scans preprocessed by the AnalysisBackend
    of the Polarbear Collaboration.

    **Parameters**

    - ``filelist``:{list of str}
        list containing the path to the hdf5 files
    - ``pol``:{int}
      - ``1``: read data for intensity  data;
      - ``2``: read data for polarization data;
      - ``3``: read  for both intensity  and polarization data;

    - ``npairs``:{int}
        set how many bolo_pairs to read, default is ``None``.
    - ``filtersubscan``:{bool}
        activate the subscan selection on to data (default ``True``).

    """
    if filtersubscan:
        readf = read_from_data_with_subscan_resize
        subscan, tstart = [], []
    else:
        readf = read_from_data
    bolopairs_per_ces, samples_per_bolopair = [], []
    pixs, polang, d, weight, ground, hp_pixs = [], [], [], [], [], []
    for f in filelist:
        outdata = readf(f, pol, npairs=npairs)
        d.append(outdata[0])
        weight.append(outdata[1])
        polang.append(outdata[2])
        pixs.append(outdata[3])
        ground.append(outdata[5])
        if filtersubscan:
            samples_per_bolopair.append(outdata[6])
            bolopairs_per_ces.append(outdata[7])
            subscan.append(outdata[8][0])
            tstart.append(outdata[8][1])
    hp_pixs.append(outdata[4])
    # flagging_not_in_allCES(pixs)
    if filtersubscan:
        return (
            np.concatenate(d),
            np.concatenate(weight),
            np.concatenate(polang),
            np.concatenate(pixs),
            np.concatenate(hp_pixs),
            np.concatenate(ground),
            subscan,
            tstart,
            samples_per_bolopair,
            bolopairs_per_ces,
        )
    else:
        return (
            np.concatenate(d),
            np.concatenate(weight),
            np.concatenate(polang),
            np.concatenate(pixs),
            np.concatenate(hp_pixs),
            np.concatenate(ground),
            samples_per_bolopair,
            bolopairs_per_ces,
        )


def flagging_not_in_allCES(CES_pixs):
    """
    Flag all the pixels which are not in common in  the considered  CES.
    """
    nces = len(CES_pixs)

    # compute intersection
    inters = CES_pixs[0]
    for pix in CES_pixs:
        inters = np.intersect1d(inters, pix)

    for pixs in CES_pixs:
        to_flag = np.in1d(pixs, inters, invert=True)
        pixs[to_flag] = -1

    pass


def flagging_subscan(unflagged_pix, subscan):
    """
    Flag all the samples outside a subscan.
    """
    nsamples = subscan[0]
    tstart = subscan[1]
    k = 0
    for t, n in zip(tstart, nsamples):
        unflagged_pix[k:t] = -1
        k = t + n
    # return unflag_pix


def read_from_data_with_subscan_resize(filename, pol, npairs=None):
    """
    Read a hdf5 file preprocessed by the AnalysisBackend
    of the Polarbear Collaboration by considering, as chunks of data, only the
    subscan samples.

    **Parameters**

    - ``filename``:{str}
        path to the hdf5 file
    - ``pol``:{int}
      - ``1``: read data for temperature only data;
      - ``2,3``: read  for polarization data;

    - ``npairs``:{int}
        set how many bolo_pairs to read, default is ``None``.

    """

    f = h5.File(filename, "r")
    hp_pixs = f["obspix"][...]
    n_bolo_pair = f["n_bolo_pair"][...]
    n_ces = f["n_sample_ces"][...]
    subscan = [f["subscans/n_sample"][...], f["subscans/t_start"][...]]
    # print "Bolo Pairs in this CES : %d \t #samples per Bolo Pairs: %d "%(n_bolo_pair,n_ces)
    pixs_pair = []
    polang_pair = []
    d_pair = []
    weight_pair = []
    ground_pair = []
    if npairs is None:
        n_to_read = n_bolo_pair
    else:
        n_to_read = npairs
        print("reading %d bolopairs" % n_to_read)
    for i in range(n_to_read):
        group = f["bolo_pair_" + str(i)]
        pix = group["pixel"][...]
        flagging_subscan(pix, subscan)
        pixs_pair.append(pix)
        polang_pair.append(group["pol_angle"][...])
        ground_pair.append(group["ground"][...].astype("int"))
        if pol == 1:
            d_pair.append(group["sum"][...])
            weight_pair.append(group["weight_sum"][...])
        elif pol == 3 or pol == 2:
            d_pair.append(group["dif"][...])
            weight_pair.append(group["weight_dif"][...])

    f.close()

    d = np.concatenate(d_pair)
    weight = np.array(weight_pair)
    polang = np.concatenate(polang_pair)
    pixs = np.concatenate(pixs_pair)
    ground = np.concatenate(ground_pair)
    return d, weight, polang, pixs, hp_pixs, ground, n_ces, n_to_read, subscan


def write_ritz_eigenvectors_to_hdf5(z, filename, eigvals=None):
    """
    Save to a file the approximated eigenvectors computed via the :func:`deflationlib.arnoldi`
    routine.
    """
    datatype = z[0, 0].dtype
    if datatype == "complex128":
        dt = h5.special_dtype(vlen=datatype)
    else:
        dt = h5.h5t.IEEE_F64BE

    size_eigenvectors, n_eigenvals = z.shape
    f = h5.File(filename, "w")
    eigenvect_group = f.create_group("Ritz_eigenvectors")
    eigenvect_group.create_dataset(
        "n_eigenvectors",
        np.shape(n_eigenvals),
        dtype=h5.h5t.STD_I32BE,
        data=n_eigenvals,
    )

    eig = eigenvect_group.create_dataset("Eigenvectors", data=z, chunks=True)

    if not (eigvals is None):
        eig = f.create_dataset("Ritz_eigenvalues", data=eigvals)

    f.close()
    pass


def read_ritz_eigenvectors_from_hdf5(filename, eigvals=False):
    """
    read from hdf5 file the approximated eigenvectors
    related to the deflation subspace.

    """
    f = h5.File(filename, "r")
    n_eigenvals = f["Ritz_eigenvectors/n_eigenvectors"][...]
    eigens = f["Ritz_eigenvectors/Eigenvectors"]
    z = eigens[...]
    if eigvals:
        eigenvals = f["Ritz_eigenvalues"][...]
        f.close()
        return z, n_eigenvals, eigenvals
    else:
        f.close()
        return z, n_eigenvals


def read_obspix_from_hdf5(filename):
    """
    read from hdf5 file the obspix array containing
    the observed pixels in the Healpix ordering .

    """
    f = h5.File(filename, "r")
    obsp = f["obspix"][...]
    f.close()
    return obsp


def write_obspix_to_hdf5(filename, obspix):
    """
    Save into hdf5 file the obspix array
    """
    f = h5.File(filename, "w")
    det = f.create_dataset("obspix", data=obspix, dtype=h5.h5t.STD_I32BE)
    f.close()
    pass


def write_to_hdf5(filename, obs_pixels, noise_values, d, phi=None):
    """
    Write onto hdf5 file whose datasets are created by the routine
    :func:`utilities_functions.system_setup`.

    """

    f = h5.File(filename, "w")
    group = f.create_group("bolo_pair")

    pixs = group.create_dataset("pixel", np.shape(obs_pixels), dtype=h5.h5t.STD_I32BE)
    weight = group.create_dataset(
        "weight", np.shape(noise_values), dtype=h5.h5t.IEEE_F64BE
    )
    det = group.create_dataset("sum", np.shape(d), dtype=h5.h5t.IEEE_F64BE)
    if phi is not None:
        polang = group.create_dataset(
            "pol_angle", np.shape(phi), dtype=h5.h5t.IEEE_F64BE
        )
        polang[...] = phi

    pixs[...] = obs_pixels
    weight[...] = noise_values
    det[...] = d

    f.close()
    pass


def show_matrix_form(A):
    """
    Explicit the components of the Linear Operator A as a matrix.
    """
    import matplotlib.pyplot as plt

    matr = A.to_array()
    # print matr
    maxval = matr.max()
    matr /= maxval
    imgplot = plt.imshow(matr, interpolation="nearest", vmin=matr.min(), vmax=1)
    imgplot.set_cmap("spectral")
    plt.colorbar()
    plt.show()
    pass


def read_from_hdf5(filename):
    """
    Read from a hdf5 file whose datasets are created by the routine :func:`utilities_functions.system_setup`
    """
    f = h5.File(filename, "r")

    obs_pix = f["/bolo_pair/pixel"][...]
    polang = f["/bolo_pair/pol_angle"][...]
    weight = f["/bolo_pair/weight"][...]

    det = f["bolo_pair/sum"][...]
    f.close()

    return det, obs_pix, polang, weight


def plot_histogram_eigenvalues(z):
    """
    save a plot containing an histogram  of the eigenvalues ``z``
    """

    import matplotlib.pyplot as plt
    from matplotlib import rc

    histo, edges = np.histogram(abs(z), bins=20, normed=False)
    bins = np.array([(edges[i] + edges[i + 1]) / 2.0 for i in range(len(histo))])
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Eigenvalues Histogram")
    plt.xlabel(r"$\lambda_i $")
    plt.hist(histo, bins=bins, color="b", linewidth=1.5)
    plt.savefig("data/eigenvalues_histogram.png")
    pass


def save_maplist(maplist, filename):
    """
    Save a list of maps into a hdf5 file.
    useful when one needs to store the solutions at each iteration steps
    """
    f = h5.File(filename, "w")
    nmaps = len(maplist)
    f.create_dataset("Nmaps", np.shape(nmaps), dtype=h5.h5t.STD_I32BE, data=nmaps)
    for i in range(nmaps):
        f.create_dataset(
            "Map" + str(i),
            np.shape(maplist[i]),
            dtype=h5.h5t.IEEE_F64BE,
            data=maplist[i],
        )

    f.close()
    pass


def read_maplist(filename):
    """
    Read a list of N maps from an hdf5 file.
    """
    f = h5.File(filename, "r")
    nmaps = f["Nmaps"][...]
    m = []
    for i in range(nmaps):
        m.append(np.array(f["Map" + str(i)][...]).T)

    return m, nmaps


def full2cutskymap(hp_map, pol, npix, observpix):
    """
    Convert a map from full sky in the Healpix format into an array of concatenated maps.
    Note that the ordering is the following (in the case of pol=3):
    `I0,Q0,U0, I1,Q1,U1, ...`

    - `hp_map`: {array or sequence }
        a sequence of Mealpix maps, assumed ordering  [I, Q, U ]
    """
    x = np.zeros(pol * npix)
    obsmap = [m[observpix] for m in hp_map]
    if pol == 1:
        return obsmap
    else:
        for i in range(npix):
            x[pol * i : pol * (i + 1)] = [obsmap[k][i] for k in range(pol)]
        return x


def find_common_obspix(nside, pathtofiles, n_files):
    mask = 0.0
    nside = nside
    obspix_set = []
    for offset in range(n_files):
        hp_pixs = read_obspix_from_hdf5(pathtofiles + "obspix_" + str(offset) + ".hdf5")
        obspix_set.append(hp_pixs)
        print(len(hp_pixs), "obspix", offset)
        mask += (1 + offset) * obspix2mask(hp_pixs, nside)
        hp_pixs = 0.0
    common_obsp = reduce(np.intersect1d, (obsp for obsp in obspix_set))
    print("intersection", len(common_obsp))
    write_obspix_to_hdf5(pathtofiles + "common_obspix.hdf5", common_obsp)
    return obspix_set, mask
