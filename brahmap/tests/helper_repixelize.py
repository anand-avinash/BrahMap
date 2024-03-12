import numpy as np


def repixelization_pol1(oldnpix, mask, counts, obspix):
    old2new = np.zeros(oldnpix, dtype=int)
    n_new_pix, n_removed_pix = 0, 0
    for pix in range(oldnpix):
        boolval = 0
        for idx in range(len(mask)):
            if mask[idx] == pix:
                boolval = 1
                break
            else:
                continue

        if boolval == 1:
            old2new[pix] = n_new_pix
            counts[n_new_pix] = counts[pix]
            obspix[n_new_pix] = obspix[pix]
            n_new_pix += 1
        else:
            old2new[pix] -= 1
            n_removed_pix += 1

    return n_new_pix, n_removed_pix, old2new, counts, obspix


def repixelization_pol2(oldnpix, mask, counts, obspix, sin2, cos2, sincos):
    old2new = np.zeros(oldnpix, dtype=int)
    n_new_pix, n_removed_pix = 0, 0
    for pix in range(oldnpix):
        boolval = 0
        for idx in range(len(mask)):
            if mask[idx] == pix:
                boolval = 1
                break
            else:
                continue

        if boolval == 1:
            old2new[pix] = n_new_pix
            counts[n_new_pix] = counts[pix]
            obspix[n_new_pix] = obspix[pix]
            sin2[n_new_pix] = sin2[pix]
            cos2[n_new_pix] = cos2[pix]
            sincos[n_new_pix] = sincos[pix]
            n_new_pix += 1
        else:
            old2new[pix] -= 1
            n_removed_pix += 1

    return n_new_pix, n_removed_pix, old2new, counts, obspix, sin2, cos2, sincos


def repixelization_pol3(
    oldnpix, mask, counts, obspix, sin2, cos2, sincos, sine, cosine
):
    old2new = np.zeros(oldnpix, dtype=int)
    n_new_pix, n_removed_pix = 0, 0
    for pix in range(oldnpix):
        boolval = 0
        for idx in range(len(mask)):
            if mask[idx] == pix:
                boolval = 1
                break
            else:
                continue

        if boolval == 1:
            old2new[pix] = n_new_pix
            counts[n_new_pix] = counts[pix]
            obspix[n_new_pix] = obspix[pix]
            sin2[n_new_pix] = sin2[pix]
            cos2[n_new_pix] = cos2[pix]
            sincos[n_new_pix] = sincos[pix]
            sine[n_new_pix] = sine[pix]
            cosine[n_new_pix] = cosine[pix]
            n_new_pix += 1
        else:
            old2new[pix] -= 1
            n_removed_pix += 1

    return (
        n_new_pix,
        n_removed_pix,
        old2new,
        counts,
        obspix,
        sin2,
        cos2,
        sincos,
        sine,
        cosine,
    )
