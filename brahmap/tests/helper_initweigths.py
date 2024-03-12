import numpy as np


def initializeweights_pol1(nsamples, oldnpix, w, pixs):
    counts = np.zeros(oldnpix)

    for idx in range(nsamples):
        pixel = pixs[idx]
        if pixel == -1:
            continue
        counts[pixel] += w[idx]

    mask = np.where(counts > 0)[0]

    return counts, mask


def initializeweights_pol2(nsamples, oldnpix, w, pixs, phi):
    sin = np.sin(2.0 * phi)
    cos = np.cos(2.0 * phi)

    counts = np.zeros(oldnpix)
    sin2 = np.zeros(oldnpix)
    cos2 = np.zeros(oldnpix)
    sincos = np.zeros(oldnpix)

    for idx in range(nsamples):
        pixel = pixs[idx]
        if pixel == -1:
            continue
        counts[pixel] += w[idx]
        sin2[pixel] += w[idx] * sin[idx] * sin[idx]
        cos2[pixel] += w[idx] * cos[idx] * cos[idx]
        sincos[pixel] += w[idx] * sin[idx] * cos[idx]

    return counts, sin, cos, sin2, cos2, sincos


def initializeweights_pol3(nsamples, oldnpix, w, pixs, phi):
    sin = np.sin(2.0 * phi)
    cos = np.cos(2.0 * phi)

    counts = np.zeros(oldnpix)
    sine = np.zeros(oldnpix)
    cosine = np.zeros(oldnpix)
    sin2 = np.zeros(oldnpix)
    cos2 = np.zeros(oldnpix)
    sincos = np.zeros(oldnpix)

    for idx in range(nsamples):
        pixel = pixs[idx]
        if pixel == -1:
            continue
        counts[pixel] += w[idx]
        sine[pixel] += w[idx] * sin[idx]
        cosine[pixel] += w[idx] * cos[idx]
        sin2[pixel] += w[idx] * sin[idx] * sin[idx]
        cos2[pixel] += w[idx] * cos[idx] * cos[idx]
        sincos[pixel] += w[idx] * sin[idx] * cos[idx]

    return counts, sine, cosine, sin, cos, sin2, cos2, sincos


def get_mask_pol(counts, sin2, cos2, sincos, threshold):
    det = (sin2 * cos2) - (sincos * sincos)
    trace = sin2 + cos2
    sqrtf = np.sqrt(trace * trace / 4.0 - det)
    lambda_max = trace / 2.0 + sqrtf
    lambda_min = trace / 2.0 - sqrtf
    cond_num = np.abs(lambda_max / lambda_min)
    cond_num_mask = np.where(cond_num <= threshold)[0]
    count_mask = np.where(counts > 2)[0]
    final_mask = np.intersect1d(count_mask, cond_num_mask)
    return cond_num_mask, final_mask
