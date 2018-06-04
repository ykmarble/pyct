#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import utils
import projector
import numpy
import math
import sys
from skimage.restoration import denoise_tv_bregman as sk_tv

viewer = None

def freq_hilbert_filter(f, scale=10):
    j = complex("j")
    assert len(f.shape) == 2
    n = f.shape[1] * scale
    F = numpy.fft.fft(f, n)
    H = numpy.zeros(n, dtype=complex)
    H[1:n/2] = -j
    H[n/2+1:] = j
    F *= H
    return numpy.fft.ifft(F).real


def inv_freq_hilbert_filter(f):
    return -freq_hilbert_filter(f, 1)[:, :256]


def infinite_hilbert_filter(f):
    assert len(f.shape) == 2
    narray, length = f.shape
    filter_width = length + 1 if length % 2 == 0 else length  # must be a odd number larger than NoD
    filter_x = numpy.linspace(-(filter_width / 2), filter_width / 2, filter_width)
    filter_x = numpy.arange(-(filter_width / 2), filter_width / 2 + 1)
    ci = filter_width / 2
    odds = [(i - ci) % 2 == 1 for i in xrange(filter_width)]
    filter_h = numpy.zeros_like(filter_x, dtype=float)
    filter_h[odds] = 2. / filter_x[odds] / math.pi
    print filter_h
    h = numpy.zeros((f.shape[0], f.shape[1] + filter_h.shape[0] - 1))
    for i in xrange(f.shape[0]):
        h[i] = numpy.convolve(f[i], filter_h, "full")
    return h


def inv_infinite_hilbert_filter(f):
    return -infinite_hilbert_filter(f)
    full = -hilbert_filter(f)
    utils.show_image(full)
    length = full.shape[1]
    if length % 2 == 0:
        NoD = length / 2
        s = NoD / 2
        return full[:, s:s+NoD]
    else:
        NoD = (length + 1) / 2
        s = (NoD - 1) / 2
        return full[:, s:s+NoD]


def hilbert_filter(f):
    assert len(f.shape) == 2
    h = numpy.zeros_like(f)
    narray, length = f.shape
    filter_width = length + 1 if length % 2 == 0 else length  # must be a odd number larger than NoD
    filter_x = numpy.linspace(-(filter_width / 2), filter_width / 2, filter_width)
    ci = filter_width / 2
    odds = [(i - ci) % 2 == 1 for i in xrange(filter_width)]
    filter_h = numpy.zeros_like(filter_x)
    filter_h[odds] = 2. / filter_x[odds] / math.pi
    proj3 = numpy.concatenate((numpy.zeros((narray, length/2)), f, numpy.zeros((narray, length/2))), axis=1)  # padding 0
    for i in xrange(f.shape[0]):
        h[i] = numpy.convolve(filter_h, proj3[i], "valid")
    return h


def inv_hilbert_filter(f):
    return -hilbert_filter(f)


def dbp(A, proj, img, theta=0):
    """
    Perform dbp in geometry defined by system matrix A.
    """
    assert 0 <= theta < math.pi

    # partial differential of r
    proj_dr = proj.copy()
    proj_dr[:, 1:] -= proj_dr[:, :-1]
    proj_dr[:, 0] = 0
    proj_dr /= 2

    # decide initial sgn and sgn flipping point
    #sgn = 1 if math.cos(-theta) >= 0 else -1
    #if sgn > 0:  # theta < pi/2
    #    flip_idx = int(math.ceil((theta + math.pi/2) / A.dtheta))
    #else:  # theta >= pi/2
    #    flip_idx = int(math.ceil((theta - math.pi/2) / A.dtheta))

    sgn = 1 if math.sin(-theta) > 0 else -1
    flip_idx = int(math.ceil(theta / A.dtheta))

    proj_dr[:flip_idx] *= sgn
    proj_dr[flip_idx:] *= -sgn

    doffset = A.detectors_offset
    A.update_detectors_offset(doffset + 0.5)
    A.backward(proj_dr, img)
    A.update_detectors_offset(doffset)
    img /= 256


def dbp_pocs_reconstruction(A, b, roi, prior, prior_mask, sup_mask=None, niter=1000):
    """
    ROI reconstruction algorithm by DBP-POCS method.
    Prior region is very restricted because of implementation difficulty.
    Hilbert lines are along with x-axis.
    """
    if sup_mask is None:
        sup_mask = utils.zero_img(A)
        sup_mask[:] = 1

    prior = prior * prior_mask

    hilbert_img = utils.zero_img(A)
    dbp(A, b, hilbert_img, 0)

    energy = numpy.zeros(A.NoI)
    for i in xrange(A.NoI):
        idx = A.convidx_img2r(i, 0, 0)
        if 0 <= idx < len(energy) and roi[i, A.NoI / 2] == 1:
            energy[i] = b[0, idx]

    p4_mask = sup_mask - sup_mask * prior_mask  # maybe 2d
    p4_mask[energy == 0] = 0
    p4_energy = energy - numpy.sum(prior, axis=1)
    p4_denom = numpy.sum(p4_mask, axis=1)
    p4_denom[p4_denom == 0] = 1

    x = utils.zero_img(A)

    global viewer

    for i in xrange(niter):
        # p1 data constraint
        fullh = freq_hilbert_filter(x)
        h = fullh[:, :256]
        h[roi == 1] = hilbert_img[roi == 1]
        x = inv_freq_hilbert_filter(fullh)

        # p2 support constraint
        x[sup_mask == 0] = 0

        # p3 known constraint
        x[prior_mask == 1] = prior[prior_mask == 1]

        # p4 energy constraint: line integral along with hilbert line must be the same with observed energy
        x += ((p4_energy - numpy.sum(x * p4_mask, axis=1)) / p4_denom)[:, None]*p4_mask

        # p5 non-negative constraint
        x[x < 0] = 0

        x[:, :] = sk_tv(x, 10000)

        viewer(i, x)


def test_total_hilbert_conversions_and_dbp(img):
    NoI = NoA = NoD = img.shape[0]
    A = projector.Projector(NoI, NoA, NoD)
    proj = utils.zero_proj(A)
    A.forward(img, proj)

    dbp_img = utils.zero_img(A)
    dbp(A, proj, dbp_img, 0)

    hilbert_img = freq_hilbert_filter(img)
    print numpy.sum(hilbert_img.real), numpy.sum(hilbert_img.imag)

    # must be the same image
    print "h {}, {}".format(numpy.min(hilbert_img), numpy.max(hilbert_img))
    print "d {}, {}".format(numpy.min(dbp_img), numpy.max(dbp_img))
    utils.show_image(hilbert_img.real)
    #utils.show_image(dbp_img)

    # original image will be showen
    inv = inv_freq_hilbert_filter(hilbert_img)
    print "{}, {}".format(numpy.min(inv), numpy.max(inv))
    utils.show_image(inv)
    utils.show_image(inv-img)



def main():
    scale = 0.5

    img = utils.load_rawimage(sys.argv[1])
    #test_total_hilbert_conversions_and_dbp(img)

    NoI = NoA = NoD = img.shape[0]
    A = projector.Projector(NoI, NoA, NoD)
    A.update_detectors_length(0.5 * NoI)

    proj = utils.zero_proj(A)
    A.forward(img, proj)

    roi = utils.zero_img(A)
    utils.create_elipse_mask(((NoI-1)/2., (NoI-1)/2.), (NoI-1)/2.*scale-2, (NoI-1)/2.*scale-2, roi)

    sup = utils.zero_img(A)
    utils.create_elipse_mask(((NoI-1)/2., (NoI-1)/2.), (NoI-1)/2.-5, (NoI-1)/2.-20, sup)

    tr_hil = img.copy()
    hilbert_filter(tr_hil)

    known = img.copy()
    known_mask = utils.zero_img(A)
    known_mask[:, NoI/2-2:NoI/2+2] = 1
    known_mask[roi != 1] = 0
    known[known_mask != 1] = 0

    global viewer
    viewer = utils.IterViewer(img, proj, roi, A, clim=(0.3, 0.45))

    dbp_pocs_reconstruction(A, proj, roi, known, known_mask, sup)

if __name__ == '__main__':
    main()
