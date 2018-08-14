#!/usr/bin/env python3

from pyct import utils
from pyct.ctfilter import finite_hilbert_filter, inv_finite_hilbert_filter
from pyct.cProjector import Projector
import numpy
import math
import sys
from skimage.restoration import denoise_tv_bregman as sk_tv

viewer = None

def dbp(A, proj, img, theta=0):
    """
    Perform dbp in geometry defined by system matrix A.
    """
    th_sgn = 1
    if theta >= math.pi:
        theta -= math.pi
        th_sgn = -1

    assert 0 <= theta < math.pi

    # partial differential of r
    proj_dr = numpy.zeros_like(proj.copy())
    proj_dr[:, 1:-1] += proj[:, 1:-1] - proj[:, :-2]
    proj_dr[:, 1:-1] += proj[:, 2:] - proj[:, 1:-1]
    proj_dr /= 2.


    # decide initial sgn and sgn flipping point
    #sgn = 1 if math.cos(-theta) >= 0 else -1
    #if sgn > 0:  # theta < pi/2
    #    flip_idx = int(math.ceil((theta + math.pi/2) / A.dtheta))
    #else:  # theta >= pi/2
    #    flip_idx = int(math.ceil((theta - math.pi/2) / A.dtheta))

    sgn = 1 if math.sin(-theta) > 0 else -1
    sgn *= th_sgn
    flip_idx = int(math.ceil(theta / A.dtheta))

    proj_dr[:flip_idx] *= sgn
    proj_dr[flip_idx:] *= -sgn

    #doffset = A.detectors_offset
    #A.update_detectors_offset(doffset + 0.5)
    A.backward(proj_dr, img)
    img *= -1
    #A.update_detectors_offset(doffset)


def dbp_pocs_reconstruction(A, b, roi, prior, prior_mask, sup_mask=None, niter=1000, W_sup = 0.85):
    """
    ROI reconstruction algorithm by DBP-POCS method.
    Prior region is very restricted because of implementation difficulty.
    Hilbert lines are along with x-axis.
    """
    pocs2 = True

    if sup_mask is None:
        sup_mask = utils.zero_img(A)
        sup_mask[:] = 1

    prior = prior * prior_mask

    hilbert_img = utils.zero_img(A)
    inv_hilbert_img = utils.zero_img(A)
    dbp(A, b, hilbert_img, 0)
    dbp(A, b, inv_hilbert_img, math.pi)

    W = numpy.ones(hilbert_img.shape[1], dtype=float)
    H = freq_hilbert_filter
    Ht = inv_freq_hilbert_filter
    if pocs2:
        W = numpy.sqrt(1 - numpy.linspace(-W_sup, W_sup, hilbert_img.shape[1])**2)
        H = finite_hilbert_filter
        Ht = inv_finite_hilbert_filter

    energy = numpy.zeros(A.NoI)
    for i in range(A.NoI):
        idx = A.convidx_img2r(i, 0, 0)
        if 0 <= idx < len(energy) and roi[i, A.NoI // 2] == 1:
            energy[i] = b[0, idx]

    p4_mask = sup_mask - sup_mask * prior_mask  # maybe 2d
    p4_mask[energy == 0] = 0
    p4_energy = energy - numpy.sum(prior, axis=1)
    p4_denom = numpy.sum(p4_mask / W, axis=1)
    p4_denom[p4_denom == 0] = 1

    x = utils.zero_img(A)

    global viewer

    for i in range(niter):
        # p1 data constraint
        C = numpy.sum(x, axis=1)[:, None] / W / math.pi  / 100
        if i % 2 == 0:
            fullh = -H(x)
            h = fullh[:, :256]
            h[roi == 1] = inv_hilbert_img[roi == 1]
            x = -Ht(fullh)
        else:
            fullh = H(x)
            h = fullh[:, :256]
            h[roi == 1] = hilbert_img[roi == 1]
            x = Ht(fullh)
        if pocs2:
            x += C

        # p2 support constraint
        x[sup_mask == 0] = 0

        # p3 known constraint
        x[prior_mask == 1] = prior[prior_mask == 1]

        # p4 energy constraint: line integral along with hilbert line must be the same with observed energy
        x += ((p4_energy - numpy.sum(x * p4_mask, axis=1)) / p4_denom)[:, None] * p4_mask
        x /= W

        # p5 non-negative constraint
        x[x < 0] = 0

        #x[:, :] = sk_tv(x, 10000)
        x[:, :] = sk_tv(x, 500)

        viewer(i, x)



def test_total_hilbert_conversions_and_dbp(img):
    scale = 1
    NoI = NoA = NoD = img.shape[0]
    A = Projector(NoI, NoA, NoD)
    proj = utils.create_sinogram(img, NoA, NoD, scale)
    A.forward(img, proj)

    dbp_img = utils.zero_img(A)
    dbp(A, proj, dbp_img, 0)

    hilbert_img = finite_hilbert_filter(img, W_sup)

    # must be the same image
    print("h {}, {}".format(numpy.min(hilbert_img), numpy.max(hilbert_img)))
    print("d {}, {}".format(numpy.min(dbp_img), numpy.max(dbp_img)))

    roi = utils.zero_img(A)
    utils.create_elipse_mask(((NoI-1)/2., (NoI-1)/2.), (NoI-1)/2.*scale-2, (NoI-1)/2.*scale-2, roi)

    # original image will be showen
    inv = inv_finite_hilbert_filter(hilbert_img, W_sup)
    W = numpy.sqrt(1 - numpy.linspace(-W_sup, W_sup, inv.shape[1])**2)
    C = numpy.sum(inv, axis=1)[:, None] / W / math.pi / 23
    utils.show_image(C)
    inv += C
    print("{}, {}".format(numpy.min(inv), numpy.max(inv)))
    utils.show_image(img, clim=(0.45, 0.55))
    utils.show_image(inv, clim=(0.45, 0.55))

def printl(l):
    print(l)

def main():
    scale = 1

    img = utils.load_rawimage(sys.argv[1])
    test_total_hilbert_conversions_and_dbp(img)

    NoI = img.shape[0]
    NoA = NoI
    NoD = int(NoI * scale)
    A = Projector(NoI, NoA, NoD)
    A.update_detectors_length(scale * NoI)
    bigA = Projector(NoI, NoA, NoD)
    bigA.update_detectors_length(scale * NoI)

    proj = utils.create_sinogram(img, NoA, NoD, scale)

    roi = utils.zero_img(A)
    utils.create_elipse_mask(((NoI-1)/2., (NoI-1)/2.), (NoI-1)/2.*scale-2, (NoI-1)/2.*scale-2, roi)

    sup = utils.zero_img(A)
    utils.create_elipse_mask(((NoI-1)/2., (NoI-1)/2.), (NoI-1)/2., (NoI-1)/2., sup)

    tr_hil = img.copy()
    freq_hilbert_filter(tr_hil)

    known = img.copy()
    known_mask = utils.zero_img(A)
    known_mask[:, NoI//2-5:NoI//2+7] = 1
    known_mask[roi != 1] = 0
    known[known_mask != 1] = 0

    global viewer
    viewer = utils.IterViewer(img, proj, roi, A, clim=(0.45, 0.55))
    #viewer = utils.IterLogger(img, proj, roi, A, no_forward=True)

    dbp_pocs_reconstruction(bigA, proj, roi, known, known_mask, sup, niter=2000)

if __name__ == '__main__':
    main()
