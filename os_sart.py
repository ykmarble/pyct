#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from utils import *
from projector import Projector
from math import ceil, floor
import numpy

def make_subset(NoA, n_subset):
    mod = NoA % n_subset
    index1d = numpy.random.permutation(NoA)
    unit_len = NoA / n_subset
    h, t = 0, unit_len
    index2d = []
    for i in xrange(n_subset):
        if i < mod:
            t += 1
        index2d.append(index1d[h:t])
        h = t
        t += unit_len
    return index2d

def os_sart(A, data, n_subset, n_iter, recon=None, subsets=None, offset_i=0):
    if recon is None:
        recon = empty_img(A)
        recon[:, :] = 0
    img = empty_img(A)
    proj = empty_proj(A)

    NoA, NoD = proj.shape
    unit_len = NoA / n_subset
    step = NoA / unit_len
    if subsets is None:
        subsets = make_subset(NoA, n_subset)
    else:
        assert len(subsets) == n_subset

    # calc a_i+
    a_ip = empty_proj(A)
    img[:, :] = 1
    A.forward(img, a_ip)

    # calc a_+j for all subsets
    a_pj_subset = [empty_img(A) for i in xrange(n_subset)]
    proj[:, :] = 1
    for i in xrange(n_subset):
        A.partial_backward(proj, a_pj_subset[i], subsets[i], None)
        a_pj_subset[i][a_pj_subset[i]==0] = 1

    for i in xrange(offset_i, n_iter + offset_i):
        alpha = 2000. / (1999 + i)
        i_subset = i % n_subset
        cur_subset = subsets[i_subset]
        A.partial_forward(recon, proj, cur_subset, None)
        proj -= data
        proj /= -a_ip
        A.partial_backward(proj, img, cur_subset, None)
        img /= a_pj_subset[i_subset]
        recon += alpha * img
    return recon

def tv_minimize(img, derivative):
    epsilon = 1e-3
    mu = numpy.full_like(img, epsilon**2)
    # [1:-2, 1:-2] (m, n)
    # [2:-1, 1:-2] (m+1, n)
    # [1:-2, 2:-1] (m, n+1)
    # [0:-3, 1:-2] (m-1, n)
    # [1:-2, 0:-3] (m, n-1)  あんまり見てると目が腐る
    mu[1:-2, 1:-2] += (img[2:-1, 1:-2] - img[1:-2, 1:-2])**2    \
                      + (img[1:-2, 2:-1] - img[1:-2, 1:-2])**2  \
                      + (img[1:-2, 1:-2] - img[0:-3, 1:-2])**2  \
                      + (img[1:-2, 1:-2] - img[1:-2, 0:-3])**2
    numpy.sqrt(mu, mu)
    derivative[:, :] = 0
    derivative[1:-2, 1:-2] = \
      (4 * img[1:-2, 1:-2] - img[2:-1, 1:-2] - img[1:-2, 2:-1] - img[0:-3, 1:-2] - img[1:-2, 0:-3]) / mu[1:-2, 1:-2] \
      + (img[1:-2, 1:-2] - img[2:-1, 1:-2]) / mu[2:-1, 1:-2] + (img[1:-2, 1:-2] - img[1:-2, 2:-1]) / mu[1:-2, 2:-1] \
      + (img[1:-2, 1:-2] - img[0:-3, 1:-2]) / mu[0:-3, 1:-2] + (img[1:-2, 1:-2] - img[1:-2, 0:-3]) / mu[1:-2, 0:-3]

def os_sart_tv(A, data, alpha, n_iter, recon=None):
    if recon == None:
        recon = empty_img(A)
        recon[:, :] = 0
    alpha_s = 0.997
    n_subset = 20
    n_tv = 5
    img = empty_img(A)
    proj = empty_proj(A)
    NoA, NoD = proj.shape
    d = empty_img(A)
    subsets = make_subset(NoA, n_subset)
    for i in xrange(n_iter):
        print i
        for p_art in xrange(n_subset):
            os_sart(A, data, 20, 1, recon=recon, subsets=subsets, offset_i=p_art+n_subset*i)
            for p_tv in xrange(n_tv):
                tv_minimize(recon, d)
                beta = numpy.max(recon)/numpy.max(d)
                recon -= alpha * beta * d
                alpha *= alpha_s
        show_image(recon)

def main():
    import sys
    import os.path
    if len(sys.argv) != 2:
        print "Usage: {} <rawfile>"
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.exists(path):
        print "invalid path"
        sys.exit(1)
    img = load_rawimage(path)
    if img is None:
        print "invalid file"
        sys.exit(1)

    scale = 0.6
    angle_px = detector_px = width_px = img.shape[1]
    interiorA = Projector(width_px, angle_px, int(ceil(detector_px*scale)))
    interiorA.update_detectors_length(ceil(detector_px * scale))
    proj = empty_proj(interiorA)
    interiorA.forward(img, proj)
    os_sart_tv(interiorA, proj, 0.005, 1000)

if __name__ == '__main__':
    main()
