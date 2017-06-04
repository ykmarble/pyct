#!/usr/bin/env python2

from utils import *
from projector import Projector
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

def os_sart(A, data, n_subset, n_iter):
    recon = empty_img(A)
    recon[:, :] = 0
    img = empty_img(A)
    proj = empty_proj(A)

    NoA, NoD = proj.shape
    unit_len = NoA / n_subset
    step = NoA / unit_len
    subsets = make_subset(NoA, n_subset)

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

    for i in xrange(n_iter):
        alpha = 200. / (199 + i)
        print i, alpha
        i_subset = i % n_subset
        cur_subset = subsets[i_subset]
        A.partial_forward(recon, proj, cur_subset, None)
        proj -= data
        proj /= -a_ip
        A.partial_backward(proj, img, cur_subset, None)
        img /= a_pj_subset[i_subset]
        recon += alpha * img
        show_image(recon)
    return recon

def os_sart_tv(A, data, alpha, alpha_tv, n):
    alpha_s = 0.997
    alpha_tv_s = alpha_tv
    epsilon = 1e-4
    img = empty_img(A)
    proj = empty_proj(A)
    proj_next = empty_proj(A)
    mask = empty_proj(A)
    scale_proj = empty_proj(A)
    scale_img = empty_img(A)
    mu = empty_img(A)
    d = empty_img(A)

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

    angle_px = detector_px = width_px = img.shape[1]
    A = Projector(width_px, angle_px, detector_px)
    proj = empty_proj(A)
    A.forward(img, proj)
    os_sart(A, proj, 10, 10000)

if __name__ == '__main__':
    main()
