#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from utils import *
import projector
#import differencial
from projector import Projector
#from differencial import Projector
from math import ceil, floor
import numpy
import sys

def diff_calc_scale(A, subsets):
    # けいさんめも｡
    # 投影空間は普通に足して1/2
    # 画像空間は上0.5と下0.5のpartial_backwardを足す
    NoI, _ = A.get_image_shape()
    NoA, NoD = A.get_projector_shape()
    absorpA = projector.Projector(NoI, NoA, NoD+1)
    a_ip = empty_proj(A)
    a_pj_subset = [empty_img(A) for i in xrange(len(subsets))]

    # calc a_i+
    img = empty_img(absorpA)
    proj = empty_proj(absorpA)
    img[:, :] = 1
    absorpA.forward(img, proj)
    a_ip = 0.5 * (proj[:, :-1] + proj[:, 1:])

    # calc a_+j
    proj_u = zero_proj(absorpA)
    proj_d = zero_proj(absorpA)
    proj_u[:, :] = 0.5
    proj_d[:, :] = 0.5
    img_u = empty_img(absorpA)
    img_d = empty_img(absorpA)
    for i in xrange(len(subsets)):
        absorpA.partial_backward(proj_u, img_u, subsets[i], None)
        absorpA.partial_backward(proj_d, img_d, subsets[i], None)
        a_pj_subset[i] = img_u + img_d
        a_pj_subset[i][a_pj_subset[i]==0] = 1  # avoid zero-division

    return a_ip, a_pj_subset

def calc_scale(A, subsets):
    # 吸収版｡
    a_ip = empty_proj(A)
    img = empty_img(A)
    img[:, :] = 1
    A.forward(img, a_ip)

    a_pj_subset = [empty_img(A) for i in xrange(len(subsets))]
    proj = empty_proj(A)
    proj[:, :] = 1
    for i in xrange(len(subsets)):
        A.partial_backward(proj, a_pj_subset[i], subsets[i], None)
        a_pj_subset[i][a_pj_subset[i]==0] = 1  # avoid zero-division
        #print numpy.min(a_pj_subset[i]), numpy.max(a_pj_subset[i])
        #show_image(a_pj_subset[i])

    return a_ip, a_pj_subset

def make_subset(NoA, n_subset):
    mod = NoA % n_subset
    unit_len = NoA / n_subset
    w = (NoA+mod)/n_subset
    h = n_subset
    index1d = numpy.array([j*h + i for i in xrange(h) for j in xrange(w) if j*h + i < NoA])
    h, t = 0, unit_len
    index2d = []
    for i in xrange(n_subset):
        if i < mod:
            t += 1
        index2d.append(index1d[h:t])
        h = t
        t += unit_len
    #numpy.random.shuffle(index2d)
    return index2d

def os_sart_mainloop(A, data, recon, a_ip, a_pj_subset, subsets, i, img, proj):
    alpha = 200. / (100. + i)
    alpha = 0.9
    i_subset = i % len(subsets)
    cur_subset = subsets[i_subset]
    A.partial_forward(recon, proj, cur_subset, None)
    proj -= data
    proj /= -a_ip
    A.partial_backward(proj, img, cur_subset, None)
    img /= a_pj_subset[i_subset]
    recon += alpha * img

def os_sart(A, data, n_subset=1, n_iter=1000, recon=None, iter_callback=lambda *x: None):
    if recon is None:
        recon = empty_img(A)
        recon[:, :] = 0
    img = empty_img(A)
    proj = empty_proj(A)

    NoA, NoD = proj.shape
    subsets = make_subset(NoA, n_subset)
    a_ip, a_pj_subset = calc_scale(A, subsets)

    for i in xrange(n_iter):
        os_sart_mainloop(A, data, recon, a_ip, a_pj_subset, subsets, i, img, proj)
        iter_callback(i, recon, proj)


def tv_minimize(img, derivative):
    epsilon = 8e-3
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

def os_sart_tv(A, data, n_iter=1000, alpha=0.01, recon=None, iter_callback=lambda *x: None):
    if recon == None:
        recon = empty_img(A)
        recon[:, :] = 0
    alpha_s = 0.99997
    #alpha_s = 1
    n_subset = 20
    n_tv = 5
    img = empty_img(A)
    proj = empty_proj(A)
    NoA, NoD = proj.shape
    d = empty_img(A)
    subsets = make_subset(NoA, n_subset)
    a_ip, a_pj_subset = calc_scale(A, subsets)
    for i in xrange(n_iter):
        for p_art in xrange(n_subset):
            index = p_art+n_subset*i
            os_sart_mainloop(A, data, recon, a_ip, a_pj_subset, subsets, index, img, proj)
            for p_tv in xrange(n_tv):
                tv_minimize(recon, d)
                beta = numpy.max(recon)/numpy.max(d)
                recon -= alpha * beta * d
                alpha *= alpha_s
        iter_callback(i, recon, proj)

def main():
    import sys
    import os.path
    if len(sys.argv) != 2:
        print "Usage: {} <rawfile>"
        sys.exit(1)
    path = sys.argv[1]

    scale = 0.85
    proj, img, interiorA = create_projection(path, interior_scale=scale)

    # create roi mask
    roi = zero_img(interiorA)
    roi_c = ((roi.shape[0] - 1) / 2., (roi.shape[1] - 1) / 2.)
    roi_r = (roi.shape[0] * scale / 2., roi.shape[1] * scale / 2.)
    create_elipse_mask(roi_c, roi_r[0], roi_r[1], roi)

    (_, name, _) = decompose_path(path)
    #callback = IterViewer(img, roi, clim=(-110, 190))
    callback = IterLogger(img, roi, subname=name)

    ### CAUTION: check if scale calculataion function is properly selected (in os_sart(_tv))###
    #os_sart(interiorA, proj, n_iter=1000, iter_callback=callback) # 1497866
    os_sart_tv(interiorA, proj, n_iter=500, iter_callback=callback) # 1497866

if __name__ == '__main__':
    main()
