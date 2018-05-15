#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from worker import main
import utils
import os_sart
import numpy


def tv_derivative(img, derivative):
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


def os_sart_tv(A, b, os_alpha=0.9, tv_alpha=0.01, tv_alpha_s=0.9997,
               nsubset=20, ntv=5, niter=1000, x=None, iter_callback=lambda *x: None):
    if x is None:
        x = utils.zero_img(A)
    img = utils.empty_img(A)
    proj = utils.empty_proj(A)
    NoA, NoD = proj.shape
    d = utils.empty_img(A)
    subsets = os_sart.make_subset(NoA, nsubset)
    a_ip, a_pj_subset = os_sart.calc_scale(A, subsets)
    for i in xrange(niter):
        for p_art in xrange(nsubset):
            os_sart.os_sart_mainloop(A, b, x, os_alpha, a_ip, a_pj_subset, subsets, p_art, img, proj)
            for p_tv in xrange(ntv):
                tv_derivative(x, d)
                beta = numpy.max(x) / numpy.max(d)
                x -= tv_alpha * beta * d
                tv_alpha *= tv_alpha_s
        iter_callback(i, x, proj)


if __name__ == '__main__':
    main(os_sart_tv)
