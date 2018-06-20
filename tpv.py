#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import utils
import ctfilter
import numpy
from tv_denoise import grad, div_2
from worker import main


def tpv(A, b, p, v, ramda, eps, tau, sigma, eta, niter, iter_callback=lambda *arg: 0):
    """
    Perform interior CT image reconstruction with Total p-Variation method.
    @A : system matrix class
    @b : sinogram data
    @p : use Lp norm
    @v : parameter
    @ramda : parameter
    @eps : re-projection error tolerance
    @tau : step size
    @sigma : step size
    @eta : smoothing parameter
    @niter : number of iteration times
    @iter_callback : callback function for each iterations
    """

    x = utils.zero_img(A)
    x_bar = utils.zero_img(A)
    y = utils.zero_proj(A)
    zh = utils.zero_img(A)
    zv = utils.zero_img(A)

    img = utils.zero_img(A)
    proj = utils.zero_proj(A)
    grad_h = utils.zero_img(A)
    grad_v = utils.zero_img(A)

    last_wh = utils.zero_img(A)
    last_wv = utils.zero_img(A)

    for i in xrange(niter):
        A.forward(x_bar, proj)
        proj -= b
        ctfilter.shepp_logan_filter(proj)
        y += sigma * proj
        l2_y = numpy.sqrt(numpy.add.reduce(numpy.square(y), axis=None))
        y *= max(1 - sigma * eps / l2_y, 0)

        grad(x_bar, grad_h, grad_v)
        wh = numpy.power(numpy.sqrt(eta**2 + numpy.square(grad_h)) / eta, p - 1)
        wv = numpy.power(numpy.sqrt(eta**2 + numpy.square(grad_v)) / eta, p - 1)

        #dw = numpy.sqrt(numpy.add.reduce(numpy.square(wh - last_wh), axis=None)
        #                + numpy.add.reduce(numpy.square(wv - last_wv), axis=None))
        #
        #last_wh = wh
        #last_wv = wv

        zh += sigma * v * grad_h
        zv += sigma * v * grad_v
        zh *= (ramda * wh) / numpy.fmax(ramda * wh, v * zh)
        zv *= (ramda * wv) / numpy.fmax(ramda * wv, v * zv)

        x_bar[:, :] = -x
        div_2(zh, zv, img)
        x -= tau * v * img
        A.backward(y, img)
        x -= tau * img
        x_bar += 2 * x

        iter_callback(i, x, y, zh, zv)


if __name__ == '__main__':
    main(tpv)
