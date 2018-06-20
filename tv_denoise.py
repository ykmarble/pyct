#!/usr/bin/env python2

import numpy

def grad(img, out_x, out_y):
    out_x[:, :] = 0
    out_y[:, :] = 0
    out_x[:, :-1] = img[:, 1:] - img[:, :-1]
    out_y[:-1] = img[1:] - img[:-1]

def div_2(img_x, img_y, out):
    out[:, :] = 0
    out[:, :-1] -= img_x[:, :-1]
    out[:, 1:] += img_x[:, :-1]
    out[:-1] -= img_y[:-1]
    out[1:] += img_y[:-1]

def tv_denoise_chambolle(img, alpha, max_iter=200, mask=None):
    """
    Smoothing a image with TV denoising method.
    @img : 2D image array
    @alpha : smoothness
    @max_iter : times of iteration
    @mask : areas where peformed denoising (0: disable, 1: enable)
    """
    if mask is None:
        mask = numpy.ones_like(img)
    # parameter
    tol = 0.005
    tau = 1.0 / 4  # 1 / (2 * dimension)

    # matrices
    p_x = numpy.zeros_like(img)
    p_y = numpy.zeros_like(img)
    div_p = numpy.zeros_like(img)
    grad_x = numpy.empty_like(img)
    grad_y = numpy.empty_like(img)
    last_div_p = numpy.zeros_like(img)
    denom = numpy.empty_like(img)
    for i in xrange(max_iter):
        div_2(p_x, p_y, div_p)
        grad(div_p - img / alpha, grad_x, grad_y)
        grad_x[:, :-1] *= mask[:, 1:]
        grad_y[:-1] *= mask[1:]
        denom[:] = 1
        denom += tau * numpy.sqrt(grad_x**2 + grad_y**2)
        p_x[mask==1] += tau * grad_x[mask==1]
        p_x[mask==1] /= denom[mask==1]
        p_y[mask==1] += tau * grad_y[mask==1]
        p_y[mask==1] /= denom[mask==1]
        if i != 0 and numpy.abs(div_p - last_div_p).max() < tol:
            break
        last_div_p, div_p = div_p, last_div_p
    img[mask==1] -= div_p[mask==1] * alpha
