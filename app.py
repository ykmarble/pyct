#!/usr/bin/env python

import projector
import numpy

def rdiff(img):
    return 0.5 * (img[:-1] - img[1:])

def t_rdiff(img):
    h, w = img.shape
    out = img.zeros((h+1, w))
    out[:-1] += 0.5 * img
    out[1:] -= 0.5 * img
    return out

def grad(img):
    out_x = numpy.zeros_like(img)
    out_y = numpy.zeros_like(img)
    out_x[:, :-1] = img[:, 1:] - img[:, :-1]
    out_y[:-1] = img[1:] - img[:-1]
    return (out_x, out_y)

def div_2(img_x, img_y):
    out = numpy.zeros_like(img_x)
    out[:, :-1] += img_x[:, :-1]
    out[:, 1:] -= img_x[:, :-1]
    out[:-1] += img_y[:-1]
    out[1:] -= img_y[:-1]
    return out

def tv_denoise(img, alpha, max_iter=200, mask=None):
    if mask is None:
        mask = numpy.ones_like(img)
    # parameter
    tol = 0.001
    tau = 1.0 / 4  # 1 / (2 * dimension)

    # matrices
    p_x = numpy.zeros_like(img)
    p_y = numpy.zeros_like(img)
    div_p = numpy.zeros_like(img)
    grad_x = numpy.empty_like(img)
    grad_y = numpy.empty_like(img)
    last_div_p = None
    for i in xrange(max_iter):
        div_p = div_2(p_x, p_y)
        grad_x, grad_y = grad(div_p - img / alpha)
        grad_x *= mask
        grad_y *= mask
        denom = 1 + tau * numpy.sqrt(grad_x**2 + grad_y**2)
        p_x += tau * grad_x
        p_x /= denom
        p_y += tau * grad_y
        p_y /= denom
        if last_div_p is None:
            last_div_p = div_p
            continue
        print i, numpy.abs(div_p - last_div_p).max()
        if (numpy.abs(div_p - last_div_p).max() < tol):
            break
        last_div_p = div_p

    return img - div_p * alpha

def create_elipse_mask(shape, point, a, b):
    x, y = point
    a_2 = a**2.
    b_2 = b**2.
    ab_2 = a_2 * b_2
    mask = numpy.zeros(shape)
    for xi in xrange(mask.shape[1]):
        for yi in xrange(mask.shape[0]):
            if a_2 * (y - yi)**2 + b_2 * (x - xi)**2 < ab_2:
                mask[yi, xi] = 1
    return mask

def vcoord(shape, point):
    h, w = shape
    x, y = point
    return (x - w/2., h/2. - y)

def icoord(shape, point):
    h, w = shape
    x, y = point
    return (x + w/2., h/2. - y)
