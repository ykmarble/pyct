#!/usr/bin/env python2

import utils
from worker import main
import numpy


def grad(img, out_x, out_y):
    out_x[:, :] = -img[:, :]
    out_x[:, :-1] += img[:, 1:]
    out_y[:] = -img[:]
    out_y[:-1] += img[1:]

def div_2(img_x, img_y, out):
    out[:] = 0
    out[:, 1:] += img_x[:, 1:] - img_x[:, :-1]
    out[1:] += img_y[1:] - img_y[:-1]

def cp(A, b, alpha, beta, ramda, niter,
       x=None, p=None, iter_callback=lambda *arg: 0):
    """
    Perform interior CT image reconstruction with CP method.
    @A : system matrix class
    @b : sinogram
    @alpha : step size parameter
    @beta : step sizeparameter
    @ramda : tv strength
    @niter : iteration times
    @x: initial image
    @p: initial dual variable
    @iter_callback : callback function called each iterations
    """
    if x is None:
        x = utils.zero_img(A)
    if p is None:
        p = utils.zero_proj(A)

    qx = utils.zero_img(A)
    qy = utils.zero_img(A)

    img = utils.zero_img(A)
    proj = utils.zero_proj(A)
    gradx = utils.zero_img(A)
    grady = utils.zero_img(A)

    x_bar = x.copy()

    for i in xrange(niter):

        A.forward(x_bar, proj)
        proj -= b
        p += alpha * proj
        p /= 1 + alpha

        grad(x_bar, gradx, grady)
        qx += alpha * gradx
        qy += alpha * grady

        denom = numpy.maximum(numpy.sqrt(qx * qx + qy * qy), ramda) / ramda

        qx /= denom
        qy /= denom

        x_bar[:] = -x[:]

        A.backward(p, img)
        x -= beta * img
        div_2(qx, qy, img)
        x += beta * img

        x_bar += 2 * x

        iter_callback(i, x)

    return x

if __name__ == '__main__':
    main(cp)
