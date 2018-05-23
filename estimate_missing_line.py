#!/usr/bin/env python2

import utils
from worker import main
import numpy


def estimate_missing_line(A, alpha, niter, phi_x=lambda x: x, phi_y=lambda y: y,
                  G=lambda y: y, x=None, y=None, iter_callback=lambda *arg: 0):
    """
    Reconstruct image x from data b such that Ax = b.
    @A: projection matrix
    @alpha: step size
    @niter: maximum iteration number
    @phi_x: projection function for image domain
    @phi_x: projection function for prijection domain
    @G: filter function
    @x: initial image, which will be updated after reconstruction by solution
    @y: initial projection, which will be updated after reconstruction by solution
    @iter_callback: callback function which will be called when finished each iteration
    """

    if x is None:
        x = utils.zero_img(A)

    if y is None:
        y = utils.zero_img(A)

    img = utils.zero_img(A)
    proj = utils.zero_proj(A)
    A.forward(x, proj)

    for i in xrange(niter):
        A.forward(x, proj)
        phi_y(proj)
        G(proj)
        A.backward(proj, img)
        x = (1 - alpha) * x + alpha * img
        phi_x(x)
        iter_callback(i, x, y)

    return x


if __name__ == "__main__":
    main(estimate_missing_line)
