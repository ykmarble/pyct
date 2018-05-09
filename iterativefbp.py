#!/usr/bin/env python2

import utils
from worker import main

def iterative_fbp(A, b, alpha, niter, phi_x=lambda x: x, G=lambda y: y,
                  x=None, iter_callback=lambda *arg: 0):
    """
    Reconstruct image x from data b such that Ax = b with iterative FBP method.
    @A: projection matrix
    @b: input data
    @alpha: step size
    @niter: maximum iteration number
    @phi_x: projection function for image domain
    @G: filter function
    @x: initial image, which will be updated after reconstruction by solution
    @iter_callback: callback function which will be called when finished each iteration
    """

    if x is None:
        x = utils.zero_img(A)

    img = utils.zero_img(A)
    proj = utils.zero_proj(A)
    A.forward(x, proj)

    for i in xrange(niter):
        proj -= b
        G(proj)
        A.backward(proj, img)
        x -= alpha * img
        phi_x(x)
        A.forward(x, proj)
        iter_callback(i, x, proj)

    return x


if __name__ == "__main__":
    main(iterative_fbp)
