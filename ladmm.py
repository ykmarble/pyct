#!/usr/bin/env python3

from pyct import utils
from pyct.worker import main


def ladmm(A, alpha, niter, phi_x=lambda x: x, phi_y=lambda y: y, G=lambda y: y,
          x=None, y=None, mu=None, iter_callback=lambda *arg: 0):
    """
    Perform interior CT image reconstruction with Linearized ADMM method.
    @A : system matrix class
    @alpha : step size parameter
    @beta : step sizeparameter
    @niter : iteration times
    @phi_x: projection function for image domain
    @phi_y: projection function for projection domain
    @G: preconditionor like high-pass filter function
    @x: initial image
    @y: initial projecction
    @mu: initial dual variable
    @iter_callback : callback function called each iterations
    """
    beta = 1

    if x is None:
        x = utils.zero_img(A)
    if mu is None:
        mu = utils.zero_proj(A)
    if y is None:
        y = utils.zero_proj(A)

    img = utils.zero_img(A)
    proj_k1 = utils.zero_proj(A)
    proj_k = utils.zero_proj(A)

    for i in range(niter):
        proj_k -= y
        proj_k += mu
        G(proj_k)
        A.backward(proj_k, img)
        x -= alpha / beta * img
        phi_x(x)

        A.forward(x, proj_k1)
        y = proj_k1 + mu
        phi_y(y)

        mu += proj_k1 - y
        proj_k[:] = proj_k1[:]

        iter_callback(i, x, y)

    return x


if __name__ == '__main__':
    main(ladmm)
