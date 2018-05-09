#!/usr/bin/env python2

import utils
from worker import main


def app(A, alpha, beta, niter, phi_x=lambda x: x, phi_y=lambda y: y, G=lambda y: y,
        x=None, y=None, mu=None, iter_callback=lambda *arg: 0):
    """
    Perform interior CT image reconstruction with APP method.
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
    if x is None:
        x = utils.zero_img(A)
    if y is None:
        y = utils.zero_proj(A)
    if mu is None:
        mu = utils.zero_proj(A)
    img = utils.zero_img(A)
    proj = utils.zero_proj(A)

    for i in xrange(niter):
        A.forward(x, proj)
        proj -= y
        G(proj)
        mu_bar = -mu.copy()
        mu += alpha * proj
        mu_bar += 2 * mu

        A.backward(mu_bar, img)
        x -= beta * img
        phi_x(x)

        y += beta * mu_bar
        phi_y(y)

        iter_callback(i, x, y, mu)

    return x

if __name__ == '__main__':
    main(app)
