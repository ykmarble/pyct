#!/usr/bin/env python2

from worker import main
import utils


def sirt(A, data, alpha, niter=500, x=None, iter_callback=lambda *x : None):
    if x is None:
        x = utils.empty_img(A)
        x[:, :] = 0.5
    img = utils.empty_img(A)
    proj = utils.empty_proj(A)

    for i in xrange(niter):
        A.forward(x, proj)
        proj -= data
        A.backward(proj, img)
        x -= alpha * img
        iter_callback(i, x, proj)
    return


if __name__ == "__main__":
    main(sirt)
