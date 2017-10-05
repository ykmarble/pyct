#!/usr/bin/env python2

import numpy
import utils
import projector
import sys

def sirt(A, data, niter=1000, recon=None):
    if recon == None:
        recon = utils.empty_img(A)
        recon[:, :] = 0
    img = utils.empty_img(A)
    proj = utils.empty_proj(A)
    alpha = 0.00003

    for i in xrange(niter):
        A.forward(recon, proj)
        proj -= data
        print i, numpy.min(proj), numpy.max(proj)
        utils.show_image(proj)
        utils.show_image(recon, clim=(-128, 256))
        A.backward(proj, img)
        recon -= alpha * img
    return

def main():
    if len(sys.argv) != 2:
        print "Usage: {} <rawfile>".format(sys.argv[0])
        return
    path = sys.argv[1]
    img = utils.load_rawimage(path)
    if img is None:
        print "Invalid file."
        return
    NoI = img.shape[0]
    NoD = NoA = NoI
    proj = numpy.empty((NoA, NoD))
    A = projector.Projector(NoI, NoA, NoD)
    A.forward(img, proj)
    utils.show_image(img)
    utils.show_image(proj)
    sirt(A, proj)

if __name__ == "__main__":
    main()
