#!/usr/bin/env python2

import numpy
import utils
import projector
import sys
from numpy import ceil

def sirt(A, data, niter=1000, recon=None):
    if recon == None:
        recon = utils.empty_img(A)
        recon[:, :] = 0
    img = utils.empty_img(A)
    proj = utils.empty_proj(A)
    alpha = 0.000005

    for i in xrange(niter):
        A.forward(recon, proj)
        proj -= data
        print i, numpy.min(proj), numpy.max(proj)
        A.backward(proj, img)
        recon -= alpha * img
        if i == 50:
            utils.save_rawimage(recon, "sirt.dat")
            sys.exit()
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
    scale = 0.6
    NoI = img.shape[0]
    NoD = NoA = NoI
    proj = numpy.empty((NoA, NoD))
    A = projector.Projector(NoI, NoA, NoD)
    A.update_detectors_length(NoI * scale)
    A.forward(img, proj)
    sirt(A, proj)

if __name__ == "__main__":
    main()
