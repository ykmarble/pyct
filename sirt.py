#!/usr/bin/env python2

import numpy
import utils
import projector
import sys
from numpy import ceil

def sirt(A, data, niter=3000, recon=None, roi=None):
    if recon == None:
        recon = utils.empty_img(A)
        recon[:, :] = 0
    img = utils.empty_img(A)
    proj = utils.empty_proj(A)
    alpha = 0.000003

    for i in xrange(niter):
        A.forward(recon, proj)
        proj -= data
        A.backward(proj, img)
        recon -= alpha * img
        print i, recon[128, 128], numpy.sum(numpy.sqrt(((recon - img)*roi)**2))
        if i > 0 and i % 10 == 0:
            utils.save_rawimage(recon, "sirt/{}.dat".format(i))
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
    scale = 0.4
    NoI = img.shape[0]
    NoD = NoA = NoI
    proj = numpy.empty((NoA, NoD))
    A = projector.Projector(NoI, NoA, NoD)
    A.update_detectors_length(NoI * scale)
    A.forward(img, proj)

    roi = utils.zero_img(A)
    roi_c = ((roi.shape[0] - 1) / 2., (roi.shape[1] - 1) / 2.)
    roi_r = [roi.shape[0] * scale / 2., roi.shape[1] * scale / 2.]
    utils.create_elipse_mask(roi_c, roi_r[0], roi_r[1], roi)

    sirt(A, proj, roi=roi)

if __name__ == "__main__":
    main()
