#!/usr/bin/env python2

import numpy
import utils
import projector
import sys
from numpy import ceil

def sirt(A, data, niter=500, recon=None, iter_callback=lambda *x : None):
    if recon == None:
        recon = utils.empty_img(A)
        recon[:, :] = 0
    img = utils.empty_img(A)
    proj = utils.empty_proj(A)
    alpha = 0.000008

    for i in xrange(niter):
        A.forward(recon, proj)
        proj -= data
        A.backward(proj, img)
        recon -= alpha * img
        iter_callback(i, recon, proj)
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
    scale = 0.75
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

    (_, name, _) = utils.decompose_path(path)
    callback = utils.IterViewer(img, roi, clim=(-110, 190))
    #callback = utils.IterLogger(img, roi, subname=name)

    utils.show_image(img*roi, clim=(-110, 190))
    sirt(A, proj, iter_callback=callback)

if __name__ == "__main__":
    main()
