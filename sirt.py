#!/usr/bin/env python2

import numpy
import utils
import projector
import sys
from numpy import ceil

def sirt(A, data, niter=500, recon=None, iter_callback=lambda *x : None):
    if recon == None:
        recon = utils.empty_img(A)
        recon[:, :] = 0.5
    img = utils.empty_img(A)
    proj = utils.empty_proj(A)
    alpha = 10000

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

    scale = 1

    proj, img, A = utils.create_projection(path, interior_scale=scale, detector_scale=1.5, angular_scale=1.5)

    if img is None:
        print "Invalid file."
        return

    roi = utils.zero_img(A)
    roi_c = ((roi.shape[0] - 1) / 2., (roi.shape[1] - 1) / 2.)
    roi_r = [roi.shape[0] * scale / 2., roi.shape[1] * scale / 2.]
    utils.create_elipse_mask(roi_c, roi_r[0], roi_r[1], roi)

    (_, name, _) = utils.decompose_path(path)
    callback = utils.IterViewer(img, roi, clim=(0, 1))
    #callback = utils.IterViewer(img, roi, clim=(utils.normalizedHU(-110), utils.normalizedHU(190)))
    #callback = utils.IterLogger(img, roi, subname=name)

    sirt(A, proj, iter_callback=callback)

if __name__ == "__main__":
    main()
