#!/usr/bin/env python2

import numpy
import utils
import cProjector
import differencial
import ctfilter
import dbp
import sys

def main():
    if len(sys.argv) != 2:
        print "Usage: {} <rawfile>".format(sys.argv[0])
        return
    path = sys.argv[1]

    #proj = utils.load_rawimage(path)
    #NoA, NoD = proj.shape
    #NoA, NoD = (256, 256)
    #NoI = NoA
    #A = projector.Projector(NoI, NoA, NoD)
    #img = numpy.empty((NoI, NoI))
    #import fbp
    #proj = proj[:, 23:256+23]
    #fbp.shepp_logan_filter(proj)
    #print proj.shape
    #utils.show_image(proj)
    #A.backward(proj, img)
    #print numpy.min(img), numpy.max(img)
    #print proj.shape
    #utils.show_image(img, clim=(-20, 20))
    #return

    img = utils.load_rawimage(path)
    if img is None:
        print "Invalid file."
        return
    NoI = img.shape[0]
    NoD = NoA = NoI
    A = cProjector.Projector(NoI, NoA, NoD)
    A.update_detectors_length(NoI * 1)
    proj = utils.zero_proj(A)
    A.forward(img, proj)
    utils.show_image(proj)
    ctfilter.shepp_logan_filter(proj)
    #proj[:] = dbp.inv_freq_hilbert_filter(proj)
    #ctfilter.inv_ramp_filter(proj)
    utils.show_image(proj)
    ctfilter.shepp_logan_filter(proj)
    rproj = utils.zero_img(A)
    A.backward(proj, rproj)
    utils.show_image(rproj)
    #utils.save_rawimage(proj, "proj.dat")

if __name__ == "__main__":
    main()
