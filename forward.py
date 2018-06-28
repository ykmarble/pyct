#!/usr/bin/env python2

import numpy
import utils
import cProjector
import ctfilter
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
    A = cProjector.Projector(256, 256, 256)
    A.update_detectors_length(256 * 1)
    proj = utils.zero_proj(A)
    A.forward(img, proj)
    ctfilter.shepp_logan_filter(proj)
    utils.show_image(proj)
    rproj = utils.zero_img(A)
    A.backward(proj, rproj)
    utils.show_image(rproj)
    #utils.save_rawimage(proj, "proj.dat")
    import time
    t1 = time.time()
    A.forward(img, proj)
    t2 = time.time()
    A.backward(proj, rproj)
    t3 = time.time()
    print "f", t2 - t1
    print "b", t3 - t2
    print A.sysmat.dtype

if __name__ == "__main__":
    main()
