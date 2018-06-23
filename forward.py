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
    print "gen sysmat"
    A = cProjector.sysmat(NoI, NoI, NoA, NoD)
    print "forwardp"
    proj = A * img.reshape(-1)
    print "filtering"
    ctfilter.shepp_logan_filter(proj.reshape(NoA, NoD))
    utils.show_image(proj.reshape(NoA, NoD))
    print "backwordp"
    rproj = A.transpose() * proj
    utils.show_image(rproj.reshape(NoI, NoI))
    #utils.save_rawimage(proj, "proj.dat")

if __name__ == "__main__":
    main()
