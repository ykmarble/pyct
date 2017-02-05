#!/usr/bin/env python2

import numpy
import utils
import projector
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: {} <rawfile>".format(sys.argv[0]))
        return
    path = sys.argv[1]
    img = utils.load_rawimage(path)
    NoI = img.shape[0]
    NoD = NoA = NoI
    proj = numpy.empty((NoA, NoD))
    A = projector.Projector(NoI, NoD, NoA)
    A.forward(img, proj)
    utils.show_image(proj)
    utils.save_rawimage(proj, "proj.dat")

if __name__ == "__main__":
    main()
