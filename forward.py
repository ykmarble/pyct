#!/usr/bin/env python3

import numpy
from pyct import utils
from pyct import cProjector
from pyct import ctfilter
from pyct import dbp
import sys
import time


def main():
    if len(sys.argv) != 2:
        print("Usage: {} <rawfile>".format(sys.argv[0]))
        return
    path = sys.argv[1]

    img = utils.load_rawimage(path)
    if img is None:
        print("Invalid file.")
        return
    NoI = img.shape[0]
    NoA = 307
    NoD = 410

    img /= 2.

    A = cProjector.Projector(NoI, NoA, NoD)
    A.update_detectors_length(NoI * 1)

    proj = utils.zero_proj(A)
    A.forward(img, proj)
    t1 = time.time()
    A.forward(img, proj)
    t2 = time.time()
    utils.show_image(proj)
    utils.save_rawimage(proj, "proj.dat")
    print(numpy.min(proj), numpy.max(proj))
    return

    ctfilter.shepp_logan_filter(proj)
    utils.show_image(proj)

    rproj = utils.zero_img(A)
    t3 = time.time()
    A.backward(proj, rproj)
    t4 = time.time()

    print(numpy.min(rproj), numpy.max(rproj))
    utils.show_image(rproj)
    #utils.save_rawimage(proj, "proj.dat")

    print("f", t2 - t1)
    print("b", t4 - t3)

if __name__ == "__main__":
    main()
