#!/usr/bin/env python2

import utils
from pylab import *
import sys
import numpy

def main():
    if len(sys.argv) != 2 and len(sys.argv) != 4:
        print "Usage: {} rawfile".format(sys.argv[0])
        return
    path = sys.argv[1]
    img = utils.load_rawimage(path)
    if img is None:
        print "Invalid file."
        return
    print numpy.min(img), numpy.max(img)
    if len(sys.argv) == 4:
        clim = [float(i) for i in  sys.argv[2:4]]
        utils.show_image(img, clim)
    else:
        utils.show_image(img)

if __name__ == '__main__':
    main()
