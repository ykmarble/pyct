#!/usr/bin/env python2

import utils
from pylab import *
import sys

def main():
    if (len(sys.argv) != 2):
        print "Usage: {} rawfile".format(sys.argv[0])
        return
    path = sys.argv[1]
    img = utils.load_rawimage(path)
    if img is None:
        print "Invalid file."
        return
    utils.show_image(img)

if __name__ == '__main__':
    main()
