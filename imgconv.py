#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import utils
import sys

from pylab import *

def main():
    if (len(sys.argv) < 2):
        print "Usage: {} raw-file...".format(sys.argv[0])
        return
    paths = sys.argv[1:]
    for p in paths:
        img = utils.load_rawimage(p)
        if img is None:
            print "Invalid file: {}".foramt(p)
            continue
        basename = p.rsplit(".", 1)[0]
        imshow(img, "gray")
        xticks([])
        yticks([])
        #colorbar()
        savefig("{}.png".format(basename))


if __name__ == '__main__':
    main()
