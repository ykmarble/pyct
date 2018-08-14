#!/usr/bin/env python3

from PIL import Image
import numpy
from pyct import utils
import sys

def main():
    if (len(sys.argv) < 2):
        print("Usage: {} image-file...".format(sys.argv[0]))
        return
    paths = sys.argv[1:]
    for p in paths:
        img = numpy.array(Image.open(p).convert("F"))
        h, w = img.shape
        basename = ".".join(p.split(".")[:-1])
        if basename == "":
            basename = p
        utils.save_rawimage(img, "{}_{}x{}_f.dat".format(basename, w, h))


if __name__ == '__main__':
    main()
