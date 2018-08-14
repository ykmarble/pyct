#!/usr/bin/env python3

import struct
import numpy
from ctpy import utils
import sys


def main():
    if (len(sys.argv) != 4):
        print("Usage: {} <image-file> <width> <height>".format(sys.argv[0]))
        print("byte order: column -> row")
        return
    _, path, width, height = sys.argv
    width = int(width)
    height = int(height)
    with open(path, "rb") as f:
        img = numpy.array(struct.unpack("<" + "d" * (width * height), f.read())).reshape(height, width)
    basename = ".".join(path.split(".")[:-1])
    if basename == "":
        basename = path
    utils.save_rawimage(img, "{}_{}x{}_f.dat".format(basename, width, height))


if __name__ == '__main__':
    main()
