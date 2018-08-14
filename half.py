#!/usr/bin/env python3

from pyct import utils
import numpy
import os
import sys


def main():
    if len(sys.argv) == 1:
        print("Usage: {} <file> ...".format(sys.argv[0]))
        sys.exit(1)

    for path in sys.argv[1:]:
        if not os.path.isfile(path):
            print("{} is not file".format(path))
            continue

        img = utils.load_rawimage(path)

        if img is None or img.shape[0] != img.shape[1]:
            print("{} is invalid file.".format(path))
            continue

        NoI = img.shape[0]
        halfimg = numpy.zeros((NoI//2, NoI//2))
        for i in range(0, NoI, 2):
            for j in range(0, NoI, 2):
                halfimg[i//2, j//2] = numpy.sum(img[i:i+2, j:j+2]) / img[i:i+2, j:j+2].size

        dname, fname, ext = utils.decompose_path(path)
        outpath = os.path.join(dname, "half_{}.{}".format(fname, ext))
        utils.save_rawimage(halfimg, outpath)
        print("saved {}".format(outpath))


if __name__ == '__main__':
    main()
