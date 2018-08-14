#!/usr/bin/env python3

from pyct import utils
import numpy
import skimage


def main():
    import sys
    import os
    if len(sys.argv) != 3:
        print("usage: {} <img-path> <scale>")
        sys.exit(1)
    path = sys.argv[1]
    scale = float(sys.argv[2])
    img = utils.load_rawimage(path)
    newsize = (int(round(img.shape[0]*scale)), int(round(img.shape[1]*scale)))
    scaled = skimage.transform.resize(img, newsize)
    print(numpy.min(scaled), numpy.max(scaled))
    utils.show_image(scaled)
    dirname, basename, ext = utils.decompose_path(path)
    outpath = os.path.join(dirname, "{}_scale{}.{}".format(basename, scale, ext))
    print("save scaled image to {}".format(outpath))
    utils.save_rawimage(scaled, outpath)


if __name__ == "__main__":
    main()
