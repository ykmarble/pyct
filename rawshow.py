#!/usr/bin/env python3

from pyct import utils
from pylab import *
import sys
import os
import numpy
import cv2

def path_picker(path):
    if not os.path.isdir(path):
        img = utils.load_rawimage(path)
        return lambda _ : (img, path)
    paths = [os.path.join(path, p) for p in os.listdir(path)]
    l = len(paths)
    env = {"index": 0}

    def closure(inc=1):
        img = None
        while img is None:
            env["index"] += inc
            env["index"] %= l
            p = paths[env["index"]]
            img = utils.load_rawimage(p)
        return img, p

    return closure

def main():
    if len(sys.argv) != 2 and len(sys.argv) != 4:
        print("Usage: {} <rawfile> [c-low, c-high]".format(sys.argv[0]))
        return

    path = sys.argv[1]

    if len(sys.argv) == 4:
        clim = [float(i) for i in  sys.argv[2:4]]
    else:
        clim = None

    img_gen = path_picker(path)
    inc = 0

    while True:
        img, path = img_gen(inc)
        print(path)
        print(numpy.min(img), numpy.max(img))
        cv2.destroyAllWindows()
        key = utils.show_image(img, clim, caption=path)
        if key == "f":
            inc = 1
        elif key == "b":
            inc = -1
        else:
            inc = 0

if __name__ == '__main__':
    main()
