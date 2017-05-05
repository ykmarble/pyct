#!/usr/bin/env python2

import sys
import struct
import numpy
import pylab
import cv2

def load_rawimage(path):
    with open(path, "rb") as f:
        header = struct.unpack("ccxxII", f.read(12))
        if not (header[0] == b"P" and header[1] == b"0"):
            return None
        width = header[2]
        height = header[3]
        img = numpy.array(struct.unpack("{}f".format(width*height), f.read()))
    img.resize(height, width)
    return img

def save_rawimage(img, outpath):
    h, w = img.shape
    header = struct.pack("ccxxII", b"P", b"0", w, h)
    payload = struct.pack("{}f".format(h * w), *img.ravel())
    with open(outpath, "wb") as f:
        f.write(header)
        f.write(payload)

def show_image(img, clim=None):
    """
    @img: 2d or 3d numpy array
    @clim: tuple contains pair of color bound, or None which means deriving them from array
    """
    if clim is None:
        clim = (numpy.min(img), numpy.max(img))
    normalized = img.copy()
    normalized -= clim[0]
    normalized /= clim[1] - clim[0]
    cv2.imshow("ctpy", normalized)
    cv2.waitKey(0)
