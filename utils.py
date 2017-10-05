#!/usr/bin/env python2

import sys
import os
import struct
import numpy
import pylab
import cv2
import sys
from math import sqrt, atan2, pi

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

def normalize(img):
    img -= numpy.min(img)
    img /= numpy.max(img) - numpy.min(img)


def show_image(img, clim=None, output_path="saved_img.dat"):
    """
    @img: 2d or 3d numpy array
    @clim: tuple contains pair of color bound, or None which means deriving them from array
    @output_path: location image will be stored
    """
    if clim is None:
        clim = (numpy.min(img), numpy.max(img))
    normalized = img.copy()
    normalize(normalized)
    cv2.imshow("ctpy", normalized)
    key = cv2.waitKey(0)
    if key == ord("q"):
        sys.exit()
    elif key == ord("s"):
        save_rawimage(img, "saved_img.dat")
        print "saved image at {}".format(os.path.join(os.getcwd(), "saved_img.dat"))
    return key

def reshape_to_polar(img, polar_img):
    Nth, Nr = img.shape
    polar_img = numpy.zeros((Nr, Nr))
    pc = Nr / 2.
    rc = Nr / 2.
    for pyi in xrange(Nr):
        for pxi in xrange(Nr):
            py = pc - pyi
            px = pxi - pc
            r = sqrt(py**2 + px**2)
            th = atan2(py, px)
            if th < 0:
                r *= -1
                th = pi + th
            ri = int(round(r + rc))
            thi = int(round(th / pi * Nth))

            if ri < Nr and ri >= 0 and thi < Nth and thi >= 0:
                polar_img[pyi, pxi] += img[thi, ri]
    show_image(polar_img)
    return


def crop_elipse(img, center, r, value=0):

    rx, ry = r
    cx, cy = center
    rx_2 = rx**2
    ry_2 = ry**2

    for yi in xrange(img.shape[1]):
        for xi in xrange(img.shape[0]):
            if (yi - cy)**2 / ry_2 + (xi - cx)**2 / rx_2 > 1:
                img[yi, xi] = value

def create_elipse_mask(center, a, b, out, value=1):
    """
    Create a array which contains a elipse.
    Inside the elipse are filled by 1.
    The others are 0.
    @shape : the shape of return array
    @center : the center point of the elipse
    @a : the radius of x-axis
    @b : the radius of y-axis
    """
    x, y = center
    a_2 = a**2.
    b_2 = b**2.
    ab_2 = a_2 * b_2
    out[:, :] = 0
    for xi in xrange(out.shape[1]):
        for yi in xrange(out.shape[0]):
            if a_2 * (y - yi)**2 + b_2 * (x - xi)**2 < ab_2:
                out[yi, xi] = value

def empty_img(A):
    return numpy.empty(A.get_image_shape())

def empty_proj(A):
    return numpy.empty(A.get_projector_shape())

def zero_img(A):
    return numpy.zeros(A.get_image_shape())

def zero_proj(A):
    return numpy.zeros(A.get_projector_shape())

def draw_graph(data, canvas):
    d_mini, d_maxi = numpy.min(data), numpy.max(data)
    c_height, c_width = canvas.shape
    assert len(data) == c_width
    canvas[:, :] = 1
    for i in xrange(c_width):
        h = round((data[i] - d_mini) / float(d_maxi - d_mini) * (c_height - 1))  # 0 <= h <= c_height - 1
        h = int(c_height - h - 1)
        canvas[h, i] = 0
    return d_mini, d_maxi
