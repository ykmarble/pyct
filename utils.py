#!/usr/bin/env python2

import projector
import sys
import os
import struct
import numpy
import cv2
import time
from math import sqrt, atan2, pi, ceil

def decompose_path(path):
    dname = os.path.dirname(path)
    bname = os.path.basename(path).rsplit(".", 1)
    fname = bname[0]
    if len(bname) > 1:
        ext = bname[1]
    else:
        ext = ""
    return dname, fname, ext

def load_rawimage(path):
    with open(path, "rb") as f:
        header = struct.unpack("ccxxII", f.read(12))
        if not (header[0] == b"P" and header[1] == b"0"):
            return None
        width = header[2]
        height = header[3]
        img = numpy.array(struct.unpack("{}f".format(width*height), f.read()))
    img.resize(height, width)
    NoI = img.shape[0]
    r =(NoI-1)/2.
    mask = numpy.zeros_like(img)
    create_elipse_mask((r, r), r, r, mask)
    img[mask!=1] = 0
    return img

def save_rawimage(img, outpath):
    h, w = img.shape
    header = struct.pack("ccxxII", b"P", b"0", w, h)
    payload = struct.pack("{}f".format(h * w), *img.ravel())
    with open(outpath, "wb") as f:
        f.write(header)
        f.write(payload)

def normalize(img, clim):
    img -= clim[0]
    img /= clim[1] - clim[0]


def show_image(img, clim=None, output_path="saved_img.dat", caption="ctpy"):
    """
    @img: 2d or 3d numpy array
    @clim: tuple contains pair of color bound, or None which means deriving them from array
    @output_path: location image will be stored
    """
    if clim is None:
        clim = (numpy.min(img), numpy.max(img))
    normalized = img.copy()
    normalize(normalized, clim)
    cv2.imshow(caption, normalized)
    key = cv2.waitKey(0)
    if key == ord("q"):
        sys.exit()
    elif key == ord("s"):
        save_rawimage(img, "saved_img.dat")
        print "saved image at {}".format(os.path.join(os.getcwd(), "saved_img.dat"))
    return chr(key)

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

def vcoord(shape, point):
    """
    Convert the coordinate systems from 2D-array to the virtual system, taken as arguments.
    @shape : a shape of 2D-array, or length of image sides
    @point : a point in the 2D-array coordinate system
    """
    h, w = shape
    x, y = point
    return (x - w/2., h/2. - y)

def acoord(shape, point):
    """
    Convert the coordinate systems from the virtual system to 2D-array.
    This fuction is inverse transformation of `vcoord`.
    @shape : a shape of 2D-array, or length of image sides
    @point : a point in the virtual coordinate system
    """
    h, w = shape
    x, y = point
    return (x + w/2., h/2. - y)

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

def create_projection(path, interior_scale=1, detector_scale=1, sample_scale=8):
    assert type(sample_scale) == int
    img = load_rawimage(path)
    NoI = img.shape[0]
    NoA = NoI
    NoD = int(ceil(NoI * detector_scale))
    if NoD % 2 !=0:
        NoD += 1
    overNoA = NoA# * sample_scale
    overNoD = NoD * sample_scale
    A = projector.Projector(NoI, overNoA, overNoD)
    A.update_detectors_length(int(ceil(NoI*interior_scale)))
    over_proj = zero_proj(A)
    A.forward(img, over_proj)
    proj = numpy.zeros((NoA, NoD))
    for i in xrange(NoA):
        for j in xrange(NoD):
            for k in xrange(sample_scale):
                proj[i, j] += over_proj[i, j*sample_scale + k]
    proj /= sample_scale
    normalA = projector.Projector(NoI, NoA, NoD)
    normalA.update_detectors_length(NoI * interior_scale)
    return proj, img, normalA

def interpolate(n1, n2, l):
    x = numpy.zeros(l)
    x += n1 * numpy.cos(numpy.linspace(0, numpy.pi/2., l))**2
    x += n2 * numpy.sin(numpy.linspace(0, numpy.pi/2., l))**2
    return x

class IterLogger(object):
    def __init__(self, original_img, roi, subname=""):
        self.img = original_img
        self.roi = roi

        # generate the output directory path and create its directory
        timestamp = str(int(time.time()))
        (_, method, _) = decompose_path(sys.argv[0])

        self.dirpath = os.path.join(os.getcwd(), "iterout", method, timestamp+subname)
        print "output dir: {}".format(self.dirpath)
        self._mkdir(self.dirpath)

        # log file
        logname = "iterlog.txt"
        self.log_handler = open(os.path.join(self.dirpath, logname), "w")

    def __call__(self, i, x, y):
        rmse = self._rmse(x)
        print i+1, rmse
        self.log_handler.write("{} {}\n".format(i+1, rmse))
        self.log_handler.flush()

        if (i+1) % 10 == 0:
            imgname = "{}.dat".format(i+1)
            save_rawimage(x, os.path.join(self.dirpath, imgname))

    def _mkdir(self, path):
        if os.path.isdir(path):
            return
        if os.path.exists(path):
            raise Exception("{} is already used as a file.".format(path))
        os.makedirs(path)

    def _rmse(self, x):
        N = x.shape[0] * x.shape[1]
        return numpy.sqrt(numpy.sum(((x - self.img)*self.roi)**2) / N)


class IterViewer(object):
    def __init__(self, original_img, roi, clim=None):
        self.img = original_img
        self.roi = roi
        self.clim = clim
        self.N = self.img.shape[0] * self.img.shape[0]

    def __call__(self, i, x, y):
        print i, numpy.sqrt(numpy.sum(((x - self.img)*self.roi)**2) / self.N)
        show_image(x, clim=self.clim)
