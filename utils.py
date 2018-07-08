#!/usr/bin/env python2

import cProjector
import sys
import os
import struct
import numpy
import cv2
import time
from math import sqrt, atan2, pi

DTYPE_t = numpy.float

def decompose_path(path):
    """
    Decompose path into 3 elements, dirname, filename witout extention, extention
    """
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
        img = numpy.array(struct.unpack("{}f".format(width*height), f.read()), dtype=DTYPE_t)
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
    minv = numpy.min(img)
    maxv = numpy.max(img)
    img -= minv
    img /= maxv - minv


def show_image(img, clim=None, output_path="saved_img.dat", caption="ctpy"):
    """
    @img: 2d or 3d numpy array
    @clim: tuple contains pair of color bound, or None which means deriving them from array
    @output_path: location image will be stored
    """
    if clim is None:
        clim = (numpy.min(img), numpy.max(img))
    normalized = img.copy()
    normalized[normalized < clim[0]] = clim[0]
    normalized[normalized > clim[1]] = clim[1]
    normalize(normalized)
    cv2.imshow(caption, normalized)
    key = cv2.waitKey(0)
    if key == ord("q"):
        sys.exit()
    elif key == ord("s"):
        save_rawimage(img, "saved_img.dat")
        print "saved image at {}".format(os.path.join(os.getcwd(), "saved_img.dat"))
    return chr(key)

def convert_to_polar(img, polar_img):
    Nth, Nr = img.shape
    polar_img = numpy.zeros((Nr, Nr), dtype=DTYPE_t)
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

def crop_elipse(img, center, a, b, value=0):
    """
    update image inner the elipsoid by replacing new value
    @img: input image which will be updated
    @center: pair of the index number which points center of elipsoid
    @r: tuple (a, b) that (x/a)**2 + (y/b)**2 == 1
    """
    x, y = center
    a_2 = a**2.
    b_2 = b**2.
    ab_2 = a_2 * b_2
    for xi in xrange(img.shape[1]):
        for yi in xrange(img.shape[0]):
            if a_2 * (y - yi)**2 + b_2 * (x - xi)**2 < ab_2:
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
    out[:, :] = 0
    crop_elipse(out, center, a, b, value)

def empty_img(A):
    return numpy.empty(A.get_image_shape(), dtype=DTYPE_t)

def empty_proj(A):
    return numpy.empty(A.get_projector_shape(), dtype=DTYPE_t)

def zero_img(A):
    return numpy.zeros(A.get_image_shape(), dtype=DTYPE_t)

def zero_proj(A):
    return numpy.zeros(A.get_projector_shape(), dtype=DTYPE_t)

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

def create_sinogram(img, NoA, NoD, scale=1, sample_scale=1, projector=cProjector.Projector):
    """
    Generate sinogram from `img`.
    @img: cross sectional image
    @NoA: the number of angles
    @NoD: the number of detectors
    @scale: the scale of detectors array
    @sample_scale: detector sampling scale
    """
    assert type(sample_scale) == int

    NoI = img.shape[0]
    overNoA = NoA
    overNoD = NoD * sample_scale
    A = projector(NoI, overNoA, overNoD)
    A.update_detectors_length(NoI*scale)
    over_proj = zero_proj(A)
    A.forward(img, over_proj)
    proj = numpy.zeros((NoA, NoD), dtype=DTYPE_t)
    for i in xrange(NoD):
        proj[:, i] = numpy.sum(over_proj[:, i*sample_scale:(i+1)*sample_scale], axis=1)
    proj /= sample_scale
    return proj

def interpolate(n1, n2, l):
    """
    Calculate interploated sample points for [n1, n2].
    @n1: first value of interpolation
    @n2: last value of interpolation
    @l: length of sample points
    """
    x = numpy.zeros(l, dtype=DTYPE_t)
    x += n1 * numpy.cos(numpy.linspace(0, numpy.pi/2., l))**2
    x += n2 * numpy.sin(numpy.linspace(0, numpy.pi/2., l))**2
    return x

def inpaint_metal(proj, support=0):
    """
    Inpainting infinity of `proj` by interpolation.
    """
    NoA, NoD = proj.shape
    for i in xrange(NoA):
        inf_len = 0
        bound_number = [0, 0]  # previous ct-number, next ct-number (both are not inf)
        for j in xrange(NoD):
            if proj[i, j] == float("inf"):
                if inf_len == 0:
                    if j > 0:  # left bound of inf
                        bound_number[0] = proj[i, j-1]
                    else:
                        bound_number[0] = support
                inf_len += 1
            elif inf_len > 0:  # right bound of inf
                bound_number[1] = proj[i, j]
                #proj[i, j-inf_len:j] = numpy.linspace(bound_number[0], bound_number[1], inf_len)
                proj[i, j-inf_len:j] = interpolate(bound_number[0], bound_number[1], inf_len)
                inf_len = 0
        if inf_len > 0:  # inf on right proj bound
            bound_number[1] = support
            #proj[i, -inf_len:] = numpy.linspace(bound_number[0], bound_number[1], inf_len)
            proj[i, NoD-inf_len:NoD] = interpolate(bound_number[0], bound_number[1], inf_len)

def normalizedHU(hu):
    hu_lim = [-1050., 1500.]
    hu -= hu_lim[0]
    hu /= hu_lim[1] - hu_lim[0]
    return hu


class IterLogger(object):
    def __init__(self, xtr, ytr, xmask, A, subname="", no_forward=False):
        self.xtr = xtr
        self.ytr = ytr
        self.xmask = xmask
        self.A = A

        self.xn = numpy.sum(xmask, axis=None)
        self.yn = ytr.shape[0] * ytr.shape[0]
        self.proj = zero_proj(A)
        self.y_max = numpy.max(numpy.abs(self.ytr))
        self.initialized = False
        self.no_forward = no_forward

        # generate the output directory path and create its directory
        timestamp = str(int(time.time()))
        (_, method, _) = decompose_path(sys.argv[0])

        self.dirpath = os.path.join(os.getcwd(), "iterout", method, timestamp+subname)

    def initialize(self):
        print "output dir: {}".format(self.dirpath)
        self._mkdir(self.dirpath)

        # log file
        logname = "iterlog.txt"
        self.log_handler = open(os.path.join(self.dirpath, logname), "w")
        self.initialized = True

    def __call__(self, i, x, *argv, **argdict):
        if not self.initialized:
            self.initialize()

        if not self.no_forward:
            self.A.forward(x, self.proj)
        xrmse = numpy.sqrt(numpy.sum(((x - self.xtr)*self.xmask)**2) / self.xn)
        yrmse = numpy.sqrt(numpy.sum((self.proj - self.ytr)**2) / self.yn)

        print i+1, xrmse, yrmse
        self.log_handler.write("{} {} {} {}\n".format(i+1, xrmse, yrmse, time.time()))
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


class IterViewer(object):
    def __init__(self, xtr, ytr, xmask, A, clim=None, niter=10):
        self.xtr = xtr
        self.ytr = ytr
        self.xmask = xmask
        self.A = A
        self.clim = clim
        self.niter = niter

        self.xn = numpy.sum(xmask, axis=None)
        self.yn = ytr.shape[0] * ytr.shape[0]
        self.xmax = numpy.max(xtr) - numpy.min(xtr)
        self.ymax = numpy.max(ytr) - numpy.min(ytr)
        self.proj = zero_proj(A)
        self.proj_lim = (numpy.max(self.ytr), numpy.min(self.ytr))

    def __call__(self, i, x, *argv, **argdict):
        self.A.forward(x, self.proj)
        xrmse = numpy.sqrt(numpy.sum(((x - self.xtr)*self.xmask)**2) / self.xn)
        yrmse = numpy.sqrt(numpy.sum((self.proj - self.ytr)**2) / self.yn)
        print i+1, xrmse, yrmse
        if (i+1) % self.niter == 0:
            show_image(x*self.xmask, clim=self.clim)
