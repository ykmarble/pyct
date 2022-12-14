#!/usr/bin/env python3

import ctfilter
import skimage.filters
import numpy
cimport numpy
DTYPE = numpy.float
ctypedef numpy.float_t DTYPE_t
cimport cython
from cython.parallel import *
from libc.math cimport M_PI, sin, cos, floor, ceil, round, fabs

class Projector(object):
    """
    instance variables:
    x_offset
    y_offset
    center_x
    center_y
    detectors_offset
    angular_range
    detectors_length

    image and projection data geometory:
    img[y, x]
    proj[theta, r]

    method:
    update_`instance variable`
    forward
    backward
    """
    def __init__(self, length_of_image_side, num_of_angles, num_of_detectors):
        self.NoI = length_of_image_side  # number of pixels of image side
        self.NoD = num_of_detectors      # number of detectors
        self.NoA = num_of_angles         # number of projection angles

        self.image_origin = (self.NoI - 1) / 2.      # num_of_image_sides / 2
        self.detectors_origin = (self.NoD - 1) / 2.  # num_of_detectors / 2

        # physical configuration
        # "*_offset" means the offset from CENTER of the image or detectors
        # the size of each pixel is 1x1
        self.update_x_offset(0)
        self.update_y_offset(0)
        self.update_detectors_offset(0)
        self.update_angular_range(M_PI)         # domain of theta [0, angular_range)
        self.update_detectors_length(self.NoI)  # length of the string of detectors

        # variables used when computing projection
        # calculated from above variables automatically
        # dr                : in other words, the length of each detector
        # dtheta
        # center_x          : derived from image_origin and x_offset
        # center_y          : derived from image_origin and y_offset
        # detectors_center  : derived from detectors_origin and detectors_offset

        # cache
        self.all_th_indexes = numpy.arange(self.NoA)
        self.all_r_indexes = numpy.arange(self.NoD)

    def get_image_shape(self):
        return (self.NoI, self.NoI)

    def get_projector_shape(self):
        return (self.NoA, self.NoD)

    def convidx_img2r(self, yi, xi, ti):
        th = ti * self.dtheta
        x = xi - self.center_x
        y = self.center_y - yi
        r = (x * sin(th) + y * cos(th)) / self.dr
        r += self.detectors_center
        return int(round(r))

    def update_x_offset(self, offset):
        self.x_offset = offset
        self.center_x = self.x_offset + self.image_origin

    def update_y_offset(self, offset):
        self.y_offset = offset
        self.center_y = -self.y_offset + self.image_origin

    def update_detectors_offset(self, offset):
        self.detectors_offset = offset
        self.detectors_center = self.detectors_offset + self.detectors_origin

    def update_angular_range(self, max_angle):
        self.angular_range = max_angle
        self.dtheta = self.angular_range / self.NoA

    def update_detectors_length(self, length):
        self.detectors_length = float(length)
        self.dr = self.detectors_length / (self.NoD - 1)
        self.backward_signal_scale = self.detectors_length / self.NoD

    def update_center_x(self, x):
        self.center_x = x
        self.x_offset = self.center_x - self.image_origin

    def update_center_y(self, y):
        self.center_y = y
        self.y_offset = -self.center_y + self.image_origin

    def is_valid_dimension(self, numpy.ndarray[DTYPE_t, ndim=2] img, numpy.ndarray[DTYPE_t, ndim=2] proj):
        return img.shape[0] == img.shape[1] \
          and img.shape[0] == self.NoI \
          and proj.shape[0] == self.NoA \
          and proj.shape[1] == self.NoD

    def forward(self, numpy.ndarray[DTYPE_t, ndim=2] img, numpy.ndarray[DTYPE_t, ndim=2] proj):
        assert self.is_valid_dimension(img, proj)
        self._projection(img, proj, False)
        #ctfilter.fir_gauss_1d(proj)

    def backward(self, numpy.ndarray[DTYPE_t, ndim=2] proj, numpy.ndarray[DTYPE_t, ndim=2] img):
        assert self.is_valid_dimension(img, proj)
        self._projection(proj, img, True)
        img *= self.backward_signal_scale
        img /= 2 * self.NoA

        #img[:, :] = skimage.filters.gaussian(img, 0.8)

    def partial_forward(self, numpy.ndarray[DTYPE_t, ndim=2] img, numpy.ndarray[DTYPE_t, ndim=2] proj,
                        numpy.ndarray[numpy.int_t, ndim=1] th_indexes):
        assert self.is_valid_dimension(img, proj)
        if th_indexes is not None:
            assert 0 <= numpy.min(th_indexes) and numpy.max(th_indexes) < self.NoA
        self._projection(img, proj, False, th_indexes=th_indexes, r_indexes=None)
        #ctfilter.fir_gauss_1d(proj)

    def partial_backward(self, numpy.ndarray[DTYPE_t, ndim=2] proj, numpy.ndarray[DTYPE_t, ndim=2] img,
                         numpy.ndarray[numpy.int_t, ndim=1] th_indexes):
        assert self.is_valid_dimension(img, proj)
        if th_indexes is not None:
            assert 0 <= numpy.min(th_indexes) and numpy.max(th_indexes) < self.NoA
        self._projection(proj, img, True, th_indexes=th_indexes, r_indexes=None)
        #img[:, :] = skimage.filters.gaussian(img, 0.8)

    @cython.boundscheck(False)
    def _projection(self, numpy.ndarray[DTYPE_t, ndim=2] src, numpy.ndarray[DTYPE_t, ndim=2] dst, int backward,
                    numpy.ndarray[numpy.int_t, ndim=2] mask=None,
                    numpy.ndarray[numpy.int_t, ndim=1] th_indexes=None,
                    numpy.ndarray[numpy.int_t, ndim=1] r_indexes=None):
        cdef int ti, ri, xi, yi, ti_i, ri_i
        cdef DTYPE_t th, sin_th, cos_th, abs_sin, abs_cos, sin_cos, cos_sin,
        cdef DTYPE_t inv_cos_th, inv_abs_cos, inv_sin_th, inv_abs_sin
        cdef DTYPE_t r, ray_offset, xs, ys, rayx, rayy, aij, aijp
        cdef int NoI, NoA, NoD
        cdef DTYPE_t center_x, center_y, detectors_center, dr, dtheta

        if mask is None:
            mask = numpy.zeros_like(src, numpy.int)
        if th_indexes is None:
            th_indexes = self.all_th_indexes
        if r_indexes is None:
            r_indexes = self.all_r_indexes

        dst[:, :] = 0
        NoI = self.NoI
        NoA = th_indexes.shape[0]
        NoD = r_indexes.shape[0]
        center_x = self.center_x
        center_y = self.center_y
        detectors_center = self.detectors_center
        dr = self.dr
        dtheta = self.dtheta

        for ti_i in prange(NoA, nogil=True):
        #for ti_i in range(NoA):
            ti = th_indexes[ti_i]
            th = ti * dtheta
            sin_th = sin(th)
            cos_th = cos(th)
            abs_sin = fabs(sin_th)
            abs_cos = fabs(cos_th)

            if (abs_sin < abs_cos):
                sin_cos = sin_th / cos_th
                inv_cos_th = 1 / cos_th
                inv_abs_cos = fabs(inv_cos_th)

                for ri_i in range(NoD):
                    ri = r_indexes[ri_i]
                    r = (ri - detectors_center) * dr
                    ray_offset = r * inv_cos_th

                    for xi in range(NoI):
                        xs = xi - center_x
                        rayy = -(sin_cos * xs + ray_offset) + center_y
                        yi = int(floor(rayy))
                        aijp = rayy - yi
                        aij = 1 - aijp

                        if (backward):
                            if (is_valid_index(xi, yi, center_x, center_y, NoI)):
                                dst[yi, xi] += aij * src[ti, ri] * inv_abs_cos
                            if (is_valid_index(xi, yi+1, center_x, center_y, NoI)):
                                dst[yi+1, xi] += aijp * src[ti, ri] * inv_abs_cos
                        else:
                            if (is_valid_index(xi, yi, center_x, center_y, NoI)):
                                dst[ti, ri] += aij * src[yi, xi] * inv_abs_cos
                            if (is_valid_index(xi, yi+1, center_x, center_y, NoI)):
                                dst[ti, ri] += aijp * src[yi+1, xi] * inv_abs_cos
            else:
                cos_sin = cos_th / sin_th
                inv_sin_th = 1 / sin_th
                inv_abs_sin = fabs(inv_sin_th)

                for ri_i in range(NoD):
                    ri = r_indexes[ri_i]
                    r = (ri - detectors_center) * dr
                    ray_offset = r * inv_sin_th

                    for yi in range(NoI):
                        ys = center_y - yi
                        rayx = cos_sin * ys - ray_offset + center_x
                        xi = int(floor(rayx))
                        aijp = rayx - xi
                        aij = 1 - aijp

                        if (backward):
                            if (is_valid_index(xi, yi, center_x, center_y, NoI)):
                                dst[yi, xi] +=  aij * src[ti, ri] * inv_abs_sin
                            if (is_valid_index(xi+1, yi, center_x, center_y, NoI)):
                                dst[yi, xi+1] += aijp * src[ti, ri] * inv_abs_sin
                        else:
                            if (is_valid_index(xi, yi, center_x, center_y, NoI)):
                                dst[ti, ri] += aij * src[yi, xi] * inv_abs_sin
                            if (is_valid_index(xi+1, yi, center_x, center_y, NoI)):
                                dst[ti, ri] += aijp * src[yi, xi+1] * inv_abs_sin

    @cython.boundscheck(False)
    def _projection2(self, numpy.ndarray[DTYPE_t, ndim=2] src, numpy.ndarray[DTYPE_t, ndim=2] dst, int backward=False,
                    numpy.ndarray[numpy.int_t, ndim=1] th_indexes=None,
                    numpy.ndarray[numpy.int_t, ndim=1] r_indexes=None):
        cdef int NoI, NoA, NoD
        cdef DTYPE_t center_x, center_y, detectors_center, dr, dtheta
        cdef int th_i, ti, xi, yi, l, h
        cdef DTYPE_t th, a, b, x, y, dist, dist_on_det, l_ratio, h_ratio, val

        if th_indexes is None:
            th_indexes = self.all_th_indexes
        if r_indexes is None:
            r_indexes = self.all_r_indexes

        dst[:, :] = 0
        NoI = self.NoI
        NoA = th_indexes.shape[0]
        NoD = r_indexes.shape[0]
        center_x = self.center_x
        center_y = self.center_y
        detectors_center = self.detectors_center
        dr = self.dr
        dtheta = self.dtheta

        #for th_i in prange(NoA, nogil=True):
        for th_i in range(NoA):

            ti = th_indexes[th_i]
            th = ti * dtheta

            # ???????????????????????????????????????????????? ax + by = 0
            a = sin(th)
            b = -cos(th)

            for xi in range(NoI):
                for yi in range(NoI):

                    x = xi - center_x
                    y = center_y - yi  # ?????????????????????????????????????????????????????????

                    # dist???????????????????????????????????????X????????????????????????????????????????????????????????????????????????????????????
                    dist = a * x + b * y
                    dist_on_det = dist / dr + detectors_center
                    l = int(floor(dist_on_det))
                    h = l + 1
                    l_ratio = h - dist_on_det
                    h_ratio = 1 - l_ratio

                    if (backward):
                        if (0 <= l and l < NoD):
                            dst[yi, xi] += src[ti, l] * l_ratio
                        if (0 <= h and h < NoD):
                            dst[yi, xi] += src[ti, h] * h_ratio
                    else:
                        if (0 <= l and l < NoD):
                            dst[ti, l] += src[yi, xi] * l_ratio
                        if (0 <= h and h < NoD):
                            dst[ti, h] += src[yi, xi] * h_ratio


cdef inline int is_valid_index(int xi, int yi, double center_x, double center_y, int NoI) nogil:
    cdef double x = xi - center_x
    cdef double y = center_y - yi
    return 0 <= xi and xi < NoI \
      and 0 <= yi and yi < NoI \
      #and 4 * (x * x + y * y) < (NoI - 1) * (NoI -1)
