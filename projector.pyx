#!/usr/bin/env python2

import numpy
cimport numpy
DTYPE = numpy.float
ctypedef numpy.float_t DTYPE_t
cimport cython
from libc.math cimport M_PI, sin, cos, floor

class Projector:

    def __init__(self, length_of_image_side, num_of_detectors, num_of_angles):
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
        self.update_angular_range(M_PI)          # domain of theta [0, angular_range]
        self.update_detectors_length(self.NoI) # length of the string of detectors

        # variables used when computing projection
        # calculated from above variables automatically
        # dr                : in other words, the length of each detector
        # dtheta
        # center_x          : derived from image_origin and x_offset
        # center_y          : derived from image_origin and y_offset
        # detectors_center  : derived from detectors_origin and detectors_offset

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

    def update_center_x(self, x):
        self.center_x = x
        self.x_offset = self.center_x - self.image_origin

    def update_center_y(self, y):
        self.center_y = y
        self.y_offset = -self.center_y + self.image_origin

    def is_valid_dimension(self, img, proj):
        return img.shape[0] == img.shape[1] \
          and img.shape[0] == self.NoI \
          and proj.shape[0] == self.NoA \
          and proj.shape[1] == self.NoD

    def forward(self, img, proj):
        assert self.is_valid_dimension(img, proj)
        self.projection(img, proj, False)

    def backward(self, proj, img):
        assert self.is_valid_dimension(img, proj)
        self.projection(proj, img, True)

    @cython.boundscheck(False)
    def projection(self, numpy.ndarray[DTYPE_t, ndim=2] src, numpy.ndarray[DTYPE_t, ndim=2] dst, int backward=False):
        cdef int ti, ri, xi, yi
        cdef DTYPE_t th, sin_th, cos_th, abs_sin, abs_cos, sin_cos, cos_sin,
        cdef DTYPE_t inv_cos_th, inv_abs_cos, inv_sin_th, inv_abs_sin
        cdef DTYPE_t r, ray_offset, xs, ys, rayx, rayy, aij, aijp
        cdef int NoI, NoA, NoD
        cdef DTYPE_t center_x, center_y, detectors_center, dr, dtheta

        dst[:, :] = 0
        NoI = self.NoI
        NoA = self.NoA
        NoD = self.NoD
        center_x = self.center_x
        center_y = self.center_y
        detectors_center = self.detectors_center
        dr = self.dr
        dtheta = self.dtheta

        for ti in xrange(0, NoA):
            th = ti * dtheta
            sin_th = sin(th)
            cos_th = cos(th)
            abs_sin = abs(sin_th)
            abs_cos = abs(cos_th)

            if (abs_sin < abs_cos):
                sin_cos = sin_th / cos_th
                inv_cos_th = 1 / cos_th
                inv_abs_cos = abs(inv_cos_th)

                for ri in xrange(NoD):
                    r = (ri - detectors_center) * dr
                    ray_offset = r * inv_cos_th

                    for xi in xrange(NoI):
                        xs = xi - center_x
                        rayy = -(sin_cos * xs + ray_offset) + center_y
                        yi = int(floor(rayy))
                        aijp = rayy - yi
                        aij = 1 - aijp

                        if (backward):
                            if (is_valid_index(xi, yi, center_x, center_y, NoI)):
                                dst[yi, xi] = dst[yi, xi] + aij * src[ti, ri] * inv_abs_cos
                            if (is_valid_index(xi, yi+1, center_x, center_y, NoI)):
                                dst[yi+1, xi] = dst[yi+1, xi] + aijp * src[ti, ri] * inv_abs_cos
                        else:
                            if (is_valid_index(xi, yi, center_x, center_y, NoI)):
                                dst[ti, ri] = dst[ti, ri] + aij * src[yi, xi] * inv_abs_cos
                            if (is_valid_index(xi, yi+1, center_x, center_y, NoI)):
                                dst[ti, ri] = dst[ti, ri] + aijp * src[yi+1, xi] * inv_abs_cos
            else:
                cos_sin = cos_th / sin_th
                inv_sin_th = 1 / sin_th
                inv_abs_sin = abs(inv_sin_th)

                for ri in xrange(NoD):
                    r = (ri - detectors_center) * dr
                    ray_offset = r * inv_sin_th

                    for yi in xrange(NoI):
                        ys = center_y - yi
                        rayx = cos_sin * ys - ray_offset + center_x
                        xi = int(floor(rayx))
                        aijp = rayx - xi
                        aij = 1 - aijp

                        if (backward):
                            if (is_valid_index(xi, yi, center_x, center_y, NoI)):
                                dst[yi, xi] = dst[yi, xi] + aij * src[ti, ri] * inv_abs_sin
                            if (is_valid_index(xi+1, yi, center_x, center_y, NoI)):
                                dst[yi, xi+1] = dst[yi, xi+1] + aijp * src[ti, ri] * inv_abs_sin
                        else:
                            if (is_valid_index(xi, yi, center_x, center_y, NoI)):
                                dst[ti, ri] = dst[ti, ri] + aij * src[yi, xi] * inv_abs_sin
                            if (is_valid_index(xi+1, yi, center_x, center_y, NoI)):
                                dst[ti, ri] = dst[ti, ri] + aijp * src[yi, xi+1] * inv_abs_sin

cdef inline int is_valid_index(int xi, int yi, double center_x, double center_y, int NoI):
    cdef double x = xi - center_x
    cdef double y = center_y - yi
    return 0 <= xi and xi < NoI \
      and 0 <= yi and yi < NoI \
      and 4 * (x * x + y * y) < NoI * NoI
