#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy
cimport numpy
DTYPE = numpy.float
ctypedef numpy.float_t DTYPE_t
cimport cython
from cython.parallel import *
from libc.math cimport M_PI


@cython.boundscheck(False)
def weighted_hilbert_filter(numpy.ndarray[DTYPE_t, ndim=2] f, numpy.ndarray[DTYPE_t, ndim=1] W, float W_sup):
    cdef int size = f.shape[1]
    cdef numpy.ndarray[DTYPE_t, ndim=1] sl = numpy.linspace(-W_sup, W_sup, f.shape[1])
    cdef numpy.ndarray[DTYPE_t, ndim=1] tl = numpy.linspace(-W_sup, W_sup, f.shape[1])
    cdef numpy.ndarray[DTYPE_t, ndim=2] out = numpy.zeros_like(f, dtype=float)

    for i in xrange(f.shape[0]):
        for j in xrange(size):
            out[i, j] = sum([f[i, k] / (sl[k] - tl[j]) * W[k] for k in xrange(size)
                            if sl[k] - tl[j] != 0 and -1 <= sl[k] <= 1]) / W[j]
    scale = size / (W_sup * 2)
    out /= M_PI
    out /= scale
    return out
