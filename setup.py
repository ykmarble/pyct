#!/usr/bin/env python2

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [
    Extension(
        "projector",
        ["projector.pyx"],
        libraries=["m"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']),
    Extension(
        "cython_filter",
        ["cython_filter.pyx"],
        libraries=["m"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']),
]

setup(
    name = 'ctpy',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
