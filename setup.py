#!/usr/bin/env python2

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("projector", ["projector.pyx"], libraries=["m"])]

setup(
  name = 'ctpy',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
