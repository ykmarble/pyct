#!/usr/bin/env python2

import ctfilter
from worker import main
import numpy
import utils

def fbp(A, data, recon):
    ctfilter.shepp_logan_filter(data)
    A.backward(data, recon)

if __name__ == '__main__':
    main(fbp)
