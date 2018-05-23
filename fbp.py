#!/usr/bin/env python2

import ctfilter
from worker import main
import numpy
import utils

def fbp(A, data, recon):
    #ctfilter.ram_lak_filter(data, sample_rate=0.5)
    #data /= 500 * 4.25
    ctfilter.shepp_logan_filter(data, sample_rate=1.4142)
    data /= 44
    A.backward(data, recon)

if __name__ == '__main__':
    main(fbp)
