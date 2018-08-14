#!/usr/bin/env python3

from pyct import ctfilter
from pyct import utils
from pyct.worker import main
import numpy

def fbp(A, data, recon):
    ctfilter.shepp_logan_filter(data)
    utils.show_image(data)
    A.backward(data, recon)

if __name__ == '__main__':
    main(fbp)
