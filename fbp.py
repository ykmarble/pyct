#!/usr/bin/env python2

import ctfilter
from worker import main


def fbp(A, data, recon):
    ctfilter.shepp_logan_filter(data, sample_rate=1.41421356)
    A.backward(data, recon)


if __name__ == '__main__':
    main(fbp)
