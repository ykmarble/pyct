#!/usr/bin/env python2

import numpy
import numpy.fft

def fbp(A, proj, recon):
    freq_proj = numpy.fft.fft(proj)
    NoA, NoD = freq_proj.shape
    maxfreq = NoD / 2
    ramp = numpy.linspace(0, 1, NoD)
    ramp[maxfreq:] = 0
    freq_proj *= ramp[None, :]
    A.backward(numpy.fft.ifft(freq_proj).real, recon)
