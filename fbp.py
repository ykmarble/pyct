#!/usr/bin/env python2

import numpy
import numpy.fft
import math

def fbp(A, proj, recon):
    freq_proj = numpy.fft.fft(proj)
    NoA, NoD = freq_proj.shape
    maxfreq = NoD / 2
    ramp = numpy.linspace(0, 1, NoD)
    ramp[maxfreq:] = 0
    freq_proj *= ramp[None, :]
    A.backward(numpy.fft.ifft(freq_proj).real, recon)

def cut_filter(proj):
    freq_proj = numpy.fft.fft(proj)
    NoA, NoD = freq_proj.shape
    maxfreq = int(math.floor(((NoD - 1) / 2 + 1) / 1.41421356))
    ramp = numpy.ones(NoD)
    ramp[maxfreq:] = 0
    freq_proj *= ramp[None, :]
    proj[:] = numpy.fft.ifft(freq_proj).real

def ramp_filter(proj):
    freq_proj = numpy.fft.fft(proj)
    NoA, NoD = freq_proj.shape
    maxfreq = int(math.floor(((NoD - 1) / 2 + 1) / 1.41421356))
    ramp = numpy.linspace(0, 2, NoD)
    ramp[maxfreq:] = 0
    freq_proj *= ramp[None, :]
    proj[:] = numpy.fft.ifft(freq_proj).real

def inv_ramp_filter(proj):
    freq_proj = numpy.fft.fft(proj)
    NoA, NoD = freq_proj.shape
    epsilon = 1. / NoD
    maxfreq = int(math.floor(((NoD - 1) / 2 + 1) / 1.41421356))
    ramp = numpy.linspace(0, 2, NoD)
    ramp[:] = 1 / (ramp + epsilon)
    ramp[maxfreq:] = 0
    freq_proj *= ramp[None, :]
    proj[:] = numpy.fft.ifft(freq_proj).real

def shepp_logan_filter(proj, sample_rate=1):
    NoA, NoD = proj.shape
    filter_width = NoD + 1 if NoD % 2 == 0 else NoD  # must be a odd number larger than NoD
    filter = numpy.linspace(-(filter_width / 2), filter_width / 2, filter_width)
    filter = 1 / (math.pi * sample_rate ** 2 * (1 - 4 * filter ** 2))
    for i in xrange(NoA):
        proj[i] = numpy.convolve(filter, proj[i])[filter_width/2:filter_width/2+NoD]

def ram_lak_filter(proj, sample_rate=1):
    NoA, NoD = proj.shape
    filter_width = NoD + 1 if NoD % 2 == 0 else NoD  # must be a odd number larger than NoD
    even_index = ((filter_width - 1) / 2) % 2
    filter = numpy.linspace(-(filter_width / 2), filter_width / 2, filter_width)
    filter = -2 / (math.pi * filter ** 2 * sample_rate ** 2)
    filter[[i % 2 == even_index for i in xrange(filter_width)]] = 0
    filter[filter_width / 2] = math.pi / 2 / sample_rate ** 2
    for i in xrange(NoA):
        proj[i] = numpy.convolve(filter, proj[i])[filter_width/2:filter_width/2+NoD]
