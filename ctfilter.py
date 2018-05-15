#!/usr/bin/env python2

import numpy.fft
import math


def cut_filter(proj, maxfreq):
    freq_proj = numpy.fft.fft(proj)
    NoA, NoD = freq_proj.shape
    ramp = numpy.ones(NoD, dtype=float)
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
    maxfreq = int(math.floor(((NoD - 1) / 2 + 1) / 1.41421356))
    ramp = numpy.linspace(0, 2, NoD)
    ramp[1:] = 1. / ramp[1:]
    ramp[0] = 2 * ramp[1] - ramp[2]
    ramp[maxfreq:] = 0
    ramp *= 2
    freq_proj *= ramp[None, :]
    proj[:] = numpy.fft.ifft(freq_proj).real


def shepp_logan_filter(proj, sample_rate=1):
    NoA, NoD = proj.shape
    filter_width = NoD + 1 if NoD % 2 == 0 else NoD  # must be a odd number larger than NoD
    filter_x = numpy.linspace(-(filter_width / 2), filter_width / 2, filter_width)
    filter_h = 1 / ((math.pi * sample_rate) ** 2 * (1 - 4 * filter_x ** 2))
    #proj3 = numpy.concatenate((proj[:, -NoD/2-1::-1], proj, proj[:, :NoD/2-1:-1]), axis=1)  # flip and concat `proj`
    proj3 = numpy.concatenate((numpy.zeros((NoA, NoD/2)), proj, numpy.zeros((NoA, NoD/2))), axis=1)  # padding 0
    for i in xrange(NoA):
        proj[i] = numpy.convolve(filter_h, proj3[i], "valid")


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
