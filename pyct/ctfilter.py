#!/usr/bin/env python3

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
    maxfreq = (NoD - 1) // 2 + 1
    ramp = numpy.linspace(0, 2, NoD)
    ramp[maxfreq:] = 0
    freq_proj *= ramp[None, :]
    proj[:] = numpy.fft.ifft(freq_proj).real


def inv_ramp_filter(proj):
    freq_proj = numpy.fft.fft(proj)
    NoA, NoD = freq_proj.shape
    maxfreq = (NoD - 1) // 2 + 1
    ramp = numpy.linspace(0, 2, NoD)
    ramp[1:] = 1. / ramp[1:]
    ramp[0] = 1e-3 #2 * ramp[1] - ramp[2]
    ramp[maxfreq:] = 0
    ramp *= 2
    freq_proj *= ramp[None, :]
    proj[:] = numpy.fft.ifft(freq_proj).real


def shepp_logan_filter(proj, sample_rate=1):
    NoA, NoD = proj.shape
    filter_width = NoD + 1 if NoD % 2 == 0 else NoD  # must be a odd number larger than NoD
    filter_x = numpy.linspace(-(filter_width//2), filter_width//2, filter_width)
    filter_h = 4 / (math.pi * sample_rate**2 * (1 - 4 * filter_x**2))
    proj3 = numpy.concatenate((numpy.zeros((NoA, NoD//2)), proj, numpy.zeros((NoA, NoD//2))), axis=1)  # padding 0
    for i in range(NoA):
        proj[i] = numpy.convolve(filter_h, proj3[i], "valid")


def ram_lak_filter(proj, sample_rate=1):
    NoA, NoD = proj.shape
    filter_width = NoD + 1 if NoD % 2 == 0 else NoD  # must be a odd number larger than NoD
    filter_x = numpy.linspace(-(filter_width // 2), filter_width // 2, filter_width)
    ci = filter_width // 2
    odds = [(i - ci) % 2 == 1 for i in range(filter_width)]
    filter_h = numpy.zeros_like(filter_x)
    filter_h[odds] = -2 / (filter_x[odds] * sample_rate)** 2 / math.pi
    filter_h[ci] = math.pi / 2 / sample_rate**2
    proj3 = numpy.concatenate((numpy.zeros((NoA, NoD//2)), proj, numpy.zeros((NoA, NoD//2))), axis=1)  # padding 0
    for i in range(NoA):
        proj[i] = numpy.convolve(filter_h, proj3[i], "valid")


def fir_gauss_1d(proj):
    """
    Apply finite inpluse response gaussian bluring whose sigma is 0.8.
    Window length is 3.
    """
    NoA, NoD = proj.shape
    h = numpy.array([ 0.23899427,  0.52201147,  0.23899427])  # N(0, 0.8)
    for i in range(NoA):
        proj[i] = numpy.convolve(h, proj[i], "same")


def local_shepp_logan_filter(proj, sample_rate=1, filter_width=3):
    NoA, NoD = proj.shape
    filter_x = numpy.linspace(-(filter_width//2), filter_width//2, filter_width)
    filter_h = 2 / ((math.pi * sample_rate) ** 2 * (1 - 4 * filter_x ** 2))
    proj3 = numpy.concatenate((numpy.zeros((NoA, filter_width//2)), proj, numpy.zeros((NoA, filter_width//2))), axis=1)
    for i in range(NoA):
        proj[i] = numpy.convolve(filter_h, proj3[i], "valid")


#def weighted_hilbert_filter(f, W):
#    assert len(f.shape) == 2
#    sl = numpy.linspace(-W_sup, W_sup, f.shape[1])
#    out = numpy.zeros_like(f, dtype=float)
#    for i in xrange(f.shape[0]):
#        print i
#        for j in xrange(f.shape[1]):
#            t = j / float(f.shape[1]) * 2 - 1
#            out[i, j] = sum(f[i, k] / (s - t) * W[k] for k, s in enumerate(sl) if s - t != 0 and -1 <= s <= 1) / W[j]
#    out /= math.pi
#    return out


def finite_hilbert_filter(f, W_sup=0.85):
    return -weighted_hilbert_filter(f, numpy.ones(f.shape[1]), W_sup)


def inv_finite_hilbert_filter(f, W_sup=0.85):
    W = numpy.sqrt(1 - numpy.linspace(-W_sup, W_sup, f.shape[1])**2)
    return weighted_hilbert_filter(f, W, W_sup) / W


def freq_hilbert_filter(f, scale=10):
    j = complex("j")
    assert len(f.shape) == 2
    n = f.shape[1] * scale
    F = numpy.fft.fft(f, n)
    H = numpy.zeros(n, dtype=complex)
    H[1:n//2] = -j
    H[n//2+1:] = j
    F *= H
    return numpy.fft.ifft(f).real


def inv_freq_hilbert_filter(f):
    return -freq_hilbert_filter(f, 1)[:, :256]


#def weighted_hilbert_filter(f, W):
#    assert len(f.shape) == 2
#    sl = numpy.linspace(-W_sup, W_sup, f.shape[1])
#    out = numpy.zeros_like(f, dtype=float)
#    for i in xrange(f.shape[0]):
#        print i
#        for j in xrange(f.shape[1]):
#            t = j / float(f.shape[1]) * 2 - 1
#            out[i, j] = sum(f[i, k] / (s - t) * W[k] for k, s in enumerate(sl) if s - t != 0 and -1 <= s <= 1) / W[j]
#    out /= math.pi
#    return out


def finite_hilbert_filter(f, W_sup=0.85):
    return -weighted_hilbert_filter(f, numpy.ones(f.shape[1]), W_sup)


def inv_finite_hilbert_filter(f, W_sup=0.85):
    W = numpy.sqrt(1 - numpy.linspace(-W_sup, W_sup, f.shape[1])**2)
    return weighted_hilbert_filter(f, W, W_sup) / W


def freq_hilbert_filter(f, scale=10):
    j = complex("j")
    assert len(f.shape) == 2
    n = f.shape[1] * scale
    F = numpy.fft.fft(f, n)
    H = numpy.zeros(n, dtype=complex)
    H[1:n//2] = -j
    H[n//2+1:] = j
    F *= H
    return numpy.fft.ifft(f).real


def inv_freq_hilbert_filter(f):
    return -freq_hilbert_filter(f, 1)[:, :256]

