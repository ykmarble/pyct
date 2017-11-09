#!/usr/bin/env python2

from tv_denoise import tv_denoise_chambolle as tv_denoise
import projector
import utils
import numpy
import numpy.fft
import math
import sys

#def fbp(A, proj, recon):
#    freq_proj = numpy.fft.fft(proj)
#    NoA, NoD = freq_proj.shape
#    maxfreq = NoD / 2
#    ramp = numpy.linspace(0, 1, NoD)
#    ramp[maxfreq:] = 0
#    freq_proj *= ramp[None, :]
#    A.backward(numpy.fft.ifft(freq_proj).real, recon)

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
    for i in xrange(NoA):
        th = numpy.pi * i / NoA
        if th < numpy.pi / 4 or th > numpy.pi * 3/4:
            sample_rate = abs(1./numpy.cos(th)) * 2
        else:
            sample_rate = 1./numpy.sin(th) * 2
        filter_h = 1 / (math.pi * sample_rate ** 2 * (1 - 4 * filter_x ** 2))
        proj[i] = numpy.convolve(filter_h, proj[i])[filter_width/2:filter_width/2+NoD]

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

def iterative_fbp(A, data, alpha, niter, recon=None, iter_callback=lambda *arg: 0):
    if recon is None:
        recon = utils.zero_img(A)
    img = utils.zero_img(A)
    proj = utils.zero_proj(A)
    for i in xrange(niter):
        A.forward(recon, proj)
        proj -= data
        shepp_logan_filter(proj)
        A.backward(proj, img)
        recon -= alpha * img
        iter_callback(i, recon)

def main():
    path = sys.argv[1]
    scale = 0.4
    proj, orig, A = utils.create_projection(path, interior_scale=scale)
    roi_mask = utils.zero_img(A)
    roi_c = (roi_mask.shape[0]-1) / 2.
    roi_r = roi_mask.shape[0] / 2. * scale
    utils.create_elipse_mask((roi_c, roi_c), roi_r, roi_r, roi_mask)
    #A = projector.Projector(proj.shape[0], proj.shape[1], proj.shape[0])
    #A.update_detectors_length(proj.shape[0] * scale)
    recon = utils.zero_img(A)
    def iter_callback(i, x):
        tv_denoise(x, 1, mask=roi_mask)
        print i, numpy.min(x*roi_mask), numpy.max(x*roi_mask)
        if i != 0 and i % 10 == 0:
            utils.show_image(x*roi_mask, clim=(1000, 1255))
    print numpy.min(orig), numpy.max(orig)
    utils.show_image(orig, clim=(1000, 1255))
    iterative_fbp(A, proj, 0.001, 1000, recon=recon, iter_callback=iter_callback)


if __name__ == '__main__':
    main()
