#!/usr/bin/env python2

from projector import Projector
#from differencial import Projector
import fbp
import utils
from tv_denoise import tv_denoise_chambolle as tv_denoise
import numpy
from math import ceil, floor
import sys

def iterative_fbp(A, data, alpha, niter, recon=None, recon_proj=None, iter_callback=lambda *arg: 0, tv_mask=None):
    if recon is None:
        recon = utils.zero_img(A)
    if recon_proj is None:
        recon_proj = utils.zero_proj(A)

    img = utils.zero_img(A)
    proj = utils.zero_proj(A)

    interior_w = data.shape[1]
    interior_pad = (recon_proj.shape[1] - interior_w) / 2  # MEMO: Some cases cause error.

    # create initial projection data

    #recon_proj[:, interior_pad:interior_pad + interior_w] = data
    #
    #recon_proj[:, :interior_pad] = (data[:, 0])[:, None]
    #recon_proj[:, interior_pad + interior_w:] = (data[:, -1])[:, None]
    #recon_proj[:, :interior_pad] *= (numpy.linspace(0, 1, interior_pad))[None, :]
    #recon_proj[:, interior_pad + interior_w:] *= (numpy.linspace(1, 0, interior_pad))[None, :]

    elipse_center = numpy.array(recon.shape) / 2.
    elipse_r = numpy.array(recon.shape) / 2.1
    elipse = img.copy()
    #utils.create_elipse_mask(elipse_center, elipse_r[0], elipse_r[1], elipse)
    utils.create_elipse_mask(elipse_center, elipse_r[0]*1.5, elipse_r[1]*1.5, elipse) # Abd
    elipse = elipse < 0.5

    for i in xrange(niter):
        A.forward(recon, proj)
        proj -= recon_proj
        fbp.shepp_logan_filter(proj)
        A.backward(proj, img)
        recon -= alpha * img
        recon[elipse] = 0
        tv_denoise(recon, alpha)

        A.forward(recon, proj)
        recon_proj -= alpha * (recon_proj - proj)
        recon_proj[:, interior_pad:interior_pad + interior_w] = data
        iter_callback(i, recon, recon_proj)


def fullapp_recon(A, data, sigma, tau, niter, recon=None, mu=None, recon_proj=None,
              iter_callback=lambda *arg: 0, filtering=False, tv_mask=None):
    """
    Perform interior CT image reconstruction with APP method.
    @A : system matrix class
    @data : projection data
    @sigma : parameter
    @tau : parameter
    @niter : iteration times
    @iter_callback : callback function called each iterations
    @recon: initial image, `None` means using zero image
    @mu: initial mu, `None` means using zero
    """
    if recon is None:
        recon = utils.zero_img(A)
    if mu is None:
        mu = utils.zero_proj(A)
    if recon_proj is None:
        recon_proj = utils.zero_proj(A)
    #recon_proj += (numpy.max(data) + numpy.min(data)) / 2.
    img = utils.zero_img(A)
    proj = utils.zero_proj(A)
    proj_mask = numpy.zeros_like(data, numpy.int)
    proj_mask[data == float("inf")] = 1

    interior_w = data.shape[1]
    interior_pad = (recon_proj.shape[1] - interior_w) / 2  # MEMO: Some cases cause error.

    # create initial projection data

    recon_proj[:, interior_pad:interior_pad + interior_w] = data

    recon_proj[:, :interior_pad] = (data[:, 0])[:, None]
    recon_proj[:, interior_pad + interior_w:] = (data[:, -1])[:, None]
    recon_proj[:, :interior_pad] *= (numpy.linspace(0, 1, interior_pad))[None, :]
    recon_proj[:, interior_pad + interior_w:] *= (numpy.linspace(1, 0, interior_pad))[None, :]

    elipse_center = numpy.array(recon.shape) / 2.
    elipse_r = numpy.array(recon.shape) / 2.1
    elipse = img.copy()
    #utils.create_elipse_mask(elipse_center, elipse_r[0], elipse_r[1], elipse)
    utils.create_elipse_mask(elipse_center, elipse_r[0]*1.5, elipse_r[1]*1.5, elipse) # Abd
    elipse = elipse < 0.5
    alpha = 0.004

    for i in xrange(niter):
        A.forward(recon, proj)
        proj -= recon_proj
        if filtering:
            fbp.shepp_logan_filter(proj, sample_rate=1.41421356)
        mu_bar = -mu.copy()
        mu += sigma * proj
        mu_bar += 2 * mu

        A.backward(mu_bar, img)
        recon -= tau * img
        recon[elipse] = 0
        tv_denoise(recon, tau / alpha, mask=tv_mask)

        recon_proj += tau * mu_bar
        recon_proj[:, interior_pad:interior_pad + interior_w] = data
        iter_callback(i, recon, recon_proj, mu)

    return recon

def app_recon(A, data, sigma, tau, niter, recon=None, mu=None, sample_rate=1,
              iter_callback=lambda *arg: 0, filtering=False, tv_mask=None):
    """
    Perform interior CT image reconstruction with APP method.
    @A : system matrix class
    @data : projection data
    @sigma : parameter
    @tau : parameter
    @niter : iteration times
    @iter_callback : callback function called each iterations
    @recon: initial image, `None` means using zero image
    @mu: initial mu, `None` means using zero
    """
    if recon is None:
        recon = utils.zero_img(A)
    if mu is None:
        mu = utils.zero_proj(A)

    img = utils.empty_img(A)
    proj = utils.empty_proj(A)
    elipse_center = numpy.array(recon.shape) / 2.
    elipse_r = numpy.array(recon.shape) / 2.1
    elipse = img.copy()
    #utils.create_elipse_mask(elipse_center, elipse_r[0]*10/11, elipse_r[1]*1.1, elipse) # SLp
    utils.create_elipse_mask(elipse_center, elipse_r[0]*1.5, elipse_r[1]*1.5, elipse) # Abd
    elipse = elipse < 0.5
    alpha = 0.01

    for i in xrange(niter):
        A.forward(recon, proj)
        proj -= data
        if filtering:
            fbp.shepp_logan_filter(proj, sample_rate)
        mu_bar = -mu.copy()
        mu += sigma * proj
        mu_bar += 2 * mu
        A.backward(mu_bar, img)
        recon -= tau * img
        recon[elipse] = 0
        tv_denoise(recon, tau / alpha, mask=tv_mask)

        iter_callback(i, recon, mu)

    return recon


def main():
    import sys
    import os.path
    if len(sys.argv) != 2:
        print "Usage: {} <rawfile>"
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.exists(path):
        print "invalid path"
        sys.exit(1)

    scale = 0.4

    # interior projection
    proj, img, interiorA = utils.create_projection(path, interior_scale=scale)

    # global projection
    full_proj, img, A = utils.create_projection(path, detector_scale=1/scale)

    # truncate global projection
    interior_w = img.shape[0]
    interior_pad = (full_proj.shape[1] - interior_w) / 2  # MEMO: Some cases cause error.
    proj = full_proj[:, interior_pad:interior_pad + interior_w]
    truncate = (numpy.sin(numpy.linspace(-numpy.pi/2., numpy.pi/2., interior_pad/2))+1)/2.

    # create roi mask
    roi = utils.zero_img(interiorA)
    roi_c = ((roi.shape[0] - 1) / 2., (roi.shape[1] - 1) / 2.)
    roi_r = [roi.shape[0] * scale / 2., roi.shape[1] * scale / 2.]
    utils.create_elipse_mask(roi_c, roi_r[0], roi_r[1], roi)

    def callback(i, *argv):
        x = argv[0]
        y = argv[1]
        print i, numpy.min(x), numpy.max(x)
        if i != 0 and i % 10 == 0:
             #utils.save_rawimage(x, "app/{}.dat".format(i))
             utils.show_image(x, clim=(1000-100, 1512-100))

    # without filter
    #alpha = 0.004
    #alpha = 0.001
    #recon = app_recon(interiorA, proj, alpha, alpha,1000, iter_callback=callback) # 1635498

    # with filter
    #alpha = 0.0465
    alpha = 0.018
    #recon = app_recon(interiorA, proj, alpha, alpha, 1000, iter_callback=callback
    #                  filtering=True, tv_mask=roi) # 201435.93 400 iter
    recon = fullapp_recon(A, proj, alpha, alpha, 1000, iter_callback=callback, filtering=True)

if __name__ == '__main__':
    main()
