#!/usr/bin/env python2

from projector import Projector
#from differencial import Projector
import fbp
import utils
from tv_denoise import tv_denoise_chambolle as tv_denoise
from skimage.restoration import denoise_tv_chambolle as sk_tv_denoise
import numpy
from scipy import ndimage
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
    elipse_r = numpy.array(recon.shape) / 2.
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
    recon_proj[:, :interior_pad] = data[:, interior_pad:0:-1]
    recon_proj[:, interior_pad + interior_w:] = data[:, -1:-interior_pad-1:-1]

    #for i in xrange(recon_proj.shape[0]):
        #recon_proj[i, :interior_pad] = utils.interpolate(0, data[i, 0], interior_pad)
        #recon_proj[i, interior_pad + interior_w:] = utils.interpolate(data[i, -1], 0, interior_pad)

    fbp.shepp_logan_filter(recon_proj)
    A.backward(recon_proj, img)
    A.forward(img, recon_proj)
    recon_proj = (numpy.max(data) - numpy.min(data)) / (numpy.max(recon_proj) - numpy.min(recon_proj)) * (recon_proj - numpy.min(recon_proj)) + numpy.min(data)

    utils.show_image(recon_proj)

    elipse_center = (numpy.array(recon.shape)-1) / 2.
    elipse_r = (numpy.array(recon.shape)-1) / 2.
    elipse = img.copy()
    #utils.create_elipse_mask(elipse_center, elipse_r[0], elipse_r[1], elipse)
    utils.create_elipse_mask(elipse_center, elipse_r[0], elipse_r[1], elipse) # Abd
    alpha = 0.005

    recon[elipse<0.5] = -1053

    for i in xrange(niter):
        A.forward(recon, proj)
        proj -= recon_proj
        if filtering:
            fbp.shepp_logan_filter(proj, sample_rate=2.5)
        mu_bar = -mu.copy()
        mu += sigma * proj
        mu_bar += 2 * mu

        A.backward(mu_bar, img)
        recon -= tau * img
        #recon[elipse] = 0
        #recon = sk_tv_denoise(recon, tau / alpha)
        ndimage.gaussian_filter(recon, output=recon, sigma=0.3)
        tv_denoise(recon, tau / alpha)
        recon_proj += tau * mu_bar
        recon_proj[:, interior_pad:interior_pad + interior_w] = \
            recon_proj[:, interior_pad:interior_pad + interior_w]*0.9 + data*0.1
        #recon_proj[:, interior_pad:interior_pad + interior_w] = data
        iter_callback(i, recon, recon_proj)

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
            fbp.shepp_logan_filter(proj, 2.5)
        mu_bar = -mu.copy()
        mu += sigma * proj
        mu_bar += 2 * mu
        A.backward(mu_bar, img)
        recon -= tau * img
        #recon[elipse] = 0
        tv_denoise(recon, tau / alpha, mask=tv_mask)
        iter_callback(i, recon, proj)

    return recon


def main():
    import os.path
    if len(sys.argv) != 2:
        print "Usage: {} <rawfile>"
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.exists(path):
        print "invalid path"
        sys.exit(1)

    scale = 0.8

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

    # setup callback routine
    (_, name, _) = utils.decompose_path(path)
    callback = utils.IterViewer(img, roi, clim=(-110, 190))
    #callback = utils.IterViewer(img, roi, clim=(-1053, 1053))
    #callback = utils.IterLogger(img, roi, subname="fapp_moving"+name)

    # without filter
    #alpha = 0.004
    #alpha = 0.004
    #recon = app_recon(interiorA, proj, alpha, alpha,1000, iter_callback=callback)

    # with filter
    #alpha = 0.0465
    #alpha = 0.018

    alpha = 0.002
    alpha = 0.15
    #recon = app_recon(interiorA, proj, alpha, alpha, 500, iter_callback=callback,
    #                  tv_mask=roi, filtering=True) # 201435.93 400 iter
    recon = fullapp_recon(A, proj, alpha*2, alpha, 500, iter_callback=callback,
                          tv_mask=roi, filtering=True)

if __name__ == '__main__':
    main()
