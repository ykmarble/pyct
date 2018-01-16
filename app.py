#!/usr/bin/env python2

from projector import Projector
#from differencial import Projector
import fbp
import utils
from skimage.restoration import denoise_tv_chambolle as sk_tv_denoise
from tv_denoise import tv_denoise_chambolle as tv_denoise
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


    # create initial projection data
    ##### begin: interior #####

    interior_w = data.shape[1]
    interior_pad = (recon_proj.shape[1] - interior_w) / 2  # MEMO: Some cases cause error.

    recon_proj[:, interior_pad:interior_pad + interior_w] = data

    # zero padding
    #recon_proj[:, :interior_pad] = 0
    #recon_proj[:, interior_pad + interior_w:] = 0

    # flip with reference to boundary
    #recon_proj[:, :interior_pad] = data[:, interior_pad:0:-1]
    #recon_proj[:, interior_pad + interior_w:] = data[:, -1:-interior_pad-1:-1]

    # padding with baundary value
    recon_proj[:, :interior_pad] = data[:, 0, None]
    recon_proj[:, interior_pad + interior_w:] = data[:, -1, None]

    # round padding data
    for i in xrange(recon_proj.shape[0]):
        recon_proj[i, :interior_pad] = utils.interpolate(0, data[i, 0], interior_pad)
        recon_proj[i, interior_pad + interior_w:] = utils.interpolate(data[i, -1], 0, interior_pad)
    ##### end: interior #####

    ##### begin: metal #####
    #recon_proj[:, :] = data[:, :]
    #utils.inpaint_metal(recon_proj)
    ##### end: metal #####

    elipse_center = (numpy.array(recon.shape)-1) / 2.
    elipse_r = (numpy.array(recon.shape)-1) / 2.
    elipse = img.copy()
    utils.create_elipse_mask(elipse_center, elipse_r[0], elipse_r[1], elipse)
    recon[elipse<0.5] = -1030

    alpha = 0.008

    for i in xrange(niter):
        A.forward(recon, proj)
        proj -= recon_proj

        #proj[proj_mask==1] = 0
        proj[:, :interior_pad] = 0
        proj[:, interior_pad + interior_w:] = 0

        if filtering:
            #fbp.shepp_logan_filter(proj, sample_rate=1.5)
            fbp.shepp_logan_filter(proj[:, interior_pad:interior_pad + interior_w], sample_rate=1.5)
        mu_bar = -mu.copy()
        mu += sigma * proj
        mu_bar += 2 * mu

        #mu_bar[proj_mask==1] = 0
        mu_bar[:, :interior_pad] = 0
        mu_bar[:, interior_pad + interior_w:] = 0

        A.backward(mu_bar, img)
        recon -= tau * img
        #tv_denoise(recon, tau / alpha, mask=tv_mask)
        recon = sk_tv_denoise(recon, weight=tau/alpha)

        recon_proj += tau * mu_bar

        recon_proj[:, interior_pad:interior_pad + interior_w] = data

        #recon_proj[proj_mask == 0] = data[proj_mask==0]

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
    #alpha = 0.01
    alpha = 1

    for i in xrange(niter):
        A.forward(recon, proj)
        proj -= data
        if filtering:
            fbp.shepp_logan_filter(proj, 1.4)
        mu_bar = -mu.copy()
        mu += sigma * proj
        mu_bar += 2 * mu
        A.backward(mu_bar, img)
        recon -= tau * img
        #recon[elipse] = 0
        tv_denoise(recon, tau / alpha, mask=tv_mask)
        iter_callback(i, recon, proj)

    return recon

def mask(proj, recon_proj):
    NoA, NoD = proj.shape
    for i in xrange(NoA):
        inf_len = 0
        bound_number = [0, 0]  # previous ct-number, next ct-number (both are not inf)
        for j in xrange(NoD):
            if proj[i, j] == float("inf"):
                if inf_len == 0 and j > 0:  # left bound of inf
                    bound_number[0] = proj[i, j-1]
                inf_len += 1
            elif inf_len > 0:  # right bound of inf
                bound_number[1] = proj[i, j]
                recon_proj[i, j-inf_len:j] = numpy.linspace(bound_number[0], bound_number[1], inf_len) \
                                             * numpy.abs(numpy.cos(numpy.linspace(0, numpy.pi, inf_len)))
                #proj[i, j-inf_len:j] = interpolate(bound_number[0], bound_number[1], inf_len)
                inf_len = 0
        if inf_len > 0:  # inf on right proj bound
            bound_number[1] = 0
            recon_proj[i, -inf_len:] = numpy.linspace(bound_number[0], bound_number[1], inf_len)  \
                                       * numpy.cos(numpy.linspace(0, numpy.pi, inf_len))
            #proj[i, j-inf_len:j] = interpolate(bound_number[0], bound_number[1], inf_len)

def main():
    import os.path
    if len(sys.argv) != 2:
        print "Usage: {} <rawfile>"
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.exists(path):
        print "invalid path"
        sys.exit(1)

    ##### begin: interior projection ######
    scale = 0.8
    # interior projection
    proj, img, interiorA = utils.create_projection(path, interior_scale=scale)

    # global projection
    full_proj, img, A = utils.create_projection(path, detector_scale=(1/scale)*1.5, angular_scale=1.5)

    # truncate global projection
    interior_w = int(img.shape[0]*1.5)
    interior_pad = (full_proj.shape[1] - interior_w) / 2  # MEMO: Some cases cause error.
    proj = full_proj[:, interior_pad:interior_pad + interior_w]
    truncate = (numpy.sin(numpy.linspace(-numpy.pi/2., numpy.pi/2., interior_pad/2))+1)/2.

    # create roi mask
    roi = utils.zero_img(interiorA)
    roi_c = ((roi.shape[0] - 1) / 2., (roi.shape[1] - 1) / 2.)
    roi_r = [roi.shape[0] * scale / 2., roi.shape[1] * scale / 2.]
    utils.create_elipse_mask(roi_c, roi_r[0], roi_r[1], roi)
    ##### end: interior projection ######

    ###### begin: metal projection #####
    #proj, img, A = utils.create_projection(path, detector_scale=1.5, angular_scale=1.5)
    #proj_clim = (numpy.min(proj), numpy.max(proj))
    #
    ## define metal material inner image
    #metal_mask = numpy.zeros_like(img, dtype=numpy.int)
    #mask_c = (-70 + (metal_mask.shape[0] - 1) / 2.0, -30 + (metal_mask.shape[1] - 1) / 2.0)
    #utils.create_elipse_mask(mask_c, 5, 8, metal_mask, 1)
    #masked_img = img.copy()
    #masked_img[metal_mask==1] = float("inf")
    #
    ##mask_c = (-80 + (metal_mask.shape[0] - 1) / 2.0, 30 + (metal_mask.shape[1] - 1) / 2.0)
    ##utils.create_elipse_mask(mask_c, 3, 2, metal_mask, 1)
    ##masked_img[metal_mask==1] = float("inf")
    #
    ### get metal mask of projection domain
    #proj_mask = numpy.zeros_like(proj, dtype=numpy.int)
    #tmp_proj = numpy.zeros_like(proj)
    #A.forward(masked_img, tmp_proj)
    #proj_mask[tmp_proj==float("inf")] = 1
    #recon_proj = proj.copy()
    #proj[proj_mask==1] = float("inf")
    ##mask(proj, recon_proj)
    #
    ## create roi mask
    #roi = utils.zero_img(A)
    #roi_c = ((roi.shape[0] - 1) / 2., (roi.shape[1] - 1) / 2.)
    #roi_r = [roi.shape[0] / 2., roi.shape[1] / 2.]
    #utils.create_elipse_mask(roi_c, roi_r[0], roi_r[1], roi)
    ###### end: metal projection #####

    #utils.show_image(masked_img, clim=(-110, 190))
    #utils.show_image(proj, clim=proj_clim)

    # setup callback routine
    (_, name, _) = utils.decompose_path(path)
    #callback = lambda *x: None
    #callback = utils.IterViewer(img, roi, clim=(-110, 190))
    #callback = utils.IterViewer(img, roi, clim=(-1053, 1053))
    callback = utils.IterLogger(img, roi, subname="_app_withfilter_"+name)

    # without filter
    #alpha = 0.004
    #alpha = 0.004
    #recon = app_recon(interiorA, proj, alpha, alpha,1000, iter_callback=callback)

    # with filter
    #alpha = 0.0465
    #alpha = 0.018

    #alpha = 1.5
    #alpha = 0.04

    alpha = 0.2
    #alpha = 0.0025
    #recon = app_recon(interiorA, proj, alpha, alpha, 500, iter_callback=callback,
    #                  tv_mask=roi, filtering=True) # 201435.93 400 iter
    recon = fullapp_recon(A, proj, alpha, alpha, 10000, iter_callback=callback,
                          tv_mask=roi, filtering=True)

if __name__ == '__main__':
    main()
