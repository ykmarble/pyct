#!/usr/bin/env python2
# -*- coding:utf-8 -*-

from projector import Projector
import utils
import ctfilter
from tv_denoise import tv_denoise_chambolle as prox_tv
import numpy
import os.path
import sys


def interpolate_initial_y(truncated_y, mask_y):
    y = truncated_y.copy()
    y[mask_y == 0] = float("inf")
    utils.inpaint_metal(y)  # it's named by "metal" but this is apricable for truncated sinogram

    return y


def gen_support_constraint(img, center, r):
    # 自動で覆うような楕円を計算してくれるとハッピー
    elipse_center = (numpy.array(img.shape)-1) / 2.
    elipse_r = (numpy.array(img.shape)-1) / 2.
    elipse = numpy.zeros(img.shape)

    utils.create_elipse_mask(elipse_center, elipse_r[0], elipse_r[1], elipse)

    def proj_elipse_set(x):
        x[elipse != 1] = 0

    return proj_elipse_set


def main(method):
    HU_lim = [-1050., 1500.]
    HU_lim = [-110, 190]

    scale = 0.75

    if len(sys.argv) != 2:
        print "Usage: {} <rawfile>"
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.exists(path):
        print "invalid path"
        sys.exit(1)

    img = utils.load_rawimage(path)
    NoI = img.shape[0]
    NoA = int(NoI * 1.5)
    NoD = int(NoI * 1.5)

    # interior projection
    interior_proj = utils.create_sinogram(img, NoA, NoD, scale=scale, projector=Projector)
    interior_A = Projector(NoI, NoA, NoD)
    interior_A.update_detectors_length(NoI*scale)

    # global projection
    full_NoA = NoA
    full_NoD = int(round(NoD*1/scale))
    assert NoD == int(round(full_NoD * scale))
    full_A = Projector(NoI, NoA, full_NoD)

    # calculate roi and its sinogram
    xmask = utils.zero_img(full_A)
    utils.create_elipse_mask((full_A.center_x, full_A.center_y), NoI/2*scale, NoI/2*scale, xmask)
    ymask = utils.zero_proj(full_A)
    full_A.forward(xmask, ymask)
    ymask[ymask != 0] = 1

    # create truncated sinogram
    full_proj = utils.create_sinogram(img, full_NoA, full_NoD, projector=Projector)
    full_proj[ymask == 0] = 0

    # estimate initial value
    initial_x = utils.zero_img(full_A)
    initial_y = interpolate_initial_y(full_proj, ymask)

    # generate proximal operators
    tv_alpha = 10

    def prox_tv_all(x):
        prox_tv(x, tv_alpha)

    def prox_tv_masked(x):
        prox_tv(x, tv_alpha, mask=xmask)

    def prox_sup(x):
        gen_support_constraint(img, (full_A.center_x, full_A.center_y), (NoI/2, NoI/2))

    knownmask = utils.zero_img(full_A)
    utils.create_elipse_mask((full_A.center_x+80, full_A.center_y), 10, 10, knownmask)
    known = img.copy()
    known[knownmask != 1] = -2000

    def prox_known(x):
        x[knownmask == 1] = known[knownmask == 1]

    def phi_x(x):
        #prox_sup(x)
        prox_tv_masked(x)

    def phi_y(y):
        y[ymask != 0] = full_proj[ymask != 0]

    G_id = lambda y: y

    def G_sh(y):
        ctfilter.shepp_logan_filter(y, sample_rate=2)

    # setup callback routine
    viewer = utils.IterViewer(img, xmask, clim=HU_lim)
    logger = utils.IterLogger(img, xmask)

    niter = 500

    method_id = os.path.basename(sys.argv[0])

    # setup method specific configuration and start calculation
    if "fbp.py" == method_id:
        method(full_A, initial_y, initial_x)
        utils.show_image(initial_x*xmask, clim=HU_lim)

    if "iterative_fbp.py" == method_id:
        #alpha = 0.00001  # id
        alpha = 0.004  # sh
        phi_x = prox_known
        method(interior_A, interior_proj, alpha, niter,
               #phi_x=phi_x,
               G=G_sh,
               x=initial_x,
               iter_callback=viewer)

    if "sirt.py" == method_id:
        alpha = 0.00001
        method(interior_A, interior_proj, alpha, niter,
               x=initial_x,
               iter_callback=viewer)

    if "os_sart.py" == method_id:
        alpha = 0.9
        nsubset = 10
        method(interior_A, interior_proj,
               alpha=alpha,
               nsubset=nsubset,
               niter=niter,
               x=initial_x,
               iter_callback=viewer)

    if "os_sart_tv.py" == method_id:
        os_alpha = 0.9
        tv_alpha = 1
        tv_alpha_s = 0.9997
        nsubset = 20
        ntv = 5
        method(interior_A, interior_proj,
               os_alpha=os_alpha,
               tv_alpha=tv_alpha,
               tv_alpha_s=tv_alpha_s,
               nsubset=nsubset,
               ntv=ntv,
               niter=niter,
               x=initial_x,
               iter_callback=viewer)

    if "app.py" == method_id:
        alpha = 0.1  # app
        #phi_x = prox_known
        def phi_y(y):
            y[:, :] = interior_proj
        method(interior_A, alpha, alpha, niter,
        #method(full_A, alpha, alpha, niter,
               phi_x=phi_x,
               phi_y=phi_y,
               G=G_sh,
               x=initial_x,
               #y=initial_y,
               y=None,
               mu=None,
               iter_callback=viewer)


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
