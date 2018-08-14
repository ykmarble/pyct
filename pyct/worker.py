#!/usr/bin/env python3

from .cProjector import Projector
#from differencial import Projector
from . import utils
from . import ctfilter
from .tv_denoise import tv_denoise_chambolle as prox_tv
from skimage.restoration import denoise_tv_chambolle as sk_tv
import numpy
import skimage.filters
import os.path
import sys


def interpolate_initial_y(truncated_y, inpaint_y, support=0):
    y = truncated_y.copy()
    y[inpaint_y == 1] = float("inf")
    utils.inpaint_metal(y, support)  # it's named by "metal" but this is apricable for truncated sinogram
    return y


def gen_support(img, center, r):
    # 自動で覆うような楕円を計算してくれるとハッピー
    elipse_center = center
    elipse_r = r

    #elipse_center = numpy.array(img.shape) / 2.
    #elipse_r = numpy.array(img.shape) / 2.
    #elipse_r[0] *= 0.95
    #elipse_r[1] *= 0.75

    elipse = numpy.zeros(img.shape)
    utils.create_elipse_mask(elipse_center, elipse_r[0], elipse_r[1], elipse)

    return elipse


def main(method):
    HU_lim = [0.45, 0.55]  # slp
    #HU_lim = [0., 0.08]  # lung
    #HU_lim = [0.45, 0.55]  # abd

    #scale = 0.65 #slp
    scale = 0.6  # abd
    #scale = 0.8  # lung

    if len(sys.argv) != 2:
        print("Usage: {} <rawfile>")
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.exists(path):
        print("invalid path")
        sys.exit(1)

    # derive projector constants
    img = utils.load_rawimage(path)
    NoI = img.shape[0]
    NoA = int(NoI * 1.2)
    NoD = int(NoI * 1.)
    full_NoA = NoA
    #full_NoD = int(round(NoD*1/scale))  # slp
    #center_x = (NoI - 1) / 2.  # slp
    #center_y = (NoI - 1) / 2.  # slp
    #full_NoD = int(round(NoD*1/scale)+50)  # lung
    #center_x = (NoI - 1) / 2. -20 # lung
    #center_y = (NoI - 1) / 2.  # lung
    full_NoD = int(round(NoD*1/scale))  # abd
    center_x = (NoI - 1) / 2. # abd
    center_y = (NoI - 1) / 2. # abd

    # interior projection
    interior_A = Projector(NoI, NoA, NoD)
    interior_A.update_detectors_length(NoI*scale)
    interior_A.update_center_x(center_x)
    interior_A.update_center_y(center_y)
    interior_proj = utils.zero_proj(interior_A)
    interior_A.forward(img, interior_proj)

    # global projection
    full_A = Projector(NoI, full_NoA, full_NoD)
    full_A.update_center_x(center_x)
    full_A.update_center_y(center_y)
    #full_A.update_detectors_length(NoI)  #slp
    #full_A.update_detectors_length(NoI+50)  # lung
    full_A.update_detectors_length(NoI+0)  # abd
    full_proj = utils.zero_proj(full_A)
    full_A.forward(img, full_proj)

    # calculate roi and its sinogram
    xmask = utils.zero_img(full_A)
    utils.create_elipse_mask((full_A.center_x, full_A.center_y), NoI/2.*scale, NoI/2.*scale, xmask)
    ymask = utils.zero_proj(full_A)
    full_A.forward(xmask, ymask)
    ymask[ymask != 0] = 1

    #supmask = gen_support(img, (full_A.center_x, full_A.center_y), (NoI/2.-10, NoI/2.))  # slp
    #supmask = gen_support(img, (NoI/2.-5, NoI/2.), (NoI/2.+20, NoI/2.-20))  # lung
    supmask = gen_support(img, (NoI/2., NoI/2.), (NoI/2., NoI/2.-20))  # abd
    supmask_sin = utils.zero_proj(full_A)
    full_A.forward(supmask, supmask_sin)
    supmask_sin[supmask_sin!=0] = 1
    utils.show_image(supmask*img)

    # create truncated sinogram
    #full_A.forward(img, full_proj)
    scout_proj = full_proj.copy()
    support = full_proj[0, 0]
    full_proj[ymask == 0] = 0

    # estimate initial value
    initial_x = utils.zero_img(full_A)
    initial_y = interpolate_initial_y(full_proj, (supmask_sin - ymask), support)
    #initial_y = full_proj

    # generate proximal operators
    #tv_alpha = 0.005  # slp
    #tv_alpha = 0.008  # lung
    tv_alpha = 0.012  # abd

    def prox_tv_all(x, alpha=1.):
        x[:, :] = sk_tv(x, alpha * tv_alpha)

    def prox_tv_masked(x):
        prox_tv(x, tv_alpha, mask=xmask)

    def prox_sup(x):
        x[supmask == 0] = 0
        x[x < 0] = 0
        x[x > 1] = 1

    known_mask = utils.zero_img(full_A)
    utils.create_elipse_mask((full_A.center_x+0, full_A.center_y+0), 60, 10, known_mask)
    known = img.copy()
    known[known_mask != 1] = 0

    def prox_known(x):
        x[known_mask == 1] = known[known_mask == 1]

    edge_width_2 = 5
    edge_mask = utils.zero_img(full_A)
    img_center = (full_A.center_x, full_A.center_y)
    roi_width = full_A.NoI*scale/2
    utils.crop_elipse(edge_mask, img_center, roi_width+edge_width_2, roi_width+edge_width_2, value=1)
    utils.crop_elipse(edge_mask, img_center, roi_width-edge_width_2, roi_width-edge_width_2, value=0)

    def prox_edgeblur(x):
        img = skimage.filters.gaussian(x, sigma=3)
        x[edge_mask == 1] = img[edge_mask == 1]
        x[:, :] = skimage.filters.gaussian(x)

    def phi_x(x, alpha=1.):
        prox_sup(x)
        #prox_tv_masked(x)
        #prox_tv_all(x, alpha=alpha)
        prox_known(x)

    def prox_b(y):
        y[ymask == 1] = full_proj[ymask == 1]

    scout_mask = utils.zero_proj(full_A)
    scout_mask[0] = 1
    scout_mask[scout_mask.shape[0]//2] = 1

    def prox_scout(y):
        y[scout_mask == 1] = scout_proj[scout_mask == 1]

    def prox_sup_sin(y):
        y[supmask_sin == 0] = 0

    def phi_y(y):
        prox_b(y)
        #prox_scout(y)
        prox_sup_sin(y)

    def G_id(y):
        y *= 0.001

    def G_sh(y):
        ctfilter.shepp_logan_filter(y)

    def G_hl(y):
        y[:] = dbp.freq_hilbert_filter(y, scale=1)
        y /= 360.

    def G_ir(y):
        ctfilter.inv_ramp_filter(y)
        y /= 1500.

    # setup callback routine
    viewer = utils.IterViewer(img, interior_proj, xmask, interior_A, clim=HU_lim)
    logger = utils.IterLogger(img, interior_proj, xmask, interior_A, subname="", no_forward=True)

    niter = 200

    method_id = os.path.basename(sys.argv[0])

    # setup method specific configuration and start calculation
    if "original.py" == method_id:
        utils.show_image(img*xmask, clim=HU_lim)

    if "fbp.py" == method_id:
        #method(interior_A, interior_proj, initial_x)
        method(full_A, initial_y, initial_x)
        print(numpy.min(initial_x), numpy.max(initial_x))
        utils.show_image(initial_x*xmask, HU_lim)
        logger(9, initial_x)

    if "iterative_fbp.py" == method_id:
        alpha = 0.5
        phi_x = prox_known
        method(interior_A, interior_proj, alpha, niter,
               phi_x=phi_x,
               G=G_sh,
               x=initial_x,
               iter_callback=viewer)

    if "estimate_missing_line.py" == method_id:
        alpha = 0.04
        phi_x = prox_known
        method(full_A, full_proj, alpha, niter,
               phi_x=phi_x,
               phi_y=phi_y,
               G=G_sh,
               x=initial_x,
               y=initial_y,
               iter_callback=viewer)

    if "sirt.py" == method_id:
        alpha = 0.1
        method(interior_A, interior_proj, alpha, niter,
               x=initial_x,
               iter_callback=viewer)

    if "os_sart.py" == method_id:
        alpha = 0.8
        nsubset = 20
        method(interior_A, interior_proj,
               alpha=alpha,
               nsubset=nsubset,
               niter=niter,
               x=initial_x,
               iter_callback=viewer)

    if "os_sart_tv.py" == method_id:
        os_alpha = 0.8
        tv_alpha = 0.08
        tv_alpha_s = 0.997
        nsubset = 20
        ntv = 3
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
        #alpha = 0.8 # slp
        #alpha = 0.8  # lung
        alpha = 0.75 # abd
        #def phi_y(y):
        #    y[:, :] = interior_proj
        #method(interior_A, 0.4, alpha, niter,
        method(full_A, alpha, alpha, niter,
               phi_x=phi_x,
               phi_y=phi_y,
               G=G_sh,
               x=initial_x,
               y=initial_y,
               #y=None,
               mu=None,
               iter_callback=viewer)

    if "cp.py" == method_id:
        #alpha = 0.05  # slp
        #alpha = 0.09  # lung
        alpha = 0.06  # abd
        #def phi_y(y):
        #    y[:, :] = interior_proj
        #method(interior_A, 0.4, alpha, niter,
        method(interior_A, interior_proj, alpha, alpha, tv_alpha, niter,
               x=initial_x,
               iter_callback=logger)

    if "ladmm.py" == method_id:
        #alpha = 0.004
        alpha = 0.01
        #def phi_y(y):
        #    y[:, :] = interior_proj[:, :]
        #method(interior_A, alpha, niter,
        method(full_A, alpha, niter,
               phi_x=phi_x,
               phi_y=phi_y,
               G=G_sh,
               x=initial_x,
               y=initial_y,
               #y=None,
               mu=None,
               iter_callback=viewer)

    if "tpv.py" == method_id:
        p = 0.4
        v = 0.6
        ramda = 5e-4
        eps = 0
        tau = 0.4
        sigma = tau
        eta = 1e-4
        method(interior_A, interior_proj, p, v, ramda, eps, tau, sigma, eta, niter,
               iter_callback=viewer
               )



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
