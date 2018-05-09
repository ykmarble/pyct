#!/usr/bin/env python2
# -*- coding:utf-8 -*-

import utils
import numpy
import os.path
import sys

class ExperimentWorker(object):
    """
    Worker for performing experiment.
    Following configurations are available and some of them must be set explicitely.

    algorithm configurations:
    * method: algorithm implementation
    * initial_x: initial value of image
    * initial_y: initial value of sinogram
    * initial_mu: initial value of dual variable
    * phi1: proximal operator for image domain
    * phi2: proximal operator for projection domain

    input configurations:
    * original_image_path: path to original image, sinogram will be calculated
    * sinogram_path: path to input sinogram
    * ground_truth_path: path to ground truth image
    * detectors_scale: length of detectors array normalized by image size
    * num_of_image_side_px
    * num_of_detectors
    NOTE: Abort if given configurations are collision.
          (ex. User specified original_image_path and sinogram_path)

    output configurations:

    """
    def __init__(self):
        pass

    def do():
        pass

def calc_initial_projection_data():
    # create initial projection data
    ##### begin: interior #####

    interior_w = b.shape[1]
    interior_pad = (y.shape[1] - interior_w) / 2  # MEMO: Some cases cause error.

    y[:, interior_pad:interior_pad + interior_w] = b

    # zero padding
    #y[:, :interior_pad] = 0
    #y[:, interior_pad + interior_w:] = 0

    # flip with reference to boundary
    #y[:, :interior_pad] = b[:, interior_pad:0:-1]
    #y[:, interior_pad + interior_w:] = b[:, -1:-interior_pad-1:-1]

    # padding with baundary value
    #y[:, :interior_pad] = b[:, 0, None]
    #y[:, interior_pad + interior_w:] = b[:, -1, None]

    # round padding data
    for i in xrange(y.shape[0]):
        y[i, :interior_pad] = utils.interpolate(0, b[i, 0], interior_pad)
        y[i, interior_pad + interior_w:] = utils.interpolate(b[i, -1], 0, interior_pad)
    ##### end: interior #####

    ##### begin: metal #####
    #y[:, :] = b[:, :]
    #utils.inpaint_metal(y)
    ##### end: metal #####


def gen_support_constraint():
    elipse_center = (numpy.array(x.shape)-1) / 2.
    elipse_r = (numpy.array(x.shape)-1) / 2.
    elipse = img.copy()
    utils.create_elipse_mask(elipse_center, elipse_r[0], elipse_r[1], elipse)

def gen_data_constraint():
    y[:, interior_pad:interior_pad + interior_w] = b

def

def main(method):
    HU_lim = [-1050., 1500.]
    scale = 0.8

    if len(sys.argv) != 2:
        print "Usage: {} <rawfile>"
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.exists(path):
        print "invalid path"
        sys.exit(1)

    img = load_rawimage(path)

    ##### begin: interior projection ######
    # interior projection
    proj = utils.create_projection(path, interior_scale=scale)
    interiorA = projector(NoI, NoA, NoD)
    interiorA.update_detectors_length(NoI*scale)


    # global projection
    full_proj, A = utils.create_projection(path, detector_scale=(1/scale)*1.5, angular_scale=1.5)

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
    callback = utils.IterViewer(img, roi, clim=(-110, 190))
    #callback = utils.IterLogger(img, roi, subname="_app_withfilter_"+name)

    # without filter
    #alpha = 0.004
    #alpha = 0.004
    #recon = app_recon(interiorA, proj, alpha, alpha,1000, iter_callback=callback)

    # with filter
    #alpha = 0.0465
    #alpha = 0.018

    #alpha = 1.5
    alpha = 0.04

    #alpha = 0.001
    #recon = app_recon(interiorA, proj, alpha, alpha, 500, iter_callback=callback,
    #                  tv_mask=roi, filtering=True) # 201435.93 400 iter
    recon = method(A, proj, alpha, alpha, 500, iter_callback=callback,
                          tv_mask=roi, filtering=True)
    #alpha = 0.0025

    #recon = fullapp_recon(A, proj, alpha*50, alpha/50, 1000, iter_callback=callback,
    #                      tv_mask=roi, filtering=False)
