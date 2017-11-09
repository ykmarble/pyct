#!/usr/bin/env python2

from projector import Projector
#from differencial import Projector
import fbp
import utils
from tv_denoise import tv_denoise_chambolle as tv_denoise
import numpy
from math import ceil, floor
import sys

def fullapp_recon(A, data, sigma, tau, niter, recon=None, mu=None, sample_rate=1,
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
            #fbp.ramp_filter(proj)
            #fbp.ram_lak_filter(proj, sample_rate)
            fbp.shepp_logan_filter(proj, sample_rate)
            #fbp.inv_ramp_filter(proj)
        #mu_bar = mu + sigma * proj
        mu_bar = -mu.copy()
        mu += sigma * proj
        mu_bar += 2 * mu

        A.backward_with_mask(mu_bar, img, proj_mask)
        recon -= tau * img

        # insert support constraint
        recon[elipse] = 0

        tv_denoise(recon, tau / alpha, mask=tv_mask)

        recon_proj += tau * mu_bar
        recon_proj[:, interior_pad:interior_pad + interior_w] = data
        iter_callback(i, recon, recon_proj, mu)

        #recon_proj[:, :interior_pad] = 0
        #recon_proj[:, interior_pad + interior_w:] = 0
        #utils.show_image(recon_proj)
        #fbp.cut_filter(recon_proj[:, :interior_pad], interior_pad / 4)
        #utils.show_image(recon_proj)

        #recon_proj[data != float("inf")] = data[data != float("inf")]

        #A.forward(recon, proj)
        ## insert phase differential
        #proj -= recon_proj
        #if filtering:
        #    #fbp.ramp_filter(proj)
        #    #fbp.ram_lak_filter(proj, sample_rate)
        #    fbp.shepp_logan_filter(proj, sample_rate)
        #    #fbp.inv_ramp_filter(proj)
        #mu += sigma * proj

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
            fbp.ramp_filter(proj)
#            #fbp.ram_lak_filter(proj, sample_rate)
#            #fbp.shepp_logan_filter(proj, sample_rate)
#            #fbp.inv_ramp_filter(proj)
        mu_bar = mu + sigma * proj
        A.backward(mu_bar, img)
        recon -= tau * img

        # insert support constraint
        recon[elipse] = 0

        tv_denoise(recon, tau / alpha) #, mask=tv_mask)

        A.forward(recon, proj)
        iter_callback(i, recon, mu, proj)
        # insert phase differential
        proj -= data
        if filtering:
            fbp.ramp_filter(proj)
            #fbp.ram_lak_filter(proj, sample_rate)
            #fbp.shepp_logan_filter(proj, sample_rate)
            #fbp.inv_ramp_filter(proj)
        mu += sigma * proj
#        if i == 0:
#            A.forward(recon, proj)
#            proj -= data
#            if filtering:
#                fbp.ramp_filter(proj)
#                #fbp.ram_lak_filter(proj, sample_rate)
#                #fbp.shepp_logan_filter(proj, sample_rate)
#                #fbp.inv_ramp_filter(proj)
#            mu = sigma * proj
#            old_mu = mu * 0
#        A.backward(mu, img)
#        recon -= tau * img
#
#        # insert support constraint
#        recon[elipse] = 0
#
#        tv_denoise(recon, tau / alpha, mask=tv_mask)
#
#        A.forward(recon, proj)
#        iter_callback(i, recon, mu, proj)
#        # insert phase differential
#        proj -= data
#        if filtering:
#            fbp.ramp_filter(proj)
#            #fbp.ram_lak_filter(proj, sample_rate)
#            #fbp.shepp_logan_filter(proj, sample_rate)
#            #fbp.inv_ramp_filter(proj)
#        mu, old_mu = old_mu + 2 * sigma * proj, mu

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
    img = utils.load_rawimage(path)
    if img is None:
        print "invalid file"
        sys.exit(1)

    scale = 0.4
    angle_px = detector_px = width_px = img.shape[1]

    ## define metal material inner image
    #img_mask = numpy.zeros_like(img)
    #mask_c = (10 + (img_mask.shape[0] - 1) / 2.0, 5 +(img_mask.shape[1] - 1) / 2.0)
    #utils.create_elipse_mask(mask_c, 5, 8, img_mask, float("inf"))
    #img += img_mask

    # interior projection
    proj = utils.create_projection(path, interior_scale=scale)
    NoI = img.shape[0]
    NoA, NoD = proj.shape
    interiorA = Projector(NoI, NoA, NoD)
    interiorA.update_detectors_length(int(ceil(NoI*scale)))

    # global projection
    full_proj = utils.create_projection(path, detector_scale=1/scale)
    # truncate global projection
    interior_w = img.shape[0]
    interior_pad = (full_proj.shape[1] - interior_w) / 2  # MEMO: Some cases cause error.
    proj = full_proj[:, interior_pad:interior_pad + interior_w]
    NoI = img.shape[0]
    NoA, NoD = full_proj.shape
    A = Projector(NoI, NoA, NoD)

    ## get metal mask of projection domain
    #proj_mask = numpy.zeros_like(proj, numpy.int)
    #proj_mask[proj == float("inf")] = 1

    # create roi mask
    roi = utils.zero_img(interiorA)
    roi_c = ((roi.shape[0] - 1) / 2., (roi.shape[1] - 1) / 2.)
    roi_r = [roi.shape[0] * scale / 2., roi.shape[1] * scale / 2.]
    utils.create_elipse_mask(roi_c, roi_r[0], roi_r[1], roi)

    # create roi mask
    tvroi = utils.zero_img(interiorA)
    tvroi_c = ((tvroi.shape[0] - 1) / 2., (tvroi.shape[1] - 1) / 2.)
    tvroi_r = [tvroi.shape[0] * scale / 2., tvroi.shape[1] * scale / 2.]
    tvroi_r[0] += 2
    tvroi_r[1] += 2
    utils.create_elipse_mask(tvroi_c, tvroi_r[0], tvroi_r[1], tvroi)

    def callback(i, *argv):
        x = argv[0]
        y = argv[1]
        print i, x[128, 128], numpy.sum(numpy.sqrt(((x - img)*roi)**2))
        if i % 10 == 0:
        #    utils.save_rawimage(x, "app/{}.dat".format(i))
             utils.show_image(x, clim=(1000-100, 1512-100))
        #utils.show_image(y)
        #for j in xrange(len(argv)):
        #    utils.show_image(argv[j])

    recon = img * 0

    # without filter
    #alpha = 0.004
    #alpha = 0.001
    #recon = app_recon(interiorA, proj, alpha, alpha,1000, iter_callback=callback, sample_rate=scale*1.41421356) # 1635498

    # with filter
    #alpha = 0.0465
    alpha = 0.018
    #recon = app_recon(interiorA, proj, alpha, alpha, 1000, iter_callback=callback, sample_rate=scale*1.41421356,
    #                  filtering=True, tv_mask=tvroi) # 201435.93 400 iter
    recon = fullapp_recon(A, proj, alpha, alpha, 1000, iter_callback=callback, sample_rate=1,
                          filtering=True, tv_mask=tvroi)

if __name__ == '__main__':
    main()
