#!/usr/bin/env python2

from projector import Projector
#from differencial import Projector
import fbp
import utils
import numpy
from math import ceil, floor

def grad(img, out_x, out_y):
    out_x[:, :-1] = img[:, 1:] - img[:, :-1]
    out_y[:-1] = img[1:] - img[:-1]

def div_2(img_x, img_y, out):
    out[:, :] = 0
    out[:, :-1] += img_x[:, :-1]
    out[:, 1:] -= img_x[:, :-1]
    out[:-1] += img_y[:-1]
    out[1:] -= img_y[:-1]

def tv_denoise(img, alpha, max_iter=1000, mask=None):
    """
    Smoothing a image with TV denoising method.
    @img : 2D image array
    @alpha : smoothness
    @max_iter : times of iteration
    @mask : areas where peformed denoising (0: disable, 1: enable)
    """
    if mask is None:
        mask = numpy.ones_like(img)
    # parameter
    tol = 0.01
    tau = 1.0 / 4  # 1 / (2 * dimension)

    # matrices
    p_x = numpy.zeros_like(img)
    p_y = numpy.zeros_like(img)
    div_p = numpy.zeros_like(img)
    grad_x = numpy.empty_like(img)
    grad_y = numpy.empty_like(img)
    last_div_p = numpy.zeros_like(img)
    denom = numpy.empty_like(img)
    for i in xrange(max_iter):
        div_2(p_x, p_y, div_p)
        grad(div_p - img / alpha, grad_x, grad_y)
        grad_x *= mask
        grad_y *= mask
        denom[:] = 1
        denom += tau * numpy.sqrt(grad_x**2 + grad_y**2)
        p_x += tau * grad_x
        p_x /= denom
        p_y += tau * grad_y
        p_y /= denom
        if i != 0 and numpy.abs(div_p - last_div_p).max() < tol:
            break
        last_div_p, div_p = div_p, last_div_p
    img -= div_p * alpha

def create_elipse_mask(center, a, b, out, value=1):
    """
    Create a array which contains a elipse.
    Inside the elipse are filled by 1.
    The others are 0.
    @shape : the shape of return array
    @center : the center point of the elipse
    @a : the radius of x-axis
    @b : the radius of y-axis
    """
    x, y = center
    a_2 = a**2.
    b_2 = b**2.
    ab_2 = a_2 * b_2
    out[:, :] = 0
    for xi in xrange(out.shape[1]):
        for yi in xrange(out.shape[0]):
            if a_2 * (y - yi)**2 + b_2 * (x - xi)**2 < ab_2:
                out[yi, xi] = value

def vcoord(shape, point):
    """
    Convert the coordinate systems from 2D-array to the virtual system, taken as arguments.
    @shape : a shape of 2D-array, or length of image sides
    @point : a point in the 2D-array coordinate system
    """
    h, w = shape
    x, y = point
    return (x - w/2., h/2. - y)

def acoord(shape, point):
    """
    Convert the coordinate systems from the virtual system to 2D-array.
    This fuction is inverse transformation of `vcoord`.
    @shape : a shape of 2D-array, or length of image sides
    @point : a point in the virtual coordinate system
    """
    h, w = shape
    x, y = point
    return (x + w/2., h/2. - y)

def fullapp_recon(A, data, sigma, tau, niter, recon=None, mu=None, sample_rate=1,
              iter_callback=lambda *arg: 0):
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

    #interior_w = data.shape[1]
    #interior_pad = (recon_proj.shape[1] - interior_w) / 2  # MEMO: Some cases cause error.
    #recon_proj[:, :interior_pad] = (data[:, 0])[:, None]
    #recon_proj[:, interior_pad + interior_w:] = (data[:, -1])[:, None]
    #recon_proj[:, interior_pad:interior_pad + interior_w] = data
    #recon_proj[:, :interior_pad] *= (numpy.linspace(0, 1, interior_pad))[None, :]
    #recon_proj[:, interior_pad + interior_w:] *= (numpy.linspace(1, 0, interior_pad))[None, :]

    elipse_center = numpy.array(recon.shape) / 2.
    elipse_r = numpy.array(recon.shape) / 2.1
    elipse = img.copy()
    create_elipse_mask(elipse_center, elipse_r[0], elipse_r[1], elipse)
    elipse = elipse < 0.5
    alpha = 0.5

    for i in xrange(niter):
        A.forward(recon, proj)
        utils.show_image(proj)
        proj -= recon_proj
        #fbp.cut_filter(proj)
        #fbp.ramp_filter(proj)
        #fbp.ram_lak_filter(proj, sample_rate)
        #fbp.shepp_logan_filter(proj, sample_rate)
        #fbp.inv_ramp_filter(proj)
        mu_bar = mu + sigma * proj
        A.backward(mu_bar, img)
        recon -= tau * img

        # insert support constraint
        recon[elipse] = 0

        tv_denoise(recon, tau / alpha)

        recon_proj -= tau * mu_bar
        #recon_proj[:, interior_pad:interior_pad + interior_w] = data
        recon_proj[data != float("inf")] = data[data != float("inf")]

        A.forward(recon, proj)
        iter_callback(i, recon, recon_proj, mu)
        # insert phase differential
        proj -= recon_proj
        #fbp.cut_filter(proj)
        #fbp.ramp_filter(proj)
        #fbp.ram_lak_filter(proj, sample_rate)
        #fbp.shepp_logan_filter(proj, sample_rate)
        #fbp.inv_ramp_filter(proj)
        mu += sigma * proj

    return recon

def app_recon(A, data, sigma, tau, niter, recon=None, mu=None, sample_rate=1,
              iter_callback=lambda *arg: 0):
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
    create_elipse_mask(elipse_center, elipse_r[0], elipse_r[1], elipse)
    elipse = elipse < 0.5
    alpha = 0.5

    for i in xrange(niter):
        A.forward(recon, proj)
        proj -= data
        #fbp.ramp_filter(proj)
        #fbp.ram_lak_filter(proj, sample_rate)
        fbp.shepp_logan_filter(proj, sample_rate)
        #fbp.inv_ramp_filter(proj)
        mu_bar = mu + sigma * proj
        A.backward(mu_bar, img)
        recon -= tau * img

        # insert support constraint
        recon[elipse] = 0

        tv_denoise(recon, tau / alpha)

        A.forward(recon, proj)
        iter_callback(i, recon, mu, proj)
        # insert phase differential
        proj -= data
        #fbp.ramp_filter(proj)
        #fbp.ram_lak_filter(proj, sample_rate)
        fbp.shepp_logan_filter(proj, sample_rate)
        #fbp.inv_ramp_filter(proj)
        mu += sigma * proj

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

    scale = 1
    angle_px = detector_px = width_px = img.shape[1]

    mask = numpy.zeros_like(img)
    mask_c = (10 + (mask.shape[0] - 1) / 2.0, 5 +(mask.shape[1] - 1) / 2.0)
    create_elipse_mask(mask_c, 5, 8, mask, float("inf"))
    img += mask

    # interior projection
    interiorA = Projector(width_px, angle_px, detector_px)
    interiorA.update_detectors_length(int(ceil(width_px*scale)))
    proj = utils.empty_proj(interiorA)
    interiorA.forward(img, proj)

    # global projection
    full_detector_px = int(ceil(detector_px/scale))
    full_detector_px += 1 if (full_detector_px - detector_px) % 2 != 0 else 0
    A = Projector(width_px, angle_px, full_detector_px)
    full_proj = utils.zero_proj(A)
    A.forward(img, full_proj)

    # truncate global projection
    interior_w = proj.shape[1]
    interior_pad = (full_proj.shape[1] - interior_w) / 2  # MEMO: Some cases cause error.
    proj = full_proj[:, interior_pad:interior_pad + interior_w]

    # create roi mask
    roi = utils.zero_img(A)
    roi_c = ((roi.shape[0] - 1) / 2., (roi.shape[1] - 1) / 2.)
    roi_r = (roi.shape[0] * scale / 2., roi.shape[1] * scale / 2.)
    create_elipse_mask(roi_c, roi_r[0], roi_r[1], roi)

    def callback(i, *argv):
        x = argv[0]
        y = argv[1]
        print i, x[128, 128], numpy.sum(numpy.sqrt(((x - img)*roi)**2))
        #utils.show_image(x*roi, clim=(1., 1.1))
        utils.show_image(x, clim=(0., 2.))
        #utils.show_image(y)
        #for j in xrange(len(argv)):
        #    utils.show_image(argv[j])

    print img[128, 128]

    #recon = app_recon(interiorA, proj, 0.035, 0.035, 1000, iter_callback=callback, sample_rate=scale*1.41421356)
    #recon = fullapp_recon(A, proj, 0.035, 0.035, 1000, iter_callback=callback, sample_rate=scale*1.41421356)
    recon = fullapp_recon(A, proj, 0.0035, 0.0035, 1000, iter_callback=callback, sample_rate=scale*1.41421356)

if __name__ == '__main__':
    main()
