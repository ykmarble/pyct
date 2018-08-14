#!/usr/bin/env python3

from pyct import utils
from pyct import cProjector
from pyct import ctfilter
import numpy
import sys


imgpath = sys.argv[1]
reconpath = sys.argv[2]

orig = utils.load_rawimage(imgpath)
recon = utils.load_rawimage(reconpath)

HU_lim = [0.05, 0.2]
scale = 0.8

NoI = orig.shape[0]
NoA = int(NoI * 1.2)
NoD = int(NoI * 1.)
center_x = (NoI - 1) / 2. -20 # lung
center_y = (NoI - 1) / 2.  # lung

# global projection
full_NoA = NoA
full_NoD = int(round(NoD*1/scale))
assert NoD == int(round(full_NoD * scale))
full_A = cProjector.Projector(NoI, full_NoA, full_NoD)
full_A.update_detectors_length(NoI+50)
full_A.update_center_x(center_x)
full_A.update_center_y(center_y)
full_A.sysmat_builder = cProjector.sysmat_joseph

# calculate roi and its sinogram
xmask = utils.zero_img(full_A)
utils.create_elipse_mask((full_A.center_x, full_A.center_y), NoI/2.*scale, NoI/2.*scale, xmask)
ymask = utils.zero_proj(full_A)
full_A.forward(xmask, ymask)
ymask[ymask != 0] = 1

# create truncated sinogram
orig_proj = utils.zero_proj(full_A)
recon_proj = utils.zero_proj(full_A)
full_A.forward(orig, orig_proj)
full_A.forward(recon, recon_proj)
recon_proj[ymask == 1] = orig_proj[ymask == 1]
utils.show_image(recon_proj)

final_img = utils.zero_img(full_A)
ctfilter.shepp_logan_filter(recon_proj)
full_A.backward(recon_proj, final_img)
print(numpy.min(final_img), numpy.max(final_img))
utils.show_image(final_img*xmask, HU_lim)
