#!/usr/bin/env python2

import cProjector
import math
import numpy

class Projector(cProjector.Projector):
    def __init__(self, length_of_image_side, num_of_angles, num_of_detectors):
        """
        Notice: `num_of_detectors` means a shape of *differential* projection data.
        """
        super(self.__class__, self).__init__(length_of_image_side, num_of_angles, num_of_detectors)
        self.last_proj = numpy.empty((num_of_angles, num_of_detectors))

    def get_image_shape(self):
        return (self.NoI, self.NoI)

    def get_projector_shape(self):
        return (self.NoA, self.NoD)

    def forward(self, img, proj):
        super(self.__class__, self).forward(img, self.last_proj)
        self.rdiff(self.last_proj, proj)

    def backward(self, proj, img):
        self.t_rdiff(proj, self.last_proj)
        super(self.__class__, self).backward(self.last_proj, img)

    def partial_forward(self, img, proj, th_indexes):
        super(self.__class__, self).partial_forward(img, self.last_proj, th_indexes)
        self.rdiff(self.last_proj, proj)

    def partial_backward(self, proj, img, th_indexes):
        self.t_rdiff(proj, self.last_proj)
        super(self.__class__, self).partial_backward(self.last_proj, img, th_indexes)

    def rdiff(self, proj, diff_proj):
        diff_proj[:, 1:-1] = proj[:, 2:] - proj[:, :-2]
        diff_proj[:, 0] = 0
        diff_proj[:, -1] = 0
        diff_proj /= 2.
        #diff_proj[:, :-1] = proj[:, 1:] - proj[:, :-1]
        #diff_proj[:, -1] = 0

    def t_rdiff(self, diff_proj, proj):
        proj[:, 1:-1] = diff_proj[:, :-2] - diff_proj[:, 2:]
        proj[:, 0] = -diff_proj[:, 0]
        proj[:, -1] = diff_proj[:, -1]
        proj /= 2.

def rotate(proj, angle):
    """
    Calculate `angle` degree(radian) rotated projection data whom [0, pi] degree data.
    """
    NoA, NoD = proj.shape
    dth = math.pi / NoA
    for i in xrange(NoA):
        if dth * i > angle:
            break
        proj[i] *= -1


def main():
    import sys
    import os.path
    import utils
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

    angle_px = detector_px = width_px = img.shape[1]
    A = Projector(width_px, angle_px, detector_px)
    accA = projector.Projector(width_px, angle_px, detector_px)
    proj = utils.empty_proj(A)
    recon = utils.empty_img(A)
    dbp = utils.empty_img(accA)
    A.forward(img, proj)
    orig_proj = proj
    for i in xrange(361):
        proj = orig_proj.copy()
        rotate(proj, math.pi/360*1*i)
        accA.backward(proj, dbp)
        recon += dbp
        utils.show_image(proj)
        print i

if __name__ == '__main__':
    main()
