#!/usr/bin/env python2

import projector
import numpy

class Projector(projector.Projector):
    def __init__(self, length_of_image_side, num_of_angles, num_of_detectors):
        """
        Notice: `num_of_detectors` means a shape of *differential* projection data.
        """
        super(self.__class__, self).__init__(length_of_image_side, num_of_angles, num_of_detectors+1)
        self.last_proj = numpy.empty((num_of_angles, num_of_detectors+1))

    def get_image_shape(self):
        return (self.NoI, self.NoI)

    def get_projector_shape(self):
        return (self.NoA, self.NoD-1)

    def forward(self, img, proj):
        super(self.__class__, self).forward(img, self.last_proj)
        self.rdiff(self.last_proj, proj)

    def backward(self, proj, img):
        self.t_rdiff(proj, self.last_proj)
        super(self.__class__, self).backward(self.last_proj, img)

    def partial_forward(self, img, proj, th_indexes, r_indexes):
        raise NotImplementedError("differncial.Projector.partial_forward()")

    def partial_backward(self, proj, img, th_indexes, r_indexes):
        raise NotImplementedError("differncial.Projector.partial_backward()")

    def rdiff(self, proj, diff_proj):
        diff_proj[:] = 0.5 * (proj[:, :-1] - proj[:, 1:])

    def t_rdiff(self, diff_proj, proj):
        proj[:] = 0
        proj[:, :-1] += 0.5 * diff_proj
        proj[:, 1:] -= 0.5 * diff_proj

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
    detector_px -= 1
    A = Projector(width_px, angle_px, detector_px)
    proj = utils.empty_proj(A)
    recon = utils.empty_img(A)
    A.forward(img, proj)
    A.backward(proj, recon)
    utils.show_image(proj)
    utils.show_image(recon)

if __name__ == '__main__':
    main()
