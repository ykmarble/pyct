from .sysmat_cpp import sysmat_data_joseph, sysmat_data_dd
import numpy
import math


def sysmat_joseph(nx, nth, nr, cx, cy, detectors_length):
    return sysmat_data_joseph(nx, nth, nr, cx, cy, detectors_length)


def sysmat_dd(nx, nth, nr, cx, cy, detectors_length):
    return sysmat_data_dd(nx, nth, nr, cx, cy, detectors_length)


class Projector(object):
    def __init__(self, length_of_image_side, num_of_angles, num_of_detectors):
        self.NoI = length_of_image_side  # number of pixels of image side
        self.NoD = num_of_detectors      # number of detectors
        self.NoA = num_of_angles         # number of projection angles

        self.detectors_origin = self.NoD / 2. + 0.5 # num_of_detectors / 2

        # variables used when computing projection
        # calculated from above variables automatically
        # dr                : in other words, the length of each detector
        # dtheta
        # center_x          : derived from image_origin and x_offset
        # center_y          : derived from image_origin and y_offset
        # detectors_center  : derived from detectors_origin and detectors_offset
        self.center_x = (self.NoI - 1)/2.
        self.center_y = (self.NoI - 1)/2.
        self.x_offset = 0
        self.y_offset = 0
        self.detectors_offset = 0
        self.detectors_center = self.detectors_offset + self.detectors_origin
        self.update_detectors_length(self.NoI)
        self.dtheta = math.pi / self.NoA
        self.sig_scale = numpy.sqrt(1. / (2. * self.NoA))

        self.sysmat_need_update = True
        self.sysmat = None
        self.sysmatT = None
        self.partial_sysmat = {}
        self.partial_sysmatT = {}
        self.sysmat_builder = sysmat_joseph
        #self.sysmat_builder = sysmat_dd

    def get_image_shape(self):
        return (self.NoI, self.NoI)

    def get_projector_shape(self):
        return (self.NoA, self.NoD)

    def convidx_img2r(self, yi, xi, ti):
        th = ti * self.dtheta
        x = xi - self.center_x
        y = self.center_y - yi
        r = (x * math.sin(th) + y * math.cos(th)) / self.dr
        r += self.detectors_center
        return int(round(r))

    def update_x_offset(self, offset):
        raise NotImplementedError

    def update_y_offset(self, offset):
        raise NotImplementedError

    def update_detectors_offset(self, offset):
        raise NotImplementedError

    def update_angular_range(self, max_angle):
        raise NotImplementedError

    def update_detectors_length(self, length):
        self.detectors_length = float(length)
        self.dr = self.detectors_length / self.NoD
        self.sysmat_need_update = True

    def update_center_x(self, x):
        self.center_x = float(x)
        self.sysmat_need_update = True

    def update_center_y(self, y):
        self.center_y = float(y)
        self.sysmat_need_update = True

    def is_valid_dimension(self, img, proj):
        return img.shape[0] == img.shape[1] \
          and img.shape[0] == self.NoI \
          and proj.shape[0] == self.NoA \
          and proj.shape[1] == self.NoD

    def forward(self, img, proj):
        assert self.is_valid_dimension(img, proj)
        self._fit_sysmat()
        proj[:] = (self.sysmat * img.reshape(-1)).reshape(self.NoA, self.NoD)
        #proj *= self.sig_scale

    def backward(self, proj, img):
        assert self.is_valid_dimension(img, proj)
        self._fit_sysmat()
        img[:] = (self.sysmatT * proj.reshape(-1)).reshape(self.NoI, self.NoI)
        #img *= self.sig_scale
        img *= self.sig_scale * self.sig_scale

    def partial_forward(self, img, proj, th_indexes):
        assert self.is_valid_dimension(img, proj)
        assert th_indexes is not None and 0 <= numpy.min(th_indexes) and numpy.max(th_indexes) < self.NoA

        rows = self._row_array_from_th(th_indexes)
        part_key = th_indexes.tobytes()

        self._fit_sysmat()
        if not part_key in self.partial_sysmat:
            self.partial_sysmat[part_key] = self.sysmat[rows]
        try:
            proj[th_indexes] = (self.partial_sysmat[part_key] * img.reshape(-1)).reshape(th_indexes.size, self.NoD)
        except:
            import pdb; pdb.set_trace()

    def partial_backward(self, proj, img, th_indexes):
        assert self.is_valid_dimension(img, proj)
        assert th_indexes is not None and 0 <= numpy.min(th_indexes) and numpy.max(th_indexes) < self.NoA

        rows = self._row_array_from_th(th_indexes)
        part_key = th_indexes.tobytes()

        self._fit_sysmat()
        if not part_key in self.partial_sysmatT:
            self.partial_sysmatT[part_key] = self.sysmatT[:, rows]

        img[:] = (self.partial_sysmatT[part_key] * proj.reshape(-1)[rows]).reshape(self.NoI, self.NoI)
        img /= 2 * self.NoA

    def _fit_sysmat(self):
        if self.sysmat_need_update:
            self.sysmat_need_update = False
            self.sysmat = self.sysmat_builder(self.NoI, self.NoA, self.NoD, self.center_x, self.center_y, self.detectors_length)
            self.sysmatT = self.sysmat.transpose()
            self.partial_sysmat = {}
            self.partial_sysmatT = {}

    def _row_array_from_th(self, thidxs):
        return ((thidxs * self.NoD)[None].T + numpy.arange(self.NoD)).reshape(-1)
