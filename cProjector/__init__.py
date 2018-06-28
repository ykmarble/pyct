from .sysmat_cpp import sysmat_data_joseph, sysmat_data_dd
import numpy
import math


def sysmat_joseph(nx, nth, nr, detectors_length):
    return sysmat_data_joseph(nx, nth, nr, detectors_length)


def sysmat_dd(nx, nth, nr, detectors_length):
    return sysmat_data_dd(nx, nth, nr, detectors_length)


class Projector(object):
    def __init__(self, length_of_image_side, num_of_angles, num_of_detectors):
        self.NoI = length_of_image_side  # number of pixels of image side
        self.NoD = num_of_detectors      # number of detectors
        self.NoA = num_of_angles         # number of projection angles

        self.image_origin = self.NoI / 2. + 0.5     # num_of_image_sides / 2
        self.detectors_origin = self.NoD / 2. + 0.5 # num_of_detectors / 2

        # variables used when computing projection
        # calculated from above variables automatically
        # dr                : in other words, the length of each detector
        # dtheta
        # center_x          : derived from image_origin and x_offset
        # center_y          : derived from image_origin and y_offset
        # detectors_center  : derived from detectors_origin and detectors_offset
        self.x_offset = 0
        self.center_x = self.x_offset + self.image_origin
        self.y_offset = 0
        self.center_y = -self.y_offset + self.image_origin
        self.detectors_offset = 0
        self.detectors_center = self.detectors_offset + self.detectors_origin
        self.update_detectors_length(self.NoI)
        self.dtheta = math.pi / self.NoA

        self.sysmat = None
        self.sysmatT = None
        #self.sysmat_builder = sysmat_joseph
        self.sysmat_builder = sysmat_dd

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
        self.sysmat = None
        self.sysmatT = None

    def update_center_x(self, x):
        raise NotImplementedError

    def update_center_y(self, y):
        raise NotImplementedError

    def is_valid_dimension(self, img, proj):
        return img.shape[0] == img.shape[1] \
          and img.shape[0] == self.NoI \
          and proj.shape[0] == self.NoA \
          and proj.shape[1] == self.NoD

    def forward(self, img, proj):
        assert self.is_valid_dimension(img, proj)
        self._fit_sysmat()
        proj[:] = (self.sysmat * img.reshape(-1)).reshape(self.NoA, self.NoD)

    def backward(self, proj, img):
        assert self.is_valid_dimension(img, proj)
        self._fit_sysmat()
        img[:] = (self.sysmatT * proj.reshape(-1)).reshape(self.NoI, self.NoI)
        img /= 2 * self.NoA

    def partial_forward(self, img, proj, th_indexes, r_indexes):
        assert self.is_valid_dimension(img, proj)
        if th_indexes is not None:
            assert 0 <= numpy.min(th_indexes) and numpy.max(th_indexes) < self.NoA
        if r_indexes is not None:
            raise NotImplementedError

        if th_indexes is None:
            th_indexes = range(self.NoA)
        r_indexes = range(self.NoD)

        self._fit_sysmat()
        rows = self._row_array_from_th_r(th_indexes, r_indexes)  # 25 us (including the avobe procs)
        #ps = self.sysmat[rows]  # 36600 us
        #pp = proj[th_indexes]   # 6 us
        #pp = (ps * img.reshape(-1)).reshape(th_indexes.size, self.NoD)  # 4215 us
        proj[th_indexes] = (self.sysmat[rows] * img.reshape(-1)).reshape(th_indexes.size, self.NoD)

    def partial_backward(self, proj, img, th_indexes, r_indexes):
        assert self.is_valid_dimension(img, proj)
        if th_indexes is not None:
            assert 0 <= numpy.min(th_indexes) and numpy.max(th_indexes) < self.NoA
        if r_indexes is not None:
            raise NotImplementedError

        if th_indexes is None:
            th_indexes = range(self.NoA)
        r_indexes = range(self.NoD)

        self._fit_sysmat()
        rows = self._row_array_from_th_r(th_indexes, r_indexes)  # 25 us (including the avobe procs)
        #ps = self.sysmatT[:, rows]   # 36600 us
        #pp = proj.reshape(-1)[rows]  # 13 us
        #img[:] = (ps * pp).reshape(self.NoI, self.NoI)  # 5537 us
        img[:] = (self.sysmatT[:, rows] * proj.reshape(-1)[rows]).reshape(self.NoI, self.NoI)
        img /= 2 * self.NoA

    def _fit_sysmat(self):
        if self.sysmat is None:
            self.sysmat = self.sysmat_builder(self.NoI, self.NoA, self.NoD, self.detectors_length)
            self.sysmatT = self.sysmat.transpose()

    def _row_array_from_th_r(self, thidxs, ridxs):
        return ((thidxs * self.NoD)[None].T + ridxs).reshape(-1)
