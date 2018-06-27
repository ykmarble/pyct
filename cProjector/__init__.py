from .sysmat_cpp import sysmat_data_d, sysmat_data
from scipy.sparse import csr_matrix
import math


def sysmat_rd(ny, nx, nt, nr, dr=1.):
    data,indices,indptr = sysmat_data_d(ny, nx, nt, nr, dr)
    return csr_matrix((data, indices, indptr), shape=(nt*nr, ny*nx))


def sysmat_dd(nx, nth, nr, detectors_length):
    return sysmat_data(nx, nth, nr, detectors_length)


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
        img /= 4 * self.NoA

    def partial_forward(self, img, proj, th_indexes, r_indexes):
        raise NotImplementedError
        assert self.is_valid_dimension(img, proj)
        if th_indexes is not None:
            assert 0 <= numpy.min(th_indexes) and numpy.max(th_indexes) < self.NoA
        if r_indexes is not None:
            assert 0 <= numpy.min(r_indexes) and numpy.max(r_indexes) < self.NoD
        self._projection(img, proj, False, th_indexes=th_indexes, r_indexes=r_indexes)

    def partial_backward(self, proj, img, th_indexes, r_indexes):
        raise NotImplementedError
        assert self.is_valid_dimension(img, proj)
        if th_indexes is not None:
            assert 0 <= numpy.min(th_indexes) and numpy.max(th_indexes) < self.NoA
        if r_indexes is not None:
            assert 0 <= numpy.min(r_indexes) and numpy.max(r_indexes) < self.NoD
        self._projection(proj, img, True, th_indexes=th_indexes, r_indexes=r_indexes)

    def forward_with_mask(self, img, proj, mask):
        raise NotImplementedError
        assert self.is_valid_dimension(img, proj)
        self._projection(img, proj, False, mask)

    def backward_with_mask(self, proj, img, mask):
        raise NotImplementedError
        assert self.is_valid_dimension(img, proj)
        self._projection(proj, img, True, mask)

    def _fit_sysmat(self):
        if self.sysmat is None:
            self.sysmat = self.sysmat_builder(self.NoI, self.NoA, self.NoD, self.detectors_length)
            self.sysmatT = self.sysmat.transpose()
