from .sysmat_cpp import sysmat_data_d, sysmat_data
from scipy.sparse import csr_matrix


def sysmat_rd(ny, nx, nt, nr, dr=1.):
    data,indices,indptr = sysmat_data_d(ny, nx, nt, nr, dr)
    return csr_matrix((data, indices, indptr), shape=(nt*nr, ny*nx))


def sysmat_dd(nx, nth, nr, detectors_length):
    return sysmat_data(nx, nth, nr, detectors_length)
