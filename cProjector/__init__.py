from .sysmat_cpp import sysmat_data_d
from scipy.sparse import csr_matrix


def sysmat(ny, nx, nt, nr, dr=1.):
    data,indices,indptr = sysmat_data_d(ny, nx, nt, nr, dr)
    return csr_matrix((data, indices, indptr), shape=(nt*nr, ny*nx))
