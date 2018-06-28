#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Sparse>

using CSRMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using CSCMat = Eigen::SparseMatrix<double, Eigen::ColMajor>;

CSRMat buildMatrixWithJosephMethod(size_t nx, size_t nth, size_t nr, double detectors_length);
CSCMat buildMatrixWithDistanceMethod(size_t nx, size_t nth, size_t nr, double detectors_length);

namespace py = pybind11;
PYBIND11_MODULE(sysmat_cpp, m)
{
    m.def("sysmat_data_joseph", &buildMatrixWithJosephMethod, "");
    m.def("sysmat_data_dd", &buildMatrixWithDistanceMethod, "");
}
