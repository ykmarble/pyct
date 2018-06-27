#include "sysmat.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Sparse>

Eigen::SparseMatrix<double, Eigen::RowMajor>
buildMatrixWithDistanceMethod(size_t nx, size_t nth, size_t nr, double detectors_length);

namespace py = pybind11;
PYBIND11_MODULE(sysmat_cpp, m)
{
    m.def("sysmat_data_f", &matrix_coeff_rd<float>, "");
    m.def("sysmat_data_d", &matrix_coeff_rd<double>, "");
    m.def("sysmat_data", &buildMatrixWithDistanceMethod, "");
}
