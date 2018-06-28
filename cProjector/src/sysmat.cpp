#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Sparse>

#include "projectorImplJoseph.hpp"
#include "projectorImplDD.hpp"

namespace {
    template <typename T>
    using CSRMat = Eigen::SparseMatrix<T, Eigen::RowMajor>;
}

namespace py = pybind11;
PYBIND11_MODULE(sysmat_cpp, m)
{
    m.def("sysmat_data_joseph", &buildMatrixWithJosephMethod<float>, "");
    m.def("sysmat_data_dd", &buildMatrixWithDistanceMethod<float>, "");
}
