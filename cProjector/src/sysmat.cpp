#include "sysmat.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;
PYBIND11_MODULE(sysmat_cpp, m)
{
    m.def("sysmat_data_f", &matrix_coeff<float>, "");
    m.def("sysmat_data_d", &matrix_coeff<double>, "");
}
