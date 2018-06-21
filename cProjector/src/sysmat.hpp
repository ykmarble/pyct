#pragma once
#define _USE_MATH_DEFINES
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

template <class Vector>
pybind11::array_t<typename Vector::value_type> pyarray_from_vector(Vector& v)
{
    auto tmp = new Vector();
    tmp->swap(v);
    auto capsule = pybind11::capsule(
      tmp, [](void* p) { delete reinterpret_cast<Vector*>(p); });
    return pybind11::array_t<typename Vector::value_type>(tmp->size(),
                                                          tmp->data(), capsule);
}

template <class datatype>
std::tuple<pybind11::array_t<datatype>,  // data
           pybind11::array_t<ssize_t>,   // indices
           pybind11::array_t<ssize_t>>   // indptr
matrix_coeff(const int ny, const int nx, const int nt, const int nr, const double dr)
{
    static_assert(std::is_floating_point<datatype>::value,
                  "datatype must be floating point");
    std::vector<datatype> data;
    std::vector<ssize_t> indices;
    std::vector<ssize_t> indptr(nt * nr + 1);

    const auto guess_num = std::max(ny, nx) * 2;
    data.reserve(guess_num);
    indices.reserve(guess_num);
    indptr[0] = 0;

    auto add_element = [ny, nx, &data, &indices](const int xidx, const int yidx,
                                             const auto value,
                                             const ssize_t nrow) {
        if (xidx >= 0 && xidx < nx && yidx >= 0 && yidx < ny) {
            data.push_back(value);
            indices.push_back(xidx + yidx * nx);
            return nrow + 1;
        };
        return nrow;
    };

    ssize_t nrow = 0, nrow_prev = 0;
    const double dtheta = M_PI / nt;
    const double cx = (nx - 1) / 2, cy = (ny - 1) / 2, cr = (nr - 1) / 2;
    for (int ti = 0; ti < nt; ++ti) {
        const double th = ti * dtheta;
        const double sin_th = std::sin(th);
        const double cos_th = std::cos(th);
        const double inv_sin_th = 1 / sin_th;
        const double inv_cos_th = 1 / cos_th;
        const double sin_cos = sin_th / cos_th;
        const double cos_sin = cos_th / sin_th;
        const bool xdriven = std::abs(sin_th) < std::abs(cos_th);
        const int np = xdriven ? nx : ny;
        const double cp = xdriven ? cx : cy;
        for (int ri = 0; ri < nr; ++ri) {
            const double r = (ri - cr) * dr;
            for (int pi = 0; pi < np; ++pi) {
                if (xdriven) {
                    const double p = pi - cp;
                    const double ray = -(sin_cos * p + r * inv_cos_th) + cy;
                    const int xi = pi;
                    const int xip = pi + 1;
                    const int yi = (int)std::floor(ray);
                    const int yip = yi + 1;
                    const double a = (yip - ray) * std::abs(inv_cos_th);
                    const double ap = (ray - yi) * std::abs(inv_cos_th);
                    nrow = add_element(xi, yi, a, nrow);
                    nrow = add_element(xip, yip, ap, nrow);
                } else {
                    const double p = -(pi - cp);
                    const double ray = cos_sin * p - r * inv_sin_th + cx;
                    const int xi = (int)std::floor(ray);
                    const int xip = xi + 1;
                    const int yi = pi;
                    const int yip = yi + 1;
                    const double a = (xip - ray) * std::abs(inv_sin_th);
                    const double ap = (ray - xi) * std::abs(inv_sin_th);
                    nrow = add_element(xi, yi, a, nrow);
                    nrow = add_element(xip, yip, ap, nrow);
                }
            }
            indptr[ti * nr + ri + 1] =
              indptr[ti * nr + ri] + (nrow - nrow_prev);
            nrow_prev = nrow;
        }
    }

    data.shrink_to_fit();
    indices.shrink_to_fit();
    indptr.shrink_to_fit();
    return std::make_tuple(pyarray_from_vector(data),
                           pyarray_from_vector(indices),
                           pyarray_from_vector(indptr));
}
