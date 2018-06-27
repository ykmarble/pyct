#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Sparse>

using CSRMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;

CSRMat buildMatrixWithJosephMethod(
    const size_t nx,
    const size_t nt,
    const size_t nr,
    const double detectors_length)
{
    std::cout << "Generate system matrix with Joseph method." << std::endl;
    std::vector< Eigen::Triplet<double> > elements;
    elements.reserve(nt * nr * nx * 2);

    auto add_element = [nx, nr, &elements](const int xidx, const int yidx, const int tidx,
                                           const int ridx, const double value) {
        if (0 <= xidx && xidx < nx && 0 <= yidx && yidx < nx) {
            const int row = tidx * nr + ridx;
            const int col = yidx * nx + xidx;
            elements.emplace_back(row, col, value);
        }
    };

    const double dtheta = M_PI / nt;
    const double dr = detectors_length / nr;
    const double cx = (nx - 1) / 2.;
    const double cy = (nx - 1) / 2.;
    const double cr = (nr - 1) / 2.;

    for (int ti = 0; ti < nt; ++ti) {
        const double th = ti * dtheta;
        const double sin_th = std::sin(th);
        const double cos_th = std::cos(th);
        const double inv_sin_th = 1 / sin_th;
        const double inv_cos_th = 1 / cos_th;
        const double sin_cos = sin_th / cos_th;
        const double cos_sin = cos_th / sin_th;
        const bool xdriven = std::abs(sin_th) < std::abs(cos_th);
        const double cp = xdriven ? cx : cy;
        for (int ri = 0; ri < nr; ++ri) {
            const double r = (ri - cr) * dr;
            for (int pi = 0; pi < nx; ++pi) {
                if (xdriven) {
                    const double p = pi - cp;
                    const double ray = -(sin_cos * p + r * inv_cos_th) + cy;
                    const int xi = pi;
                    const int xip = pi;
                    const int yi = (int)std::floor(ray);
                    const int yip = yi + 1;
                    const double a = (yip - ray) * std::abs(inv_cos_th);
                    const double ap = (ray - yi) * std::abs(inv_cos_th);
                    add_element(xi, yi, ti, ri, a);
                    add_element(xip, yip, ti, ri, ap);
                } else {
                    const double p = -(pi - cp);
                    const double ray = cos_sin * p - r * inv_sin_th + cx;
                    const int xi = (int)std::floor(ray);
                    const int xip = xi + 1;
                    const int yi = pi;
                    const int yip = yi;
                    const double a = (xip - ray) * std::abs(inv_sin_th);
                    const double ap = (ray - xi) * std::abs(inv_sin_th);
                    add_element(xi, yi, ti, ri, a);
                    add_element(xip, yip, ti, ri, ap);
                }
            }
        }
    }
    CSRMat csr(nt*nr, nx*nx);
    csr.setFromTriplets(elements.begin(), elements.end());
    csr.makeCompressed();
    return csr;
}
