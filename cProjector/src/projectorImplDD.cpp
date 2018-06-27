#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Sparse>

using CSRMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;

CSRMat buildMatrixWithDistanceMethod(
    const size_t nx,
    const size_t nth,
    const size_t nr,
    const double detectors_length)
{
    std::cout << "Generate system matrix with distance-driven method." << std::endl;
    std::vector< Eigen::Triplet<double> > elements;
    elements.reserve(nth * nr * nx * 2);

    auto add_element = [nx, nr, &elements](const int xidx, const int yidx, const int thidx,
                                           const int ridx, const double value) {
        if (0 <= ridx && ridx < nr) {
            const int row = thidx * nr + ridx;
            const int col = yidx * nx + xidx;
            elements.emplace_back(row, col, value);
        }
    };

    // coordinate translation memo
    // - each pixel is 1x1 square
    // - left-top pixel's corner on the pixel array space coord origin
    // - right(left)most boundary of pixel array at cx(-cx) in image space virtual coord
    // - each detector width is derived from detectors_length which means whole detectors array length
    // - detectors array's centor is decided by the same scheme how to decide cx
    // - coefficiant coord is parallel to detectors array and throughs the image space virtual coord origin
    const double cx = nx / 2.;
    const double dth = M_PI / nth;
    const double dr = detectors_length / nr;
    const double cr = nr / 2.;  // devide by dr to calc correct cr
    for (int thi = 0; thi < nth; ++thi) {
        const double th = thi * dth;
        const double costh = std::cos(th);
        const double sinth = std::sin(th);
        for (int yi = 0; yi < nx; ++yi) {
            for (int xi = 0; xi < nx; ++xi) {
                const double pc = (costh * (xi + 0.5 - cx) + sinth * (cx - yi - 0.5)) / dr + cr;

                const double lb = pc - 0.7071067811865476 / dr;
                const double hb = pc + 0.7071067811865476 / dr;
                //const double lb = pc - 0.9 / dr;
                //const double hb = pc + 0.9 / dr;

                const double al = std::floor(lb+1) - lb;
                const int ril = std::floor(lb);
                const double ah = hb - std::floor(hb);
                const int rih = std::floor(hb);

                const double scale = dr / 1.4142135623730951;

                if (ril == rih) {  // projected pixel within a detector's boundary
                    add_element(xi, yi, thi, rih, (hb - lb) * scale);
                } else {  // or acrossing two or more detectors
                    add_element(xi, yi, thi, ril, al * scale);
                    add_element(xi, yi, thi, rih, ah * scale);
                }

                // whole detector within projected pixel boundary
                for (int ri = ril + 1; ri < rih; ++ri) {
                    add_element(xi, yi, thi, ri, 1 * scale);
                }
            }
        }
    }
    CSRMat csr(nth*nr, nx*nx);
    csr.setFromTriplets(elements.begin(), elements.end());
    csr.makeCompressed();
    return csr;
}
