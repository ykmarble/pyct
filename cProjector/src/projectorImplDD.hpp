#include <iostream>
#include <cmath>
#include <vector>
#include <thread>
#include <Eigen/Sparse>

namespace {
    template <typename T>
    using CSRMat = Eigen::SparseMatrix<T, Eigen::RowMajor>;

    template <typename T>
    using CoeffTriplets = std::vector< Eigen::Triplet<T> >;
}

template <typename T>
CoeffTriplets<T> inner(
    const size_t nx,
    const size_t nth,
    const size_t nr,
    const double cx,
    const double cy,
    const double detectors_length)
{
    auto add_element = [nx, nr](const int xidx, const int yidx, const int thidx, const int ridx,
                                const double value, CoeffTriplets<T>& elements) {
        if (0 <= ridx && ridx < nr) {
            const int row = thidx * nr + ridx;
            const int col = yidx * nx + xidx;
            elements.emplace_back(row, col, static_cast<T>(value));
        }
    };

    // coordinate translation memo
    // - each pixel is 1x1 square
    // - left-top pixel's corner on the pixel array space coord origin
    // - right(left)most boundary of pixel array at cx(-cx) in image space virtual coord
    // - each detector width is derived from detectors_length which means whole detectors array length
    // - detectors array's centor is decided by the same scheme how to decide cx
    // - coefficiant coord is parallel to detectors array and throughs the image space virtual coord origin
    const double dth = M_PI / nth;
    const double dr = detectors_length / nr;
    const double cr = nr / 2.;  // devide by dr to calc correct cr
    const double blob_r = 0.7071067811865476;
    //const double blob_r = 0.9;
    const double scale = 1 / blob_r / 2;

    auto calc_coeff_yi = [nx, nth, cx, cy, dth, dr, cr, blob_r, scale, add_element](const int yi, CoeffTriplets<T>& out) {
        for (int xi = 0; xi < nx; ++xi) {
            for (int thi = 0; thi < nth; ++thi) {
                const double th = thi * dth + dth/2.;
                const double costh = std::cos(th);
                const double sinth = std::sin(th);
                const double pc = (costh * (xi - cx) + sinth * (cy - yi)) / dr + cr;

                const double lb = pc - blob_r / dr;
                const double hb = pc + blob_r / dr;

                const double al = std::floor(lb+1) - lb;
                const int ril = std::floor(lb);
                const double ah = hb - std::floor(hb);
                const int rih = std::floor(hb);

                if (ril == rih) {  // projected pixel within a detector's boundary
                    add_element(xi, yi, thi, ril, (hb - lb) * scale, out);
                } else {  // or acrossing two or more detectors
                    add_element(xi, yi, thi, ril, al * scale, out);
                    add_element(xi, yi, thi, rih, ah * scale, out);
                }

                // whole detector within projected pixel boundary
                for (int ri = ril + 1; ri < rih; ++ri) {
                    add_element(xi, yi, thi, ri, 1 * scale, out);
                }
            }
        }
    };

    const size_t nthread = std::thread::hardware_concurrency();
    const size_t stride = (nx - 1) / nthread + 1;
    auto work = [calc_coeff_yi](int start, int end, CoeffTriplets<T> &elements) {
                    for (int yi = start; yi < end; ++yi)
                        calc_coeff_yi(yi, elements);
                    elements.shrink_to_fit();
                };

    std::vector< CoeffTriplets<T> > workspace(nthread);
    std::vector<std::thread> workers;
    for (int i = 0; i < nthread; ++i) {
        const int start = i * stride;
        const int end = std::min(start + stride, nx);
        auto& w = workspace[i];
        w.reserve(stride * nx * nth / scale * 10);
        workers.push_back(std::thread(work, start, end, std::ref(w)));
    }

    for (auto& w : workers)
        w.join();

    auto nonzero_elements = 0;
    for (const auto& w: workspace)
        nonzero_elements += w.size();

    CoeffTriplets<T> elements;
    elements.reserve(nonzero_elements);

    for (auto& w : workspace) {
        std::move(w.begin(), w.end(), std::back_inserter(elements));
    }

    return elements;
}

template <typename T>
CSRMat<T> buildMatrixWithDistanceMethod(
    const size_t nx,
    const size_t nth,
    const size_t nr,
    const double cx,
    const double cy,
    const double detectors_length)
{
    std::cout << "0" << std::endl;
    auto elements = inner<T>(nx, nth, nr, cx, cy, detectors_length);
    std::cout << "1" << std::endl;
    const size_t nrows = nth * nr;
    const size_t ncols = nx * nx;
    CSRMat<T> csr(nrows, ncols);
    csr.setFromTriplets(elements.begin(), elements.end());
    std::cout << "2" << std::endl;

    return csr;
}
