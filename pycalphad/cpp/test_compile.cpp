// Simple test to verify C++ headers compile correctly
#include "include/pycalphad_types.hpp"
#include "utils/numpy_wrapper.hpp"
#include "utils/memory_utils.hpp"

#include <iostream>

int main() {
    using namespace pycalphad;

    // Test type definitions
    std::cout << "Testing pycalphad C++ infrastructure..." << std::endl;

    // Test ArrayView1D
    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto view = numpy::make_array_view_1d(data, 5);
    std::cout << "ArrayView1D size: " << view.size << std::endl;
    std::cout << "First element: " << view[0] << std::endl;

    // Test ArrayView2D
    double data2d[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    auto view2d = numpy::make_array_view_2d_c(data2d, 2, 3);
    std::cout << "ArrayView2D dimensions: " << view2d.rows << "x" << view2d.cols << std::endl;
    std::cout << "Element (0,0): " << view2d(0, 0) << std::endl;

    // Test MallocPtr
    auto ptr = memory::MallocPtr<double>(10);
    ptr.zero();
    std::cout << "MallocPtr size: " << ptr.size() << std::endl;

    // Test constants
    std::cout << "COMP_DIFFERENCE_TOL: " << COMP_DIFFERENCE_TOL << std::endl;
    std::cout << "MIN_SITE_FRACTION: " << MIN_SITE_FRACTION << std::endl;

    // Test SolverResult
    SolverResult result;
    result.status = SolverStatus::SUCCESS;
    result.iterations = 10;
    result.residual = 1e-9;
    std::cout << "Solver converged: " << (result.converged() ? "yes" : "no") << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
