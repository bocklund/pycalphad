#ifndef PYCALPHAD_NUMPY_WRAPPER_HPP
#define PYCALPHAD_NUMPY_WRAPPER_HPP

#include "pycalphad_types.hpp"
#include <stdexcept>

namespace pycalphad {
namespace numpy {

/**
 * @brief Wrapper for NumPy array data access from C++
 *
 * This provides a safe interface to access NumPy array data from C++ code
 * that is called from Cython. The Cython layer is responsible for:
 * - Ensuring the array is contiguous (or properly strided)
 * - Ensuring proper memory layout (C or Fortran order)
 * - Keeping the Python object alive during C++ execution
 */

/**
 * @brief Create a 1D array view from pointer and size
 * @param data Pointer to array data
 * @param size Number of elements
 * @return ArrayView1D<T> Non-owning view
 */
template<typename T>
inline ArrayView1D<T> make_array_view_1d(T* data, size_type size) {
    if (!data && size > 0) {
        throw std::invalid_argument("Null pointer with non-zero size");
    }
    return ArrayView1D<T>(data, size);
}

/**
 * @brief Create a 2D array view from pointer and dimensions
 * @param data Pointer to array data
 * @param rows Number of rows
 * @param cols Number of columns
 * @param stride Stride between rows (cols for C-order, 1 for Fortran-order)
 * @return ArrayView2D<T> Non-owning view
 */
template<typename T>
inline ArrayView2D<T> make_array_view_2d(T* data, size_type rows, size_type cols,
                                          size_type stride) {
    if (!data && (rows * cols) > 0) {
        throw std::invalid_argument("Null pointer with non-zero dimensions");
    }
    return ArrayView2D<T>(data, rows, cols, stride);
}

/**
 * @brief Create a 2D C-ordered array view
 */
template<typename T>
inline ArrayView2D<T> make_array_view_2d_c(T* data, size_type rows, size_type cols) {
    return make_array_view_2d(data, rows, cols, cols);
}

/**
 * @brief Create a 2D Fortran-ordered array view
 */
template<typename T>
inline ArrayView2D<T> make_array_view_2d_f(T* data, size_type rows, size_type cols) {
    return make_array_view_2d(data, rows, cols, 1);
}

/**
 * @brief Bounds checking for debug builds
 */
#ifdef PYCALPHAD_DEBUG
    template<typename T>
    inline void check_bounds_1d(const ArrayView1D<T>& view, size_type idx) {
        if (idx >= view.size) {
            throw std::out_of_range("Array index out of bounds");
        }
    }

    template<typename T>
    inline void check_bounds_2d(const ArrayView2D<T>& view, size_type i, size_type j) {
        if (i >= view.rows || j >= view.cols) {
            throw std::out_of_range("Array indices out of bounds");
        }
    }
#else
    template<typename T>
    inline void check_bounds_1d(const ArrayView1D<T>&, size_type) {}

    template<typename T>
    inline void check_bounds_2d(const ArrayView2D<T>&, size_type, size_type) {}
#endif

} // namespace numpy
} // namespace pycalphad

#endif // PYCALPHAD_NUMPY_WRAPPER_HPP
