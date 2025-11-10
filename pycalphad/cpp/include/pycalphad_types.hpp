#ifndef PYCALPHAD_TYPES_HPP
#define PYCALPHAD_TYPES_HPP

#include <cstddef>
#include <cstdint>
#include <vector>
#include <array>
#include <memory>
#include <string>

namespace pycalphad {

// Common type aliases
using index_t = std::ptrdiff_t;
using size_type = std::size_t;

// Tolerance constants
constexpr double COMP_DIFFERENCE_TOL = 1e-8;
constexpr double MIN_SITE_FRACTION = 1e-12;
constexpr double MIN_PHASE_FRACTION = 1e-6;

// Mathematical constants
constexpr double TINY = 1e-30;
constexpr double HUGE_VAL_DBL = 1e300;

// Array view types (non-owning)
template<typename T>
struct ArrayView1D {
    T* data;
    size_type size;

    ArrayView1D() : data(nullptr), size(0) {}
    ArrayView1D(T* ptr, size_type n) : data(ptr), size(n) {}

    T& operator[](size_type i) { return data[i]; }
    const T& operator[](size_type i) const { return data[i]; }

    T* begin() { return data; }
    T* end() { return data + size; }
    const T* begin() const { return data; }
    const T* end() const { return data + size; }
};

template<typename T>
struct ArrayView2D {
    T* data;
    size_type rows;
    size_type cols;
    size_type stride;  // for handling Fortran/C ordering

    ArrayView2D() : data(nullptr), rows(0), cols(0), stride(0) {}
    ArrayView2D(T* ptr, size_type r, size_type c, size_type s)
        : data(ptr), rows(r), cols(c), stride(s) {}

    T& operator()(size_type i, size_type j) {
        return data[i * stride + j];
    }

    const T& operator()(size_type i, size_type j) const {
        return data[i * stride + j];
    }
};

// Result types for algorithms
enum class SolverStatus {
    SUCCESS,
    NOT_CONVERGED,
    SINGULAR_MATRIX,
    PHASE_RULE_VIOLATION,
    INVALID_INPUT
};

struct SolverResult {
    SolverStatus status;
    int iterations;
    double residual;
    std::string message;

    SolverResult()
        : status(SolverStatus::NOT_CONVERGED),
          iterations(0),
          residual(HUGE_VAL_DBL),
          message("") {}

    bool converged() const { return status == SolverStatus::SUCCESS; }
};

} // namespace pycalphad

#endif // PYCALPHAD_TYPES_HPP
