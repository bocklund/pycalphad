#ifndef PYCALPHAD_LAPACK_WRAPPER_HPP
#define PYCALPHAD_LAPACK_WRAPPER_HPP

#include "pycalphad_types.hpp"
#include <vector>
#include <cmath>

// Forward declare LAPACK functions (will be linked from LAPACK library)
extern "C" {
    // Linear system solver: AX = B
    void dgesv_(const int* n, const int* nrhs, double* a, const int* lda,
                int* ipiv, double* b, const int* ldb, int* info);

    // Least squares solver
    void dgelsd_(const int* m, const int* n, const int* nrhs, double* a,
                 const int* lda, double* b, const int* ldb, double* s,
                 const double* rcond, int* rank, double* work, const int* lwork,
                 int* iwork, int* info);

    // Matrix inversion (requires dgetrf + dgetri)
    void dgetrf_(const int* m, const int* n, double* a, const int* lda,
                 int* ipiv, int* info);

    void dgetri_(const int* n, double* a, const int* lda, int* ipiv,
                 double* work, const int* lwork, int* info);
}

namespace pycalphad {
namespace lapack {

/**
 * @brief Solve linear system AX = B using LAPACK dgesv
 * @param n Dimension of matrix A (n x n)
 * @param a Matrix A (will be overwritten with LU factorization)
 * @param b Right-hand side vector(s) (will be overwritten with solution)
 * @param nrhs Number of right-hand sides
 * @return true if successful, false if matrix is singular
 */
inline bool solve_linear_system(int n, double* a, double* b, int nrhs = 1) {
    std::vector<int> ipiv(n);
    int info = 0;
    dgesv_(&n, &nrhs, a, &n, ipiv.data(), b, &n, &info);

    // If info != 0, matrix is singular or invalid input
    // For pycalphad compatibility, set solution to sentinel value
    if (info != 0) {
        for (int i = 0; i < n * nrhs; ++i) {
            b[i] = -1e19;
        }
        return false;
    }
    return true;
}

/**
 * @brief Solve least squares problem using LAPACK dgelsd
 * @param m Number of rows in A
 * @param n Number of columns in A
 * @param a Matrix A (m x n, will be modified)
 * @param b Right-hand side vector (length m, will be overwritten with solution)
 * @param rcond Relative condition number threshold
 * @return true if successful
 */
inline bool solve_least_squares(int m, int n, double* a, double* b,
                                 double rcond = 1e-16) {
    // Check for NaN in input
    bool has_nan = false;
    for (int i = 0; i < m * n; ++i) {
        if (std::isnan(a[i])) {
            has_nan = true;
            break;
        }
    }

    if (has_nan) {
        // Set solution to zero if input has NaN
        for (int i = 0; i < n; ++i) {
            b[i] = 0.0;
        }
        return false;
    }

    std::vector<double> singular_values(n);
    int rank = 0;
    int info = 0;
    int nrhs = 1;

    // Workspace query
    int lwork = -1;
    double work_query;
    int iwork_size = 0;
    dgelsd_(&m, &n, &nrhs, a, &n, b, &m, singular_values.data(),
            &rcond, &rank, &work_query, &lwork, &iwork_size, &info);

    lwork = static_cast<int>(work_query);
    std::vector<double> work(lwork);

    // Estimate iwork size
    int smlsiz = 25;  // LAPACK parameter
    int nlvl = static_cast<int>(std::log2(std::min(m, n) / (smlsiz + 1))) + 1;
    iwork_size = 3 * std::min(m, n) * nlvl + 11 * std::min(m, n);
    std::vector<int> iwork(iwork_size);

    // Actual computation
    dgelsd_(&m, &n, &nrhs, a, &n, b, &m, singular_values.data(),
            &rcond, &rank, work.data(), &lwork, iwork.data(), &info);

    if (info != 0) {
        for (int i = 0; i < n; ++i) {
            b[i] = -1e19;
        }
        return false;
    }

    return true;
}

/**
 * @brief Invert matrix in-place using LAPACK dgetrf + dgetri
 * @param n Dimension of matrix (n x n)
 * @param a Matrix to invert (will be overwritten with inverse)
 * @return true if successful, false if matrix is singular
 */
inline bool invert_matrix(int n, double* a) {
    // Check for NaN in input
    bool has_nan = false;
    for (int i = 0; i < n * n; ++i) {
        if (std::isnan(a[i])) {
            has_nan = true;
            break;
        }
    }

    if (has_nan) {
        // Set to zero if input has NaN
        for (int i = 0; i < n * n; ++i) {
            a[i] = 0.0;
        }
        return false;
    }

    std::vector<int> ipiv(n);
    int info = 0;

    // LU factorization
    dgetrf_(&n, &n, a, &n, ipiv.data(), &info);
    if (info != 0) {
        for (int i = 0; i < n * n; ++i) {
            a[i] = -1e19;
        }
        return false;
    }

    // Compute inverse
    std::vector<double> work(n);
    int lwork = n;
    dgetri_(&n, a, &n, ipiv.data(), work.data(), &lwork, &info);

    if (info != 0) {
        for (int i = 0; i < n * n; ++i) {
            a[i] = -1e19;
        }
        return false;
    }

    return true;
}

} // namespace lapack
} // namespace pycalphad

#endif // PYCALPHAD_LAPACK_WRAPPER_HPP
