#ifndef PYCALPHAD_HYPERPLANE_HPP
#define PYCALPHAD_HYPERPLANE_HPP

#include "pycalphad_types.hpp"

/**
 * @file hyperplane.hpp
 * @brief Convex hull and hyperplane calculations for phase equilibria
 *
 * This module implements the hyperplane algorithm for finding tangent planes
 * to energy surfaces, which is fundamental to equilibrium calculations.
 */

namespace pycalphad {
namespace hyperplane {

/**
 * @brief Solve linear system AX = B using LAPACK dgesv
 *
 * Wraps scipy's LAPACK interface. On singular matrix (info != 0),
 * sets solution to sentinel value -1e19.
 *
 * @param A Matrix A (N x N), will be overwritten with LU factorization
 * @param N Dimension of matrix
 * @param x Right-hand side vector, will be overwritten with solution
 * @param ipiv Pivot indices workspace (size N)
 */
void solve(double* A, int N, double* x, int* ipiv);

/**
 * @brief Compute product-sum: result -= chempots * points
 *
 * @param chempots Chemical potentials array (size n_chempots)
 * @param points Composition points (n_points x n_chempots)
 * @param result Result array (size n_points), modified in-place
 * @param n_chempots Number of chemical potentials
 * @param n_points Number of points
 */
void prodsum(const double* chempots, const double* points, double* result,
             int n_chempots, int n_points);

/**
 * @brief Find minimum value in array
 *
 * @param a Array to search
 * @param a_shape Length of array
 * @return Minimum value (or 1e300 if array is empty)
 */
double min_value(const double* a, int a_shape);

/**
 * @brief Find index of minimum value in array
 *
 * @param a Array to search
 * @param a_shape Length of array
 * @param lowest Output parameter for minimum value
 * @return Index of minimum value
 */
int argmin(const double* a, int a_shape, double* lowest);

/**
 * @brief Find index of maximum value in array
 *
 * @param a Array to search
 * @param a_shape Length of array
 * @return Index of maximum value
 */
int argmax(const double* a, int a_shape);

/**
 * @brief Compute hyperplane coefficients
 *
 * Solves for coefficients that define the hyperplane passing through
 * the trial simplex with fixed chemical potential constraints.
 *
 * @param compositions Composition matrix (n_points x n_components)
 * @param n_points Number of points
 * @param n_components Number of components
 * @param fixed_chempot_indices Indices of fixed chemical potentials
 * @param n_fixed Number of fixed indices
 * @param trial_simplex Indices of points in simplex
 * @param simplex_size Size of simplex
 * @param out_plane_coefs Output coefficients (size n_components)
 * @throws std::invalid_argument if matrix is not square
 */
void hyperplane_coefficients(const double* compositions,
                            int n_points,
                            int n_components,
                            const size_t* fixed_chempot_indices,
                            int n_fixed,
                            const int* trial_simplex,
                            int simplex_size,
                            double* out_plane_coefs);

/**
 * @brief Find intersection point of hyperplane with constraints
 *
 * Computes the point where the hyperplane intersects with linear
 * mole fraction constraints.
 *
 * @param compositions Composition matrix (n_points x n_components)
 * @param n_points Number of points
 * @param n_components Number of components
 * @param fixed_chempot_indices Indices of fixed chemical potentials
 * @param n_fixed Number of fixed indices
 * @param trial_simplex Indices of points in simplex
 * @param simplex_size Size of simplex
 * @param fixed_lincomb_molefrac_coefs Linear constraint coefficients (n_constraints x n_components)
 * @param fixed_lincomb_molefrac_rhs Right-hand side of constraints
 * @param n_constraints Number of linear constraints
 * @param out_intersecting_point Output intersection point (size n_components)
 * @throws std::invalid_argument if constraint matrix is not square
 */
void intersecting_point(const double* compositions,
                       int n_points,
                       int n_components,
                       const size_t* fixed_chempot_indices,
                       int n_fixed,
                       const int* trial_simplex,
                       int simplex_size,
                       const double* fixed_lincomb_molefrac_coefs,
                       const double* fixed_lincomb_molefrac_rhs,
                       int n_constraints,
                       double* out_intersecting_point);

/**
 * @brief Compute fractions of simplex components
 *
 * Solves for the fractions of each simplex vertex needed to
 * represent the intersecting point in the free component space.
 *
 * @param compositions Composition matrix (n_points x n_components)
 * @param n_points Number of points
 * @param n_components Number of components
 * @param fixed_chempot_indices Indices of fixed chemical potentials
 * @param n_fixed Number of fixed indices
 * @param trial_simplex Indices of points in simplex
 * @param simplex_size Size of simplex
 * @param fixed_lincomb_molefrac_coefs Linear constraint coefficients
 * @param fixed_lincomb_molefrac_rhs Right-hand side of constraints
 * @param n_constraints Number of linear constraints
 * @param out_fractions Output fractions (size simplex_size)
 */
void simplex_fractions(const double* compositions,
                      int n_points,
                      int n_components,
                      const size_t* fixed_chempot_indices,
                      int n_fixed,
                      const int* trial_simplex,
                      int simplex_size,
                      const double* fixed_lincomb_molefrac_coefs,
                      const double* fixed_lincomb_molefrac_rhs,
                      int n_constraints,
                      double* out_fractions);

/**
 * @brief Find chemical potentials for tangent hyperplane
 *
 * Main hyperplane algorithm. Iteratively finds the convex hull
 * by computing tangent hyperplanes and minimizing driving forces.
 *
 * @param compositions Composition matrix (n_points x n_components)
 * @param n_points Number of sampled points (M)
 * @param n_components Number of components (N)
 * @param energies Energy array (size n_points)
 * @param chemical_potentials Chemical potentials (size n_components), will be modified
 * @param fixed_chempot_indices Indices of fixed chemical potentials
 * @param n_fixed Number of fixed indices
 * @param fixed_lincomb_molefrac_coefs Linear constraint coefficients (n_constraints x n_components)
 * @param fixed_lincomb_molefrac_rhs Right-hand side of constraints
 * @param n_constraints Number of linear constraints
 * @param result_fractions Output simplex fractions (size simplex_size), sums to 1
 * @param result_simplex Output simplex point indices (size simplex_size)
 * @param simplex_size Size of simplex (n_components - n_fixed)
 * @return Energy of the hyperplane at the target composition
 *
 * @note M = number of sampled energy points
 * @note N = number of components
 * @note P = N+1 (max phases by Gibbs phase rule in point calculation)
 */
double hyperplane_main(const double* compositions,
                      int n_points,
                      int n_components,
                      const double* energies,
                      double* chemical_potentials,
                      const size_t* fixed_chempot_indices,
                      int n_fixed,
                      const double* fixed_lincomb_molefrac_coefs,
                      const double* fixed_lincomb_molefrac_rhs,
                      int n_constraints,
                      double* result_fractions,
                      int* result_simplex,
                      int simplex_size);

} // namespace hyperplane
} // namespace pycalphad

#endif // PYCALPHAD_HYPERPLANE_HPP
