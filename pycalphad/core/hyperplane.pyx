# distutils: language = c++
"""
Hyperplane calculations for convex hull determination.

This module provides thin Cython wrappers around the C++ hyperplane
implementation. The C++ code performs all the actual computations.
"""
cimport numpy as np
import numpy as np
cimport cython
cimport scipy.linalg.cython_lapack as cython_lapack

# Declare the callback setter from C++
cdef extern from *:
    """
    extern "C" void set_solve_callback(void (*callback)(double*, int, double*, int*));
    """
    void set_solve_callback(void (*callback)(double*, int, double*, int*))

# Implement LAPACK solve using scipy
cdef void solve_impl(double* A, int N, double* x, int* ipiv) noexcept nogil:
    """LAPACK solve implementation using scipy's cython_lapack."""
    cdef int info = 0
    cdef int NRHS = 1
    with gil:
        cython_lapack.dgesv(&N, &NRHS, A, &N, ipiv, x, &N, &info)
    # Special for our case: singular matrix results get set to a special value
    if info != 0:
        for i in range(N):
            x[i] = -1e19

# Export the solve function with C linkage so it can be passed as a callback
cdef public void hyperplane_solve_callback(double* A, int N, double* x, int* ipiv) noexcept nogil:
    """Public wrapper for solve_impl that can be used as a C callback."""
    solve_impl(A, N, x, ipiv)

# Declare C++ functions from hyperplane.hpp
cdef extern from "hyperplane.hpp" namespace "pycalphad::hyperplane":
    void prodsum(const double* chempots, const double* points, double* result,
                 int n_chempots, int n_points) nogil
    double min_value(const double* a, int a_shape) nogil
    int argmin(const double* a, int a_shape, double* lowest) nogil
    int argmax(const double* a, int a_shape) nogil

    void hyperplane_coefficients_cpp "pycalphad::hyperplane::hyperplane_coefficients" (
                                const double* compositions,
                                int n_points,
                                int n_components,
                                const size_t* fixed_chempot_indices,
                                int n_fixed,
                                const int* trial_simplex,
                                int simplex_size,
                                double* out_plane_coefs) except +

    void intersecting_point_cpp "pycalphad::hyperplane::intersecting_point" (
                           const double* compositions,
                           int n_points,
                           int n_components,
                           const size_t* fixed_chempot_indices,
                           int n_fixed,
                           const int* trial_simplex,
                           int simplex_size,
                           const double* fixed_lincomb_molefrac_coefs,
                           const double* fixed_lincomb_molefrac_rhs,
                           int n_constraints,
                           double* out_intersecting_point) except +

    void simplex_fractions_cpp "pycalphad::hyperplane::simplex_fractions" (
                          const double* compositions,
                          int n_points,
                          int n_components,
                          const size_t* fixed_chempot_indices,
                          int n_fixed,
                          const int* trial_simplex,
                          int simplex_size,
                          const double* fixed_lincomb_molefrac_coefs,
                          const double* fixed_lincomb_molefrac_rhs,
                          int n_constraints,
                          double* out_fractions) except +

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
                          int simplex_size) except +


# Thin wrappers that convert numpy arrays to pointers and call C++

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void hyperplane_coefficients(double[:,::1] compositions,
                                   size_t[::1] fixed_chempot_indices,
                                   int[::1] trial_simplex,
                                   double[::1] out_plane_coefs) except *:
    """
    Compute hyperplane coefficients.

    Thin wrapper around C++ implementation.
    """
    hyperplane_coefficients_cpp(
        &compositions[0, 0],
        compositions.shape[0],
        compositions.shape[1],
        &fixed_chempot_indices[0] if fixed_chempot_indices.shape[0] > 0 else NULL,
        fixed_chempot_indices.shape[0],
        &trial_simplex[0],
        trial_simplex.shape[0],
        &out_plane_coefs[0]
    )


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void intersecting_point(double[:,::1] compositions,
                              size_t[::1] fixed_chempot_indices,
                              int[::1] trial_simplex,
                              double[:,::1] fixed_lincomb_molefrac_coefs,
                              double[::1] fixed_lincomb_molefrac_rhs,
                              double[::1] out_intersecting_point) except *:
    """
    Find intersection point of hyperplane with constraints.

    Thin wrapper around C++ implementation.
    """
    intersecting_point_cpp(
        &compositions[0, 0],
        compositions.shape[0],
        compositions.shape[1],
        &fixed_chempot_indices[0] if fixed_chempot_indices.shape[0] > 0 else NULL,
        fixed_chempot_indices.shape[0],
        &trial_simplex[0],
        trial_simplex.shape[0],
        &fixed_lincomb_molefrac_coefs[0, 0] if fixed_lincomb_molefrac_coefs.shape[0] > 0 else NULL,
        &fixed_lincomb_molefrac_rhs[0] if fixed_lincomb_molefrac_rhs.shape[0] > 0 else NULL,
        fixed_lincomb_molefrac_rhs.shape[0],
        &out_intersecting_point[0]
    )


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double hyperplane(double[:,::1] compositions,
                        double[::1] energies,
                        double[::1] chemical_potentials,
                        size_t[::1] fixed_chempot_indices,
                        double[:, ::1] fixed_lincomb_molefrac_coefs,
                        double[::1] fixed_lincomb_molefrac_rhs,
                        double[::1] result_fractions,
                        int[::1] result_simplex) except *:
    """
    Find chemical potentials which approximate the tangent hyperplane
    at the given composition.

    Thin wrapper around C++ implementation.

    Parameters
    ----------
    compositions : ndarray
        A sample of the energy surface of the system.
        Aligns with 'energies'.
        Shape of (M, N)
    energies : ndarray
        A sample of the energy surface of the system.
        Aligns with 'compositions'.
        Shape of (M,)
    chemical_potentials : ndarray
        Shape of (N,)
        Will be overwritten
    fixed_chempot_indices : ndarray
        Variable shape from (0,) to (N-1,)
    fixed_lincomb_molefrac_coefs : ndarray
        Variable shape from (0,P) to (N-1, P)
    fixed_lincomb_molefrac_rhs : ndarray
        Variable shape from (0,) to (N-1,)
    result_fractions : ndarray
        Relative amounts of the points making up the hyperplane simplex. Shape of (P,).
        Will be overwritten. Output sums to 1.
    result_simplex : ndarray
        Energies of the points making up the hyperplane simplex. Shape of (P,).
        Will be overwritten. Output*result_fractions sums to out_energy (return value).

    Returns
    -------
    out_energy : double
        Energy of the output configuration.

    Examples
    --------
    None yet.

    Notes
    -----
    M: number of energy points that have been sampled
    N: number of components
    P: N+1, max phases by gibbs phase rule that we can find in a point calculations
    """
    cdef int n_points = compositions.shape[0]
    cdef int n_components = compositions.shape[1]
    cdef int n_fixed = fixed_chempot_indices.shape[0]
    cdef int simplex_size = n_components - n_fixed
    cdef int n_constraints = fixed_lincomb_molefrac_rhs.shape[0]

    return hyperplane_main(
        &compositions[0, 0],
        n_points,
        n_components,
        &energies[0],
        &chemical_potentials[0],
        &fixed_chempot_indices[0] if n_fixed > 0 else NULL,
        n_fixed,
        &fixed_lincomb_molefrac_coefs[0, 0] if n_constraints > 0 else NULL,
        &fixed_lincomb_molefrac_rhs[0] if n_constraints > 0 else NULL,
        n_constraints,
        &result_fractions[0],
        &result_simplex[0],
        simplex_size
    )


# Module initialization: Set the LAPACK solve callback
# This is called when the module is first imported
cdef void _init_module():
    """Initialize the C++ module with the LAPACK solve callback."""
    set_solve_callback(&hyperplane_solve_callback)

# Call initialization
_init_module()
