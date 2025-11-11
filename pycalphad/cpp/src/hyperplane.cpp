#include "hyperplane.hpp"
#include "memory_utils.hpp"
#include <cstring>
#include <vector>
#include <set>
#include <algorithm>
#include <stdexcept>

// Forward declare LAPACK function
// This will be provided by scipy.linalg.cython_lapack through the Cython wrapper
extern "C" {
    void dgesv_(const int* n, const int* nrhs, double* a, const int* lda,
                int* ipiv, double* b, const int* ldb, int* info);
}

namespace pycalphad {
namespace hyperplane {

void solve(double* A, int N, double* x, int* ipiv) {
    int info = 0;
    int NRHS = 1;
    dgesv_(&N, &NRHS, A, &N, ipiv, x, &N, &info);

    // Special for our case: singular matrix results get set to a special value
    if (info != 0) {
        for (int i = 0; i < N; ++i) {
            x[i] = -1e19;
        }
    }
}

void prodsum(const double* chempots, const double* points, double* result,
             int n_chempots, int n_points) {
    for (int i = 0; i < n_chempots; ++i) {
        for (int j = 0; j < n_points; ++j) {
            // points is stored in row-major order: points[j, i] = points[j*n_chempots + i]
            result[j] -= chempots[i] * points[j * n_chempots + i];
        }
    }
}

double min_value(const double* a, int a_shape) {
    double result = 1e300;
    for (int i = 0; i < a_shape; ++i) {
        if (a[i] < result) {
            result = a[i];
        }
    }
    return result;
}

int argmin(const double* a, int a_shape, double* lowest) {
    int result = 0;
    for (int i = 0; i < a_shape; ++i) {
        if (a[i] < lowest[0]) {
            lowest[0] = a[i];
            result = i;
        }
    }
    return result;
}

int argmax(const double* a, int a_shape) {
    int result = 0;
    double highest = -1e30;
    for (int i = 0; i < a_shape; ++i) {
        if (a[i] > highest) {
            highest = a[i];
            result = i;
        }
    }
    return result;
}

void hyperplane_coefficients(const double* compositions,
                            int n_points,
                            int n_components,
                            const size_t* fixed_chempot_indices,
                            int n_fixed,
                            const int* trial_simplex,
                            int simplex_size,
                            double* out_plane_coefs) {
    int plane_rows = simplex_size + n_fixed;
    if (plane_rows != n_components) {
        throw std::invalid_argument("Hyperplane coefficient matrix is not square");
    }

    // Allocate working memory
    auto f_plane_matrix = memory::MallocPtr<double>(plane_rows * n_components);
    auto int_tmp = memory::MallocPtr<int>(plane_rows);

    // Fill matrix with trial simplex compositions (Fortran order)
    for (int i = 0; i < simplex_size; ++i) {
        for (int j = 0; j < n_components; ++j) {
            // Fortran order: f_plane_matrix[i + j*plane_rows]
            // compositions is row-major: compositions[trial_simplex[i], j]
            f_plane_matrix[i + j * plane_rows] = compositions[trial_simplex[i] * n_components + j];
        }
        out_plane_coefs[i] = 1.0;
    }

    // Add rows for fixed chemical potentials
    for (int i = 0; i < n_fixed; ++i) {
        for (int j = 0; j < n_components; ++j) {
            f_plane_matrix[i + simplex_size + j * plane_rows] = 0.0;
        }
        f_plane_matrix[i + simplex_size + fixed_chempot_indices[i] * plane_rows] = 1.0;
        out_plane_coefs[i + simplex_size] = 0.0;
    }

    solve(f_plane_matrix.get(), plane_rows, out_plane_coefs, int_tmp.get());
}

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
                       double* out_intersecting_point) {
    // Simplex is zero-dimensional, so there is no intersection;
    // just return the point defining the 0-simplex
    if (simplex_size == 1) {
        for (int i = 0; i < n_components; ++i) {
            out_intersecting_point[i] = compositions[trial_simplex[0] * n_components + i];
        }
        return;
    }

    if ((n_constraints + 1 != n_components) && n_fixed > 0) {
        throw std::invalid_argument("Constraint matrix is not square");
    }

    // Allocate working memory
    auto int_tmp = memory::MallocPtr<int>(n_components);
    auto constraint_matrix = memory::MallocPtr<double>((n_constraints + 1) * n_components);
    auto constraint_rhs = memory::MallocPtr<double>(n_constraints + 1);

    // Initialize output to zero
    std::memset(out_intersecting_point, 0, n_components * sizeof(double));

    // Get hyperplane coefficients (stored in out_intersecting_point temporarily)
    hyperplane_coefficients(compositions, n_points, n_components,
                           fixed_chempot_indices, n_fixed,
                           trial_simplex, simplex_size,
                           out_intersecting_point);

    // Build constraint system (Fortran order)
    for (int j = 0; j < n_components; ++j) {
        for (int i = 0; i < n_constraints; ++i) {
            // Fortran order: constraint_matrix[i + j*n_components]
            // Input is row-major: fixed_lincomb_molefrac_coefs[i, j] = [i*n_components + j]
            constraint_matrix[i + j * (n_constraints + 1)] =
                fixed_lincomb_molefrac_coefs[i * n_components + j];
            constraint_rhs[i] = fixed_lincomb_molefrac_rhs[i];
        }
        constraint_matrix[n_constraints + j * (n_constraints + 1)] = out_intersecting_point[j];
        constraint_rhs[n_constraints] = 1.0;
    }

    solve(constraint_matrix.get(), n_components, constraint_rhs.get(), int_tmp.get());

    // Copy solution to output
    for (int i = 0; i < n_components; ++i) {
        out_intersecting_point[i] = constraint_rhs[i];
    }
}

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
                      double* out_fractions) {
    // Allocate working memory
    auto f_coord_matrix = memory::MallocPtr<double>(simplex_size * simplex_size);
    auto target_point = memory::MallocPtr<double>(n_components);
    auto int_tmp = memory::MallocPtr<int>(simplex_size);

    // Compute free chemical potential indices
    std::set<size_t> fixed_set(fixed_chempot_indices, fixed_chempot_indices + n_fixed);
    std::vector<size_t> free_chempot_indices;
    for (int i = 0; i < n_components; ++i) {
        if (fixed_set.find(i) == fixed_set.end()) {
            free_chempot_indices.push_back(i);
        }
    }

    // Get target point for calculation
    intersecting_point(compositions, n_points, n_components,
                      fixed_chempot_indices, n_fixed,
                      trial_simplex, simplex_size,
                      fixed_lincomb_molefrac_coefs,
                      fixed_lincomb_molefrac_rhs,
                      n_constraints,
                      target_point.get());

    // Fill coordinate matrix (Fortran order)
    for (int j = 0; j < simplex_size; ++j) {
        for (int i = 0; i < simplex_size; ++i) {
            // Fortran order: f_coord_matrix[j + simplex_size*i]
            // compositions[trial_simplex[i], free_chempot_indices[j]]
            int comp_idx = trial_simplex[i] * n_components + free_chempot_indices[j];
            f_coord_matrix[j + simplex_size * i] = compositions[comp_idx];
        }
        out_fractions[j] = target_point[free_chempot_indices[j]];
    }

    solve(f_coord_matrix.get(), simplex_size, out_fractions, int_tmp.get());
}

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
                      int simplex_size) {
    // Constants
    const int max_iterations = 1000;
    const double convergence_tol = -1e-8;

    // Initialize best guess simplex with first simplex_size non-fixed indices
    auto best_guess_simplex = memory::MallocPtr<int>(simplex_size);
    int fixed_index = 0;
    std::set<size_t> fixed_set(fixed_chempot_indices, fixed_chempot_indices + n_fixed);

    for (int i = 0; i < n_components && fixed_index < simplex_size; ++i) {
        if (fixed_set.find(i) == fixed_set.end()) {
            best_guess_simplex[fixed_index++] = i;
        }
    }

    // Allocate working arrays
    auto free_chempot_indices = memory::MallocPtr<int>(simplex_size);
    auto candidate_simplex = memory::MallocPtr<int>(simplex_size);
    auto int_tmp = memory::MallocPtr<int>(simplex_size);
    auto candidate_potentials = memory::MallocPtr<double>(simplex_size);
    auto smallest_fractions = memory::MallocPtr<double>(simplex_size);
    auto driving_forces = memory::MallocPtr<double>(n_points);
    auto trial_simplices = memory::MallocPtr<int>(simplex_size * simplex_size);
    auto fractions = memory::MallocPtr<double>(simplex_size * simplex_size);
    auto f_candidate_tieline = memory::MallocPtr<double>(simplex_size * simplex_size);

    // Initialize indices
    for (int i = 0; i < simplex_size; ++i) {
        free_chempot_indices[i] = best_guess_simplex[i];
        candidate_simplex[i] = best_guess_simplex[i];
    }

    // Initialize trial simplices
    for (int i = 0; i < simplex_size; ++i) {
        for (int j = 0; j < simplex_size; ++j) {
            trial_simplices[i * simplex_size + j] = best_guess_simplex[j];
        }
    }

    int iterations = 0;
    int saved_trial = 0;
    double out_energy = 0.0;

    while (iterations < max_iterations) {
        ++iterations;

        // Compute fractions for each trial simplex
        for (int trial_idx = 0; trial_idx < simplex_size; ++trial_idx) {
            for (int simplex_idx = 0; simplex_idx < simplex_size; ++simplex_idx) {
                fractions[trial_idx * simplex_size + simplex_idx] = 0.0;
            }

            simplex_fractions(compositions, n_points, n_components,
                            fixed_chempot_indices, n_fixed,
                            &trial_simplices[trial_idx * simplex_size], simplex_size,
                            fixed_lincomb_molefrac_coefs, fixed_lincomb_molefrac_rhs,
                            n_constraints,
                            &fractions[trial_idx * simplex_size]);

            smallest_fractions[trial_idx] = min_value(&fractions[trial_idx * simplex_size], simplex_size);
        }

        // Choose simplex with the largest smallest-fraction
        saved_trial = argmax(smallest_fractions.get(), simplex_size);
        if (smallest_fractions[saved_trial] < -simplex_size) {
            break;
        }

        // Update candidate simplex
        for (int i = 0; i < simplex_size; ++i) {
            candidate_simplex[i] = trial_simplices[saved_trial * simplex_size + i];
        }

        // Build candidate tieline matrix and potentials
        for (int i = 0; i < simplex_size; ++i) {
            int idx = candidate_simplex[i];
            for (int ici = 0; ici < simplex_size; ++ici) {
                int chempot_idx = free_chempot_indices[ici];
                // Fortran order: f_candidate_tieline[i + simplex_size*ici]
                f_candidate_tieline[i + simplex_size * ici] =
                    compositions[idx * n_components + chempot_idx];
            }
            candidate_potentials[i] = energies[idx];
            for (int ici = 0; ici < n_fixed; ++ici) {
                int chempot_idx = fixed_chempot_indices[ici];
                candidate_potentials[i] -= chemical_potentials[chempot_idx] *
                                          compositions[idx * n_components + chempot_idx];
            }
        }

        solve(f_candidate_tieline.get(), simplex_size, candidate_potentials.get(), int_tmp.get());

        if (candidate_potentials[0] == -1e19) {
            break;  // Singular matrix
        }

        // Compute driving forces
        for (int i = 0; i < n_points; ++i) {
            driving_forces[i] = energies[i];
        }

        for (int ici = 0; ici < simplex_size; ++ici) {
            int chempot_idx = free_chempot_indices[ici];
            for (int idx = 0; idx < n_points; ++idx) {
                driving_forces[idx] -= candidate_potentials[ici] *
                                      compositions[idx * n_components + chempot_idx];
            }
        }

        for (int ici = 0; ici < n_fixed; ++ici) {
            int chempot_idx = fixed_chempot_indices[ici];
            for (int idx = 0; idx < n_points; ++idx) {
                driving_forces[idx] -= chemical_potentials[chempot_idx] *
                                      compositions[idx * n_components + chempot_idx];
            }
        }

        // Update best guess
        for (int i = 0; i < simplex_size; ++i) {
            best_guess_simplex[i] = candidate_simplex[i];
        }

        for (int i = 0; i < simplex_size; ++i) {
            for (int j = 0; j < simplex_size; ++j) {
                trial_simplices[i * simplex_size + j] = best_guess_simplex[j];
            }
        }

        // Find point with minimum driving force
        double lowest_df = 1e10;
        int min_df = -1;
        for (int i = 0; i < n_points; ++i) {
            if (driving_forces[i] < lowest_df) {
                lowest_df = driving_forces[i];
                min_df = i;
            }
        }

        // Update trial simplices with minimum driving force point
        for (int i = 0; i < simplex_size; ++i) {
            trial_simplices[i * simplex_size + i] = min_df;
        }

        if (lowest_df > convergence_tol) {
            break;  // Converged
        }
    }

    // Compute output energy
    out_energy = 0.0;
    for (int i = 0; i < simplex_size; ++i) {
        int idx = best_guess_simplex[i];
        out_energy += fractions[saved_trial * simplex_size + i] * energies[idx];
    }

    // Copy results
    for (int i = 0; i < simplex_size; ++i) {
        result_fractions[i] = fractions[saved_trial * simplex_size + i];
    }

    for (int ici = 0; ici < simplex_size; ++ici) {
        int chempot_idx = free_chempot_indices[ici];
        chemical_potentials[chempot_idx] = candidate_potentials[ici];
        result_simplex[ici] = best_guess_simplex[ici];
    }

    // Hack to enforce Gibbs phase rule
    // Shape of result is comp+1, shape of hyperplane is comp
    for (int i = simplex_size; i < n_components + 1; ++i) {
        if (i < n_components + 1) {  // Safety check for array bounds
            result_fractions[i] = 0.0;
            result_simplex[i] = 0;
        }
    }

    return out_energy;
}

} // namespace hyperplane
} // namespace pycalphad
