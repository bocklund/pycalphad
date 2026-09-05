from typing import cast
import numpy as np
from collections import namedtuple
from pycalphad.core.minimizer import SystemSpecification
from pycalphad.core.errors import ConditionError

SolverResult = namedtuple('SolverResult', ['converged', 'x', 'chemical_potentials'])

class SolverBase(object):
    """"Base class for solvers."""
    ignore_convergence = False
    def solve(self, composition_sets, conditions):
        """
        *Implement this method.*
        Minimize the energy under the specified conditions using the given candidate composition sets.

        Parameters
        ----------
        composition_sets : List[pycalphad.core.composition_set.CompositionSet]
            List of CompositionSet objects in the starting point. Modified in place.
        conditions : OrderedDict[str, float]
            Conditions to satisfy.

        Returns
        -------
        pycalphad.core.solver.SolverResult
        """
        raise NotImplementedError("A subclass of Solver must be implemented.")


class Solver(SolverBase):
    def __init__(self, verbose=False, remove_metastable=True, **options):
        self.verbose = verbose
        self.remove_metastable = remove_metastable
        self.options = options


    def get_system_spec(self, composition_sets, conditions):
        """
        Create a SystemSpecification object for the specified conditions.

        Parameters
        ----------
        composition_sets : List[pycalphad.core.composition_set.CompositionSet]
            List of CompositionSet objects in the starting point. Modified in place.
        conditions : OrderedDict[StateVariable, float]
            Conditions to satisfy.

        Returns
        -------
        SystemSpecification

        """
        # Prevent circular import
        from pycalphad.variables import ChemicalPotential, MassFraction, MoleFraction, \
            SiteFraction, Moles
        compsets = composition_sets
        state_variables = compsets[0].phase_record.state_variables
        nonvacant_elements = compsets[0].phase_record.nonvacant_elements
        num_statevars = len(state_variables)
        num_components = len(nonvacant_elements)
        chemical_potentials = np.zeros(num_components)
        # Redefined-component basis. For a trivial (pure-element) basis these are unused and
        # the original element-basis construction below runs unchanged.
        phase_record = compsets[0].phase_record
        basis_is_trivial = phase_record.basis_is_trivial
        basis_component_index = phase_record.basis_component_index
        component_basis_inv_T = np.asarray(phase_record.component_basis_inv_T)
        component_molar_masses = np.asarray(phase_record.component_molar_masses)
        molar_masses = np.asarray(phase_record.molar_masses)
        inv_T_colsum = component_basis_inv_T.sum(axis=0)

        def basis_row(species):
            name = str(species)
            if name not in basis_component_index:
                raise ConditionError(
                    f"{name!r} is not part of the component basis {list(basis_component_index)}; "
                    f"conditions must be expressed in terms of basis components.")
            return basis_component_index[name]
        # X(i), W(i)
        prescribed_mole_fraction_coefficients = []
        prescribed_mole_fraction_rhs = []
        # N, N(i), B, B(i)
        prescribed_mole_amount_coefficients = []
        prescribed_mole_amount_rhs = []
        local_conditions = {key: value for key, value in conditions.items()
                            if getattr(key, 'phase_name', None) is not None}
        for compset in compsets:
            phase_local_conditions = {key: value for key, value in local_conditions.items()
                                      if compset.phase_record.phase_name == key.phase_name}
            if len(phase_local_conditions) > 0:
                compset.set_local_conditions(phase_local_conditions)
        for cond, value in conditions.items():
            # values should all be scalar floats
            value = float(np.asarray(value).flat[0])
            if isinstance(cond, MoleFraction) and cond.phase_name is None:
                # X(c) = k. The direct form dot(row, x) = k stays one-hot for the trivial
                # pure-element basis, which the Jansson-derivative matching in
                # fixed_component_differential relies on. Any redefined basis uses the
                # homogeneous form ((S^T)^-1[c,:] - k*colsum).x = 0 even when the
                # component's own row is one-hot and the colsum is all ones (e.g. a
                # pure-element basis component alongside a unit-atom multi-element
                # component): MoleFraction.jansson_deltas locates the condition row by
                # reconstructing the homogeneous coefficients, so the stored form must
                # match it.
                coefs = np.array(component_basis_inv_T[basis_row(cond.species)])
                if basis_is_trivial and np.count_nonzero(coefs) == 1 and np.isclose(coefs.sum(), 1.0):
                    prescribed_mole_fraction_rhs.append(value)
                else:
                    coefs = coefs - value * inv_T_colsum
                    prescribed_mole_fraction_rhs.append(0.0)
                prescribed_mole_fraction_coefficients.append(coefs)
            elif isinstance(cond, MoleFraction) and cond.phase_name is not None:
                # phase-local condition; already handled
                continue
            elif isinstance(cond, SiteFraction):
                # phase-local condition; already handled
                continue
            elif isinstance(cond, MassFraction):
                # W(c) = k  <=>  (MW(c) * (S^T)^-1[c,:] - k * mass_elem) . x_elem = 0.
                # For a pure-element basis this reduces to (1-k)*MW_A*x_A - k*MW_B*x_B - ... = 0.
                row = basis_row(cond.species)
                coef_vector = component_molar_masses[row] * component_basis_inv_T[row] - value * molar_masses
                prescribed_mole_fraction_rhs.append(0.)
                prescribed_mole_fraction_coefficients.append(coef_vector)
            elif str(cond).startswith('LinComb_'):
                # Linear combination of (component) mole fractions. With
                # X(c) = (S^T)^-1[c,:].x / (colsum.x), sum_k a_k X(c_k) + const = value
                # multiplies through by the (positive) component total to the homogeneous
                # row (sum_k a_k (S^T)^-1[c_k] - (value-const)*colsum).x = 0. For a molar
                # ratio the component total cancels, leaving (S^T)^-1[num] - value*(S^T)^-1[den].
                # For a pure-element basis (S = I) this is the element linear combination.
                coefs = np.zeros(num_components)
                constant = 0.0
                for symbol, coef in zip(cond.symbols, cond.coefs):
                    if symbol == 1:
                        constant = coef
                        continue
                    coefs = coefs + coef * component_basis_inv_T[basis_row(symbol.species)]
                if cond.denominator == 1:
                    coefs = coefs - (value - float(constant)) * inv_T_colsum
                else:
                    coefs = coefs - value * component_basis_inv_T[basis_row(cond.denominator.species)]
                prescribed_mole_fraction_rhs.append(0.0)
                prescribed_mole_fraction_coefficients.append(coefs)
            elif isinstance(cond, Moles) or (isinstance(cond, str) and (cond == 'N' or cond.startswith('N_'))):
                # Extensive: total N (species is None) or component N(i). Conditions may be
                # keyed by a Moles object or by string ('N', 'N_<EL>'). Total moles exposes no
                # `species` attribute, so probe defensively with getattr.
                if isinstance(cond, Moles):
                    # Phase-local moles are rejected upstream (Conditions); a direct Solver
                    # caller is backstopped by set_local_conditions / build_phase_local_constraints.
                    cond_species = getattr(cond, 'species', None)
                else:
                    # cond must be a string per outer elif
                    cond_species = None if cond == 'N' else cast(str, cond)[2:]
                if cond_species is None:
                    coefs = np.ones(num_components)  # total N: total moles of atoms (basis-independent)
                else:
                    # N(c) = (S^T)^-1[c,:] . n_elem  (a unit vector for a pure-element basis)
                    coefs = component_basis_inv_T[basis_row(cond_species)].copy()
                prescribed_mole_amount_coefficients.append(coefs)
                prescribed_mole_amount_rhs.append(value)
        prescribed_mole_fraction_coefficients = np.array(prescribed_mole_fraction_coefficients).reshape(-1, num_components)
        prescribed_mole_fraction_rhs = np.array(prescribed_mole_fraction_rhs)
        if len(prescribed_mole_amount_coefficients) > 0:
            prescribed_mole_amount_coefficients = np.atleast_2d(prescribed_mole_amount_coefficients)
        else:
            prescribed_mole_amount_coefficients = np.zeros((0, num_components))
        prescribed_mole_amount_rhs = np.array(prescribed_mole_amount_rhs, dtype=np.float64)

        # MU conditions (keyed by ChemicalPotential objects or 'MU_<NAME>' strings).
        # A pure-element MU fixes that element's chemical potential by index (works under any
        # basis; the basis only reinterprets amount conditions, not element potentials).
        # MU(component) for a genuine multi-element or multiplied component (e.g. CL2) fixes a
        # linear combination of element chemical potentials, sum_e constituents[e]*mu_e = value,
        # added as a constraint row; the element chemical potentials stay free and solved.
        fixed_chempot_coefs = []
        fixed_chempot_rhs = []
        fixed_chemical_potential_indices = []
        for key, cond_value in conditions.items():
            if isinstance(key, str) and key.startswith('MU_'):
                key = ChemicalPotential(key[3:])
            elif not isinstance(key, ChemicalPotential):
                continue
            value = float(np.asarray(cond_value).flat[0])
            constituents = {el: mult for el, mult in key.species.constituents.items() if el != 'VA'}
            if len(constituents) == 1 and np.isclose(next(iter(constituents.values())), 1.0):
                el_index = nonvacant_elements.index(next(iter(constituents)))
                fixed_chemical_potential_indices.append(el_index)
                chemical_potentials[el_index] = value
            else:
                row = np.zeros(num_components)
                for el, mult in constituents.items():
                    row[nonvacant_elements.index(el)] = mult
                fixed_chempot_coefs.append(row)
                fixed_chempot_rhs.append(value)
        fixed_chemical_potential_indices = np.array(sorted(fixed_chemical_potential_indices), dtype=np.int32)
        if len(fixed_chempot_coefs) > 0:
            fixed_chempot_coefs = np.atleast_2d(fixed_chempot_coefs)
        else:
            fixed_chempot_coefs = np.zeros((0, num_components))
        fixed_chempot_rhs = np.array(fixed_chempot_rhs, dtype=np.float64)
        free_chemical_potential_indices = np.array(sorted(set(range(num_components)) - set(fixed_chemical_potential_indices)), dtype=np.int32)
        fixed_statevar_indices = []
        for statevar_idx, statevar in enumerate(state_variables):
            if str(statevar) in [str(k) for k in conditions.keys()]:
                fixed_statevar_indices.append(statevar_idx)
        free_statevar_indices = np.array(sorted(set(range(num_statevars)) - set(fixed_statevar_indices)), dtype=np.int32)
        fixed_statevar_indices = np.array(fixed_statevar_indices, dtype=np.int32)
        fixed_stable_compset_indices = np.array([i for i, compset in enumerate(compsets) if compset.fixed], dtype=np.int32)
        spec = SystemSpecification(num_statevars, num_components,
                                   prescribed_mole_amount_coefficients, prescribed_mole_amount_rhs,
                                   chemical_potentials, prescribed_mole_fraction_coefficients,
                                   prescribed_mole_fraction_rhs,
                                   free_chemical_potential_indices, free_statevar_indices,
                                   fixed_chemical_potential_indices, fixed_statevar_indices,
                                   fixed_stable_compset_indices,
                                   fixed_chempot_coefs, fixed_chempot_rhs,
                                   debugging_output=self.options.get("debugging_output", False)
                                   )
        return spec

    @staticmethod
    def _fix_state_variables_in_compsets(composition_sets, conditions):
        "Ensure state variables in each CompositionSet are set to the fixed value."
        str_state_variables = [str(k) for k in composition_sets[0].phase_record.state_variables]
        for compset in composition_sets:
            for k,v in conditions.items():
                if str(k) in str_state_variables:
                    statevar_idx = str_state_variables.index(str(k))
                    compset.dof[statevar_idx] = v

    def solve(self, composition_sets, conditions):
        """
        Minimize the energy under the specified conditions using the given candidate composition sets.

        Parameters
        ----------
        composition_sets : List[pycalphad.core.composition_set.CompositionSet]
            List of CompositionSet objects in the starting point. Modified in place.
        conditions : OrderedDict[str, float]
            Conditions to satisfy.

        Returns
        -------
        SolverResult

        """
        if self.verbose:
            print(f"Solver: Attempting to solve system at conditions {conditions} with starting point: {composition_sets}")
        spec = self.get_system_spec(composition_sets, conditions)
        self._fix_state_variables_in_compsets(composition_sets, conditions)
        state = spec.get_new_state(composition_sets)
        converged = spec.run_loop(state, self.options.get("max_iterations", 1000))

        if self.remove_metastable:
            phase_idx = 0
            compsets_to_remove = []
            for compset in composition_sets:
                # Mark unstable phases for removal
                if compset.NP <= 0.0 and not compset.fixed:
                    compsets_to_remove.append(int(phase_idx))
                phase_idx += 1
            # Watch removal order here, as the indices of composition_sets are changing!
            for idx in reversed(compsets_to_remove):
                del composition_sets[idx]

        phase_amt = [compset.NP for compset in composition_sets]

        x = composition_sets[0].dof
        state_variables = composition_sets[0].phase_record.state_variables
        num_statevars = len(state_variables)
        for compset in composition_sets[1:]:
            x = np.r_[x, compset.dof[num_statevars:]]
        x = np.r_[x, phase_amt]
        chemical_potentials = np.array(state.chemical_potentials)

        if self.verbose:
            soln_desc = f"Chemical Potentials: {np.round(chemical_potentials, 5).tolist()}; Composition Sets: {composition_sets}"
            if converged:
                print(f"Solver: (converged in {state.iteration+1} iterations): {soln_desc}", )
            else:
                print(f"Solver: (not converged): {soln_desc}", )
        return SolverResult(converged=converged, x=x, chemical_potentials=chemical_potentials)
