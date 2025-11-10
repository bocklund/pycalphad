"""
Parallel equilibrium calculation using Python 3.13+ free threading.

This module provides functions to parallelize equilibrium calculations across
multiple condition sets using thread-based parallelism without GIL contention.

Requires Python 3.13+ with free threading enabled (PYTHON_GIL=0 or python3.13t).

Author: pycalphad development team
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Union
import itertools
import numpy as np
from pycalphad.core.equilibrium import equilibrium
from pycalphad.core.light_dataset import LightDataset


def equilibrium_threaded(
    dbf,
    comps: List[str],
    phases: Union[List[str], Dict],
    conditions_list: List[Dict],
    max_workers: Optional[int] = None,
    model=None,
    phase_records=None,
    verbose: bool = False,
    output=None,
    calc_opts=None,
    to_xarray: bool = True,
    parameters=None,
    solver=None,
    **kwargs
):
    """
    Calculate equilibrium for multiple independent condition sets using threads.

    This function leverages Python 3.13+ free threading to parallelize equilibrium
    calculations across different condition sets. Each condition set is solved
    independently in a separate thread without GIL contention.

    Parameters
    ----------
    dbf : Database
        Thermodynamic database containing the relevant parameters.
    comps : list of str
        Names of components to consider in the calculation.
    phases : list or dict
        Names of phases to consider in the calculation.
    conditions_list : list of dict
        List of condition dictionaries, each defining a separate equilibrium calculation.
        Each dictionary should contain StateVariables (e.g., v.T, v.P, v.X('AL'))
        and their corresponding values.
    max_workers : int, optional
        Maximum number of threads to use. Defaults to the number of CPUs.
        Set to 1 for serial execution.
    model : Model, dict, or sequence, optional
        Model class to use for each phase. Pre-building models is strongly
        recommended for performance.
    phase_records : PhaseRecordFactory, optional
        Pre-built phase records with 'GM' output. Strongly recommended for performance
        as it avoids rebuilding phase records for each calculation.
    verbose : bool, optional (default: False)
        Print details of calculations.
    output : str or list of str, optional
        Additional equilibrium model properties to compute (e.g., CPM, HM).
    calc_opts : dict, optional
        Keyword arguments to pass to `calculate`.
    to_xarray : bool, optional (default: True)
        Whether to return xarray Datasets (True) or EquilibriumResults (False).
    parameters : dict, optional
        Maps SymEngine Symbol to numbers, for overriding Database parameters.
    solver : SolverBase, optional
        Instance of a solver for calculating local equilibria.
    **kwargs
        Additional keyword arguments passed to equilibrium().

    Returns
    -------
    results : list
        List of equilibrium results (Dataset or EquilibriumResult) for each
        condition set. If a calculation fails, the corresponding entry will be None.

    Raises
    ------
    ImportError
        If concurrent.futures is not available.
    RuntimeError
        If running on Python < 3.13 without free threading support.

    Notes
    -----
    - Requires Python 3.13+ with free threading enabled (PYTHON_GIL=0 or python3.13t)
    - Pre-building models and phase_records is critical for performance
    - Each equilibrium calculation must be independent (no shared mutable state)
    - Failed calculations will be returned as None in the results list

    Performance Considerations
    --------------------------
    - Pre-build models and phase records once before calling this function
    - Use max_workers equal to the number of physical CPU cores for best performance
    - Memory bandwidth may become a bottleneck with >8 workers
    - Consider using calc_opts to reduce memory usage for large problems

    Examples
    --------
    Calculate equilibrium at multiple temperatures in parallel:

    >>> from pycalphad import Database, equilibrium_threaded
    >>> from pycalphad.core.utils import instantiate_models
    >>> from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
    >>> import pycalphad.variables as v
    >>>
    >>> # Load database and define system
    >>> dbf = Database('database.tdb')
    >>> comps = ['AL', 'FE', 'VA']
    >>> phases = ['LIQUID', 'FCC_A1', 'BCC_A2']
    >>>
    >>> # Pre-build models and phase records (important for performance!)
    >>> models = instantiate_models(dbf, comps, phases)
    >>> state_vars = sorted([v.T, v.P, v.N], key=str)
    >>> phase_records = PhaseRecordFactory(dbf, comps, state_vars, models)
    >>>
    >>> # Define multiple condition sets
    >>> conditions_list = [
    ...     {v.T: 1000, v.P: 101325, v.X('AL'): 0.3},
    ...     {v.T: 1200, v.P: 101325, v.X('AL'): 0.5},
    ...     {v.T: 1400, v.P: 101325, v.X('AL'): 0.7},
    ... ]
    >>>
    >>> # Calculate in parallel using 4 threads
    >>> results = equilibrium_threaded(
    ...     dbf, comps, phases, conditions_list,
    ...     max_workers=4,
    ...     model=models,
    ...     phase_records=phase_records
    ... )
    >>>
    >>> # Access results
    >>> for i, result in enumerate(results):
    ...     if result is not None:
    ...         print(f"Condition {i}: GM = {result.GM.values.flat[0]:.2f} J/mol")

    ESPEI-type workflow example:

    >>> # Simulate parameter optimization with many equilibrium calculations
    >>> temperatures = np.linspace(800, 2000, 50)
    >>> compositions = np.linspace(0.1, 0.9, 20)
    >>>
    >>> conditions_list = []
    >>> for T in temperatures:
    ...     for x_al in compositions:
    ...         conditions_list.append({
    ...             v.T: float(T),
    ...             v.P: 101325,
    ...             v.X('AL'): float(x_al)
    ...         })
    >>>
    >>> # 1000 equilibrium calculations in parallel
    >>> results = equilibrium_threaded(
    ...     dbf, comps, phases, conditions_list,
    ...     max_workers=8,
    ...     model=models,
    ...     phase_records=phase_records
    ... )

    See Also
    --------
    equilibrium : Serial equilibrium calculation
    equilibrium_batch_threaded : Parallel calculation with batched conditions
    """

    # Check Python version
    if sys.version_info < (3, 11):
        raise RuntimeError(
            "equilibrium_threaded requires Python 3.11 or higher. "
            f"Current version: {sys.version_info.major}.{sys.version_info.minor}"
        )

    # Warn if GIL is enabled (Python 3.13+)
    if sys.version_info >= (3, 13):
        try:
            if sys._is_gil_enabled():
                import warnings
                warnings.warn(
                    "GIL is enabled. For best performance with free threading, "
                    "run with PYTHON_GIL=0 or use python3.13t build.",
                    RuntimeWarning
                )
        except AttributeError:
            pass

    def _calc_single(idx_conds):
        """Helper to calculate single equilibrium point."""
        idx, conds = idx_conds
        try:
            result = equilibrium(
                dbf, comps, phases, conds,
                output=output,
                model=model,
                verbose=verbose,
                calc_opts=calc_opts,
                to_xarray=to_xarray,
                parameters=parameters,
                solver=solver,
                phase_records=phase_records,
                **kwargs
            )
            return (idx, result, None)
        except Exception as e:
            if verbose:
                print(f"Error in condition {idx}: {e}")
            return (idx, None, str(e))

    # Special case: single condition (no parallelization needed)
    if len(conditions_list) == 1:
        result = equilibrium(
            dbf, comps, phases, conditions_list[0],
            output=output,
            model=model,
            verbose=verbose,
            calc_opts=calc_opts,
            to_xarray=to_xarray,
            parameters=parameters,
            solver=solver,
            phase_records=phase_records,
            **kwargs
        )
        return [result]

    # Use ThreadPoolExecutor for parallel execution
    indexed_conditions = list(enumerate(conditions_list))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results_with_indices = list(executor.map(_calc_single, indexed_conditions))

    # Sort by original index and extract results
    results_with_indices.sort(key=lambda x: x[0])
    results = []
    error_count = 0

    for idx, result, error in results_with_indices:
        if error is not None:
            error_count += 1
        results.append(result)

    if error_count > 0 and not verbose:
        import warnings
        warnings.warn(
            f"{error_count} out of {len(conditions_list)} calculations failed. "
            "Set verbose=True for details.",
            RuntimeWarning
        )

    return results


def equilibrium_batch_threaded(
    dbf,
    comps: List[str],
    phases: Union[List[str], Dict],
    conditions: Dict,
    max_workers: Optional[int] = None,
    model=None,
    phase_records=None,
    verbose: bool = False,
    **kwargs
):
    """
    Calculate equilibrium with batched conditions using thread parallelization.

    This function takes a single conditions dictionary with array-valued entries
    and splits the calculation across threads. It's useful when you have a single
    call to equilibrium() with batched conditions that would benefit from parallelization.

    Parameters
    ----------
    dbf : Database
        Thermodynamic database
    comps : list of str
        Component names
    phases : list or dict
        Phase names
    conditions : dict
        Conditions dictionary with scalar or array values.
        Array-valued conditions will be parallelized.
        Example: {v.T: [1000, 1200, 1400], v.P: 101325, v.X('AL'): 0.5}
    max_workers : int, optional
        Maximum number of threads
    model : Model or dict, optional
        Phase models (pre-building recommended)
    phase_records : PhaseRecordFactory, optional
        Pre-built phase records (strongly recommended)
    verbose : bool, optional
        Print calculation details
    **kwargs
        Additional arguments passed to equilibrium()

    Returns
    -------
    result : list
        List of equilibrium results for each condition combination.
        If there are multiple array-valued conditions, returns results for
        the Cartesian product of all arrays.

    Notes
    -----
    - If no array-valued conditions are provided, calls equilibrium() directly
    - Multiple array-valued conditions are expanded using Cartesian product
    - Returns a list of results rather than a single combined dataset
    - Consider using equilibrium_threaded() directly for more control

    Examples
    --------
    >>> conditions = {v.T: [1000, 1200, 1400], v.P: 101325, v.X('AL'): 0.5}
    >>> results = equilibrium_batch_threaded(
    ...     dbf, comps, phases, conditions,
    ...     max_workers=3, model=models, phase_records=phase_records
    ... )
    >>> # Returns list of 3 results, one for each temperature

    Multiple array conditions (Cartesian product):

    >>> conditions = {
    ...     v.T: [1000, 1200],
    ...     v.P: 101325,
    ...     v.X('AL'): [0.3, 0.5, 0.7]
    ... }
    >>> results = equilibrium_batch_threaded(
    ...     dbf, comps, phases, conditions, max_workers=4
    ... )
    >>> # Returns list of 6 results (2 temps × 3 compositions)
    """

    # Separate array and scalar conditions
    array_conds = {k: v for k, v in conditions.items()
                   if hasattr(v, '__len__') and not isinstance(v, str)}
    scalar_conds = {k: v for k, v in conditions.items()
                    if not hasattr(v, '__len__') or isinstance(v, str)}

    # If no arrays, just call equilibrium directly
    if not array_conds:
        result = equilibrium(
            dbf, comps, phases, conditions,
            model=model,
            phase_records=phase_records,
            verbose=verbose,
            **kwargs
        )
        return [result]

    # Generate all combinations (Cartesian product)
    keys = list(array_conds.keys())
    values = [array_conds[k] for k in keys]

    conditions_list = []
    for combo in itertools.product(*values):
        cond = scalar_conds.copy()
        for k, v in zip(keys, combo):
            cond[k] = v
        conditions_list.append(cond)

    # Calculate in parallel
    results = equilibrium_threaded(
        dbf, comps, phases, conditions_list,
        max_workers=max_workers,
        model=model,
        phase_records=phase_records,
        verbose=verbose,
        **kwargs
    )

    return results


__all__ = ['equilibrium_threaded', 'equilibrium_batch_threaded']
