# Python Free Threading Implementation Plan for pycalphad

## Executive Summary

This document outlines a plan to leverage Python 3.13's free threading capabilities (PEP 703) to parallelize equilibrium calculations in pycalphad. The primary target is ESPEI-type workflows that require computing many independent equilibrium calculations across different conditions or parameter sets.

## Background: Python Free Threading (PEP 703)

Python 3.13 introduces experimental support for running Python without the Global Interpreter Lock (GIL), enabled via the `PYTHON_GIL=0` environment variable or `python3.13t` build. Key features:

- **True parallelism**: Multiple threads can execute Python bytecode simultaneously
- **Backward compatible**: Existing code works without modification
- **Thread-safe C extensions**: NumPy, SciPy, and other scientific libraries are being updated for GIL-free operation
- **Performance gains**: CPU-bound tasks see near-linear scaling with thread count

## Current Architecture Analysis

### Equilibrium Calculation Flow

1. **Entry Point**: `equilibrium()` in `pycalphad/core/equilibrium.py:15-90`
   - Creates a `Workspace` object
   - Calls `Workspace.recompute()` to compute equilibrium

2. **Computation Engine**: `Workspace.recompute()` in `pycalphad/core/workspace.py:341-365`
   - Calls `calculate()` to generate energy grids
   - Calls `_solve_eq_at_conditions()` to solve equilibrium at each condition point

3. **Sequential Bottleneck**: `_solve_eq_at_conditions()` in `pycalphad/core/eqsolver.pyx:131-297`
   ```python
   it = np.nditer(prop_GM_values, flags=['multi_index'])
   while not it.finished:
       # Solve equilibrium at current condition point
       # Lines 184-296: ~100+ lines of sequential processing
       it.iternext()
   ```

   **Key Issue**: Each condition point is processed sequentially in a single thread, even though calculations are independent.

### Existing Parallelization

- **None in core equilibrium solver**: All condition points processed serially
- **Thread-safe caching**: `pycalphad/core/cache.py:83-94` uses `threading.RLock()`
- **Vectorized operations**: NumPy operations are internally parallelized, but high-level iteration is serial

### Typical ESPEI Workflow Pattern

```python
# 1. Build models and phase records once (expensive)
models = instantiate_models(dbf, comps, phases)
phase_records = PhaseRecordFactory(dbf, comps, state_variables, models)

# 2. Compute many equilibria (ESPEI optimization loop)
for experiment in experiments:
    for condition in experiment.conditions:
        eq = equilibrium(dbf, comps, phases, condition,
                        model=models, phase_records=phase_records)
        # Extract and compare properties
```

**Parallelization Opportunity**: The inner loop iterations are independent and embarrassingly parallel.

## Proposed Implementation

### Phase 1: High-Level Thread Pool Approach (Minimal Changes)

Create a new public API function that parallelizes multiple equilibrium calculations:

**File**: `pycalphad/core/parallel_equilibrium.py` (new)

```python
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional
import numpy as np
from pycalphad.core.equilibrium import equilibrium
from pycalphad.core.light_dataset import LightDataset


def equilibrium_threaded(
    dbf,
    comps: List[str],
    phases: List[str],
    conditions_list: List[Dict],
    max_workers: Optional[int] = None,
    model=None,
    phase_records=None,
    verbose: bool = False,
    **kwargs
) -> List[LightDataset]:
    """
    Calculate equilibrium for multiple independent condition sets using threads.

    This function leverages Python 3.13+ free threading to parallelize equilibrium
    calculations across different condition sets. Each condition set is solved
    independently in a separate thread without GIL contention.

    Parameters
    ----------
    dbf : Database
        Thermodynamic database
    comps : list of str
        Component names
    phases : list of str
        Phase names
    conditions_list : list of dict
        List of condition dictionaries, each defining a separate equilibrium calculation
    max_workers : int, optional
        Maximum number of threads. Defaults to CPU count.
    model : Model or dict, optional
        Phase models (should be pre-built for best performance)
    phase_records : PhaseRecordFactory, optional
        Pre-built phase records (strongly recommended for performance)
    verbose : bool
        Print calculation details
    **kwargs
        Additional arguments passed to equilibrium()

    Returns
    -------
    results : list of LightDataset
        Equilibrium results for each condition set

    Notes
    -----
    - Requires Python 3.13+ with free threading enabled (PYTHON_GIL=0 or python3.13t)
    - Pre-building models and phase_records is critical for performance
    - Each equilibrium calculation must be independent (no shared state)

    Examples
    --------
    >>> from pycalphad import Database, equilibrium_threaded
    >>> from pycalphad.core.utils import instantiate_models
    >>> from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
    >>> import pycalphad.variables as v
    >>>
    >>> dbf = Database('database.tdb')
    >>> comps = ['AL', 'FE', 'VA']
    >>> phases = ['LIQUID', 'FCC_A1', 'BCC_A2']
    >>>
    >>> # Pre-build models and phase records
    >>> models = instantiate_models(dbf, comps, phases)
    >>> state_vars = [v.T, v.P, v.N]
    >>> phase_records = PhaseRecordFactory(dbf, comps, state_vars, models)
    >>>
    >>> # Define multiple condition sets
    >>> conditions_list = [
    ...     {v.T: 1000, v.P: 101325, v.X('AL'): 0.3},
    ...     {v.T: 1200, v.P: 101325, v.X('AL'): 0.5},
    ...     {v.T: 1400, v.P: 101325, v.X('AL'): 0.7},
    ... ]
    >>>
    >>> # Calculate in parallel
    >>> results = equilibrium_threaded(dbf, comps, phases, conditions_list,
    ...                                model=models, phase_records=phase_records)
    """

    def _calc_single(conds):
        """Helper to calculate single equilibrium point."""
        return equilibrium(
            dbf, comps, phases, conds,
            model=model,
            phase_records=phase_records,
            verbose=verbose,
            **kwargs
        )

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_calc_single, conditions_list))

    return results


def equilibrium_batch_threaded(
    dbf,
    comps: List[str],
    phases: List[str],
    conditions: Dict,
    max_workers: Optional[int] = None,
    model=None,
    phase_records=None,
    verbose: bool = False,
    **kwargs
) -> LightDataset:
    """
    Calculate equilibrium with batched conditions using thread parallelization.

    This function splits batched condition arrays into chunks and processes them
    in parallel using free threading, then recombines the results.

    Parameters
    ----------
    dbf : Database
        Thermodynamic database
    comps : list of str
        Component names
    phases : list of str
        Phase names
    conditions : dict
        Conditions with array values (e.g., {v.T: [1000, 1200, 1400], v.X('AL'): 0.5})
    max_workers : int, optional
        Maximum number of threads
    model : Model or dict, optional
        Phase models
    phase_records : PhaseRecordFactory, optional
        Pre-built phase records
    verbose : bool
        Print calculation details
    **kwargs
        Additional arguments

    Returns
    -------
    result : LightDataset
        Combined equilibrium results

    Notes
    -----
    This function is useful when you have a single call to equilibrium() with
    batched conditions that would benefit from parallelization.
    """
    import itertools

    # Find all array-valued conditions
    array_conds = {k: v for k, v in conditions.items() if hasattr(v, '__len__')}
    scalar_conds = {k: v for k, v in conditions.items() if not hasattr(v, '__len__')}

    if not array_conds:
        # No arrays, just call equilibrium directly
        return equilibrium(dbf, comps, phases, conditions,
                          model=model, phase_records=phase_records,
                          verbose=verbose, **kwargs)

    # Generate all combinations
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

    # TODO: Combine results back into single dataset with proper dimensions
    # For now, return list of results
    return results
```

**Advantages**:
- Minimal code changes (new file only)
- No modifications to existing API or Cython code
- Easy to test and benchmark
- Backward compatible (falls back to serial execution without free threading)

**Limitations**:
- Cannot parallelize within a single `equilibrium()` call with batched conditions
- Some overhead from creating/destroying thread pools
- Each thread duplicates some workspace state

### Phase 2: Low-Level Parallelization (Advanced)

Modify `_solve_eq_at_conditions()` to parallelize the condition iteration loop internally.

**File**: `pycalphad/core/eqsolver.pyx` (modify)

```python
# Add new function for parallel execution
def _solve_eq_at_conditions_parallel(properties, phase_records, grid, conds_keys,
                                     state_variables, verbose, solver=None, max_workers=None):
    """
    Parallel version of _solve_eq_at_conditions using thread pool.
    """
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np

    # Get shape and create flat index list
    shape = properties.GM.shape
    total_points = np.prod(shape)
    indices = [np.unravel_index(i, shape) for i in range(total_points)]

    def _solve_single_point(idx):
        """Solve equilibrium at a single condition point."""
        # Extract conditions for this point
        cur_conds = OrderedDict(zip(conds_keys,
                                   [np.asarray(properties.coords[str(b)][a], dtype=np.float64)
                                    for a, b in zip(idx, conds_keys)]))

        # Solve equilibrium (extracted from main loop)
        # ... implementation details ...

        return result_dict

    # Execute in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_solve_single_point, indices))

    # Populate properties arrays with results
    for idx, result in zip(indices, results):
        properties.MU[idx] = result['MU']
        properties.NP[idx] = result['NP']
        # ... etc ...

    return properties
```

**Advantages**:
- Transparently parallelizes existing `equilibrium()` API
- No user code changes required
- Parallelizes within batched condition arrays

**Challenges**:
- Significant Cython code refactoring required
- Need to ensure thread safety of all data structures
- More complex to test and debug
- Potential GIL-related issues with Cython code

**Recommendation**: Start with Phase 1, then pursue Phase 2 if benchmarks show significant benefit.

## Test Plan

### 1. Environment Setup

```bash
# Install Python 3.13 with free threading
conda create -n pycalphad-freethreading python=3.13
conda activate pycalphad-freethreading

# Install pycalphad in development mode
pip install -e .

# Verify free threading is available
python -c "import sys; print(sys._is_gil_enabled())"
# Should print: False
```

### 2. Unit Tests

**File**: `pycalphad/tests/test_parallel_equilibrium.py` (new)

```python
import pytest
import sys
import numpy as np
from numpy.testing import assert_allclose
from pycalphad import Database
from pycalphad.core.parallel_equilibrium import equilibrium_threaded
from pycalphad.core.equilibrium import equilibrium
from pycalphad.core.utils import instantiate_models
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
import pycalphad.variables as v


@pytest.mark.skipif(sys.version_info < (3, 13), reason="Requires Python 3.13+")
class TestParallelEquilibrium:

    def test_threaded_matches_serial(self, load_database):
        """Threaded equilibrium should produce identical results to serial."""
        dbf = Database('alfe.tdb')
        comps = ['AL', 'FE', 'VA']
        phases = ['LIQUID', 'FCC_A1', 'BCC_A2']

        conditions_list = [
            {v.T: 1000, v.P: 101325, v.X('AL'): 0.3},
            {v.T: 1200, v.P: 101325, v.X('AL'): 0.5},
            {v.T: 1400, v.P: 101325, v.X('AL'): 0.7},
        ]

        # Pre-build models and phase records
        models = instantiate_models(dbf, comps, phases)
        state_vars = sorted([v.T, v.P, v.N], key=str)
        phase_records = PhaseRecordFactory(dbf, comps, state_vars, models)

        # Serial execution
        serial_results = [
            equilibrium(dbf, comps, phases, cond,
                       model=models, phase_records=phase_records)
            for cond in conditions_list
        ]

        # Parallel execution
        parallel_results = equilibrium_threaded(
            dbf, comps, phases, conditions_list,
            model=models, phase_records=phase_records,
            max_workers=3
        )

        # Compare results
        for serial, parallel in zip(serial_results, parallel_results):
            assert_allclose(serial.GM.values, parallel.GM.values, rtol=1e-6)
            assert_allclose(serial.MU.values, parallel.MU.values, rtol=1e-6)
            assert_allclose(serial.NP.values, parallel.NP.values, rtol=1e-6)

    def test_single_worker_matches_serial(self):
        """Single worker should match serial execution exactly."""
        # Test that max_workers=1 produces identical results
        pass

    def test_thread_safety_stress(self):
        """Stress test with many threads and conditions."""
        # Test with 100+ conditions and multiple workers
        pass

    def test_error_handling(self):
        """Errors in one thread should not affect others."""
        pass
```

### 3. Integration Test Script (ESPEI-like Workflow)

**File**: `test_espei_parallel.py` (see separate file below)

### 4. Benchmark Tests

**File**: `benchmarks/benchmark_parallel_equilibrium.py` (new)

```python
import time
import numpy as np
from pycalphad import Database
from pycalphad.core.parallel_equilibrium import equilibrium_threaded
from pycalphad.core.equilibrium import equilibrium
from pycalphad.core.utils import instantiate_models
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
import pycalphad.variables as v


def benchmark_parallel_scaling(n_conditions=100, max_workers_list=[1, 2, 4, 8]):
    """Benchmark parallel equilibrium calculation with different thread counts."""
    dbf = Database('alfe.tdb')
    comps = ['AL', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2']

    # Pre-build
    models = instantiate_models(dbf, comps, phases)
    state_vars = sorted([v.T, v.P, v.N], key=str)
    phase_records = PhaseRecordFactory(dbf, comps, state_vars, models)

    # Generate conditions
    temperatures = np.linspace(800, 2000, n_conditions)
    conditions_list = [
        {v.T: T, v.P: 101325, v.X('AL'): 0.5}
        for T in temperatures
    ]

    results = {}

    # Benchmark serial
    start = time.perf_counter()
    for cond in conditions_list:
        equilibrium(dbf, comps, phases, cond,
                   model=models, phase_records=phase_records)
    serial_time = time.perf_counter() - start
    results['serial'] = serial_time

    # Benchmark parallel with different worker counts
    for n_workers in max_workers_list:
        start = time.perf_counter()
        equilibrium_threaded(dbf, comps, phases, conditions_list,
                           max_workers=n_workers,
                           model=models, phase_records=phase_records)
        parallel_time = time.perf_counter() - start
        results[f'{n_workers}_workers'] = parallel_time
        speedup = serial_time / parallel_time
        efficiency = speedup / n_workers * 100

        print(f"{n_workers} workers: {parallel_time:.3f}s "
              f"(speedup: {speedup:.2f}x, efficiency: {efficiency:.1f}%)")

    return results


if __name__ == '__main__':
    print("Benchmarking parallel equilibrium calculations...")
    benchmark_parallel_scaling(n_conditions=100)
```

**Expected Performance**:
- **With GIL** (Python 3.11-3.12): Minimal speedup, possibly slower due to overhead
- **Without GIL** (Python 3.13+):
  - 2 workers: ~1.8x speedup
  - 4 workers: ~3.5x speedup
  - 8 workers: ~6-7x speedup (diminishing returns due to memory bandwidth)

### 5. Validation Strategy

1. **Correctness**: All threaded results must match serial results within numerical tolerance
2. **Thread Safety**: No data races or corruption under concurrent access
3. **Performance**: Demonstrate near-linear scaling up to CPU core count
4. **Robustness**: Handle edge cases (empty phases, failed convergence) correctly

## Implementation Roadmap

### Milestone 1: Proof of Concept (2-3 weeks)
- [ ] Create `parallel_equilibrium.py` with `equilibrium_threaded()` function
- [ ] Set up Python 3.13 testing environment
- [ ] Write basic unit tests
- [ ] Create ESPEI-like integration test script
- [ ] Benchmark on small problems (10-100 conditions)

### Milestone 2: Testing & Validation (2-3 weeks)
- [ ] Comprehensive unit test suite
- [ ] Stress testing with 1000+ conditions
- [ ] Validation against serial execution
- [ ] Thread safety analysis
- [ ] Performance profiling

### Milestone 3: Optimization (2-4 weeks)
- [ ] Minimize per-thread overhead
- [ ] Optimize data structure sharing
- [ ] Tune chunk sizes for batch operations
- [ ] Investigate Phase 2 (low-level parallelization)

### Milestone 4: Documentation & Release (1-2 weeks)
- [ ] API documentation
- [ ] User guide with examples
- [ ] Performance benchmarks
- [ ] Migration guide for ESPEI users
- [ ] Release notes

## Risks & Mitigation

### Technical Risks

1. **NumPy/SciPy Thread Safety**
   - **Risk**: Underlying libraries may not be fully thread-safe without GIL
   - **Mitigation**: Test thoroughly; file upstream bug reports if needed
   - **Status**: NumPy 2.0+ is designed for free threading

2. **Cython GIL Dependencies**
   - **Risk**: Cython code may have implicit GIL dependencies
   - **Mitigation**: Review and test all Cython modules; add `nogil` annotations where possible
   - **Status**: Most pycalphad Cython code is computational (should be GIL-free)

3. **Memory Bandwidth Bottleneck**
   - **Risk**: High thread counts may saturate memory bandwidth before CPU
   - **Mitigation**: Benchmark to find optimal thread count; document limitations
   - **Expected**: 4-8 threads optimal for most systems

### Process Risks

1. **Python 3.13 Adoption**
   - **Risk**: Users may not upgrade to Python 3.13 quickly
   - **Mitigation**: Maintain backward compatibility; make parallel features optional
   - **Status**: pycalphad already supports Python 3.13 (pyproject.toml:30)

2. **Maintenance Burden**
   - **Risk**: Additional code paths increase maintenance complexity
   - **Mitigation**: Keep parallel code simple; reuse existing functions; comprehensive tests

## Success Metrics

1. **Performance**:
   - 3-4x speedup with 4 threads on typical ESPEI workload
   - 6-8x speedup with 8 threads on CPU-bound calculations

2. **Correctness**:
   - 100% pass rate on validation tests
   - Zero numerical differences from serial execution (within tolerance)

3. **Usability**:
   - API that requires minimal code changes for existing users
   - Clear documentation with examples

4. **Adoption**:
   - Integration into ESPEI within 6 months
   - Positive feedback from users in issues/discussions

## References

- [PEP 703: Making the Global Interpreter Lock Optional](https://peps.python.org/pep-0703/)
- [Python 3.13 Release Notes](https://docs.python.org/3.13/whatsnew/3.13.html)
- [NumPy Free Threading Support](https://numpy.org/devdocs/dev/depending_on_numpy.html#free-threading-support)
- [ESPEI Documentation](https://espei.org/)
- pycalphad documentation: https://pycalphad.org/

## Appendix A: Alternative Approaches Considered

### Multiprocessing
- **Pros**: Works on all Python versions; truly isolated execution
- **Cons**: High overhead; serialization costs; no shared memory
- **Decision**: Free threading is superior for CPU-bound scientific computing

### Dask
- **Pros**: Mature ecosystem; works with xarray
- **Cons**: Complex dependency; overkill for simple parallelization
- **Decision**: Too heavyweight for this use case

### Joblib
- **Pros**: Simple API; good for embarrassingly parallel tasks
- **Cons**: Uses multiprocessing (serialization overhead)
- **Decision**: Free threading provides better performance

### OpenMP (via Cython)
- **Pros**: Very fast; low overhead
- **Cons**: Requires compiling with OpenMP; platform-dependent
- **Decision**: Could complement free threading, but not replace it

## Appendix B: ESPEI Integration

ESPEI's `calculate_thermochemical_error` and `calculate_zpf_error` functions typically call `equilibrium()` in a loop. Integration would look like:

```python
# In ESPEI's error calculation
def calculate_zpf_error_parallel(dbf, comps, phases, datasets, params,
                                 max_workers=None):
    """Calculate ZPF error with parallel equilibrium calculations."""
    from pycalphad.core.parallel_equilibrium import equilibrium_threaded

    # Build models/phase records once
    models = instantiate_models(dbf, comps, phases, params)
    phase_records = PhaseRecordFactory(dbf, comps, state_vars, models)

    # Collect all conditions from all datasets
    conditions_list = []
    for dataset in datasets:
        for condition in dataset.conditions:
            conditions_list.append(condition)

    # Parallel calculation
    results = equilibrium_threaded(dbf, comps, phases, conditions_list,
                                   max_workers=max_workers,
                                   model=models,
                                   phase_records=phase_records)

    # Calculate errors from results
    errors = [calc_error(result, dataset)
              for result, dataset in zip(results, datasets)]

    return np.sum(errors)
```
