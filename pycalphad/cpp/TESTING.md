# Testing Framework for C++ Migration

This document describes the testing strategy for the Cython-to-C++ migration.

## Overview

The migration must preserve **exact** numerical behavior. We verify this through:
1. **Integration tests** - Existing test suite must pass without changes
2. **Unit tests** - Direct comparison of C++ vs Cython outputs
3. **Performance benchmarks** - C++ should match or exceed Cython performance

## Cython Modules and Test Coverage

### 1. hyperplane.pyx (~339 lines)
**Purpose**: Convex hull and hyperplane calculations for phase equilibria

**Key Functions**:
- `solve()` - LAPACK dgesv wrapper for linear systems
- `prodsum()` - Matrix-vector operations
- `argmin()`, `argmax()`, `_min()` - Array utilities
- `hyperplane_coefficients()` - Compute hyperplane coefficients
- `intersecting_point()` - Find hyperplane intersections
- `simplex_fractions()` - Compute simplex fractions
- `hyperplane()` - Main algorithm for finding tangent hyperplane

**Test Coverage**:
- Indirectly tested through `test_equilibrium.py`
- All equilibrium calculations use hyperplane for convex hull
- ~70 equilibrium tests exercise this code

**Edge Cases to Verify**:
- Singular matrices (info != 0 from LAPACK)
- Degenerate simplices (zero-dimensional)
- Fixed chemical potential constraints
- Numerical tolerance handling

---

### 2. phase_rec.pyx (~396 lines)
**Purpose**: PhaseRecord and FastFunction - interface to thermodynamic models

**Key Classes**:
- `FastFunction` - Wrapper for SymEngine function pointers
- `FastFunctionFactory` - Caching and property access
- `PhaseRecord` - Phase thermodynamic data and callables

**Key Methods**:
- `obj()`, `formulaobj()` - Energy calculations
- `formulagrad()`, `formulahess()` - Gradient/Hessian calculations
- `mass_obj()`, `formulamole_obj()` - Composition calculations
- `internal_cons_func/jac/hess()` - Internal constraints

**Test Coverage**:
- Used by all `test_calculate.py` tests (~50 tests)
- Used by all `test_equilibrium.py` tests (~70 tests)
- Direct usage in `test_property_framework.py`

**Edge Cases to Verify**:
- Vectorized vs non-vectorized calls
- Parameter arrays of various sizes
- Property caching behavior
- Memory management of function pointers

---

### 3. composition_set.pyx (~73 lines)
**Purpose**: State of a phase during equilibrium calculation

**Key Class**: `CompositionSet`
**Key Methods**:
- `__cinit__()` - Initialization
- `update()` - Update phase state (site fractions, amount, state variables)
- `set_local_conditions()` - Set phase-local constraints
- `__deepcopy__()` - Deep copy for solver iterations

**Test Coverage**:
- Used internally by equilibrium solver
- Direct imports in:
  - `test_mapping_strategy.py`
  - `test_property_framework.py`
  - `test_workspace.py`

**Edge Cases to Verify**:
- Fixed vs free composition sets
- Zero phase amounts
- Local constraint handling
- Deep copy correctness

---

### 4. eqsolver.pyx (~298 lines)
**Purpose**: Phase addition/removal logic for equilibrium solver

**Key Functions**:
- `add_new_phases()` - Add phase with largest driving force
- `add_nearly_stable()` - Add metastable phases near stability
- `argmax()` - Find maximum value index
- `_solve_eq_at_conditions()` - Main equilibrium solver loop

**Test Coverage**:
- Core of all equilibrium calculations
- Tested by all `test_equilibrium.py` tests
- Tests specific phase addition/removal scenarios

**Edge Cases to Verify**:
- Phase distinctness checks (composition tolerance)
- Driving force calculations
- Phase removal from previously stable set
- Metastability detection

---

### 5. minimizer.pyx (~1098 lines)
**Purpose**: Nonlinear optimization for equilibrium

**Key Classes**:
- `SystemSpecification` - Problem definition
- `CompsetState` - State for each composition set
- `SystemState` - Overall system state

**Key Functions**:
- `lstsq()`, `invert_matrix()` - Linear algebra
- `compute_phase_matrix()` - Hessian and constraint matrices
- `fill_equilibrium_system()` - Build linear system
- `solve_state()` - Solve for new state
- `advance_state()` - Update with step size control
- `remove_and_consolidate_phases()` - Phase cleanup
- `change_phases()` - Add/remove phases based on driving forces

**Test Coverage**:
- Heart of the equilibrium solver
- All equilibrium tests depend on this
- Tests cover:
  - Miscibility gaps
  - Stoichiometric phases
  - Chemical potential conditions
  - Fixed phase amounts
  - Gibbs phase rule enforcement

**Edge Cases to Verify**:
- Singular matrices in linear solve
- Step size reduction for bounds
- Phase consolidation in miscibility gaps
- Gibbs phase rule violations
- NaN handling in LAPACK routines

---

## Test Strategy

### Phase 1: Baseline Establishment (Current)
1. Document all existing test coverage ✓
2. Run full test suite and capture baseline results
3. Create performance benchmarks for each module

### Phase 2: Per-Module Migration
For each module being migrated:

1. **Before Migration**:
   - Run module-specific tests
   - Benchmark performance
   - Document any known numerical quirks

2. **During Migration**:
   - Implement C++ version
   - Create Cython bindings
   - Add C++ unit tests (optional but recommended)

3. **After Migration**:
   - Run full test suite - must pass 100%
   - Compare performance benchmarks
   - Verify bit-for-bit equivalence on test cases

### Phase 3: Full Integration Testing
After all modules migrated:
1. Run complete test suite on all platforms
2. Performance regression testing
3. Memory leak testing (valgrind on Linux)
4. Stress testing with large systems

---

## Performance Benchmarks

### Benchmark Suite Location
`pycalphad/tests/benchmarks/`

### Key Benchmarks

1. **hyperplane_benchmark.py**
   - Small system (3 components, 10 points)
   - Medium system (5 components, 100 points)
   - Large system (10 components, 1000 points)
   - Measure: iterations, time, memory

2. **phase_rec_benchmark.py**
   - Property calculations (vectorized vs non-vectorized)
   - Gradient/Hessian calculations
   - Mass fraction calculations
   - Measure: calls/second, memory usage

3. **minimizer_benchmark.py**
   - Single-phase equilibrium
   - Two-phase equilibrium with miscibility gap
   - Multi-phase equilibrium (5+ phases)
   - Measure: iterations, time, convergence rate

4. **full_equilibrium_benchmark.py**
   - Representative real-world systems
   - Various temperature/composition ranges
   - Different numbers of components (2-5)
   - Measure: total time, phases found, accuracy

### Running Benchmarks

```bash
# Run all benchmarks
uv run python -m pytest pycalphad/tests/benchmarks/ --benchmark-only

# Run specific benchmark
uv run python pycalphad/tests/benchmarks/hyperplane_benchmark.py

# Compare before/after
./scripts/compare_benchmarks.sh baseline.json current.json
```

---

## Verification Utilities

### Test Helpers (`pycalphad/tests/test_utils_cpp.py`)

Utilities for comparing C++ and Cython outputs:

```python
def assert_arrays_equal(arr1, arr2, rtol=1e-15, atol=1e-15):
    """Assert two arrays are numerically equivalent"""

def assert_solver_states_equal(state1, state2):
    """Compare two SystemState objects"""

def assert_phase_records_equal(pr1, pr2):
    """Compare two PhaseRecord objects"""

def benchmark_function(func, *args, iterations=1000):
    """Benchmark a function call"""
```

### Continuous Integration

Tests must pass on all platforms:
- **Linux**: Ubuntu 22.04, 24.04
- **macOS**: Latest (arm64 and x86_64)
- **Windows**: Latest

Python versions: 3.11, 3.12, 3.13

---

## Success Criteria

A module migration is complete when:

1. ✅ All existing tests pass without modification
2. ✅ Performance is within 95-105% of Cython baseline
3. ✅ No memory leaks detected
4. ✅ Builds successfully on all platforms
5. ✅ Code review approved
6. ✅ Documentation updated

---

## Known Numerical Considerations

### Floating Point Tolerance
- Comparisons use `rtol=1e-15, atol=1e-15` by default
- Some LAPACK operations may have platform-specific rounding
- Documented cases where exact bit-equivalence is not expected

### LAPACK Behavior
- Singular matrix detection: `info != 0`
- Sentinel value on error: `-1e19`
- NaN handling: explicitly checked before LAPACK calls

### Optimization Tolerance
- `COMP_DIFFERENCE_TOL = 1e-8` for phase distinction
- `MIN_SITE_FRACTION = 1e-12` for site fraction bounds
- Step size control in `advance_state()`

---

## Debugging Failed Tests

When a test fails after migration:

1. **Isolate the failure**:
   ```bash
   uv run pytest -xvs pycalphad/tests/test_equilibrium.py::test_name
   ```

2. **Compare intermediate values**:
   - Add debug prints in C++ code
   - Compare Cython vs C++ at each step
   - Check for NaN/Inf propagation

3. **Check LAPACK calls**:
   - Verify matrix layout (C vs Fortran order)
   - Check `info` return values
   - Verify workspace sizes

4. **Numerical issues**:
   - Try different tolerance values
   - Check for integer overflow in size calculations
   - Verify proper floating point precision

---

## Contact

For questions about testing:
- Check existing test files in `pycalphad/tests/`
- Review GitHub issues with `testing` label
- See migration tracking issue
