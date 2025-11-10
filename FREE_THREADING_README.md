# Python Free Threading for pycalphad

This directory contains the implementation and test plan for using Python 3.13's free threading features to parallelize equilibrium calculations in pycalphad.

## Overview

Python 3.13 introduces experimental support for running without the Global Interpreter Lock (GIL), enabling true thread-based parallelism. This feature is particularly beneficial for CPU-bound scientific computing tasks like equilibrium calculations, which can now be parallelized across multiple CPU cores using threads instead of processes.

## Files in This Implementation

1. **FREE_THREADING_IMPLEMENTATION_PLAN.md**
   - Comprehensive implementation plan and design document
   - Architecture analysis and proposed approach
   - Test plan and success metrics
   - Performance expectations and benchmarks

2. **pycalphad/core/parallel_equilibrium.py**
   - Implementation of `equilibrium_threaded()` function
   - Implementation of `equilibrium_batch_threaded()` function
   - Documented API for parallel equilibrium calculations

3. **test_espei_parallel.py**
   - Comprehensive test suite for parallel equilibrium
   - ESPEI-type workflow demonstration
   - Performance benchmarks
   - Thread safety stress tests

## Quick Start

### Requirements

- Python 3.13+ (free threading is experimental in 3.13)
- pycalphad installed in development mode
- NumPy 2.0+ (for thread-safe operations)

### Installation

```bash
# Create Python 3.13 environment
conda create -n pycalphad-freethreading python=3.13
conda activate pycalphad-freethreading

# Install pycalphad in development mode
cd pycalphad
pip install -e .
```

### Running with Free Threading

```bash
# Enable free threading (disable GIL)
export PYTHON_GIL=0

# Or use the free-threaded Python build
python3.13t script.py

# Run the test suite
python test_espei_parallel.py
```

## Usage Example

### Basic Usage

```python
from pycalphad import Database, equilibrium_threaded
from pycalphad.core.utils import instantiate_models
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
import pycalphad.variables as v

# Load database
dbf = Database('my_database.tdb')
comps = ['AL', 'FE', 'VA']
phases = ['LIQUID', 'FCC_A1', 'BCC_A2']

# Pre-build models and phase records (important for performance!)
models = instantiate_models(dbf, comps, phases)
state_vars = sorted([v.T, v.P, v.N], key=str)
phase_records = PhaseRecordFactory(dbf, comps, state_vars, models)

# Define multiple condition sets
conditions_list = [
    {v.T: 1000, v.P: 101325, v.X('AL'): 0.3},
    {v.T: 1200, v.P: 101325, v.X('AL'): 0.5},
    {v.T: 1400, v.P: 101325, v.X('AL'): 0.7},
]

# Calculate in parallel using 4 threads
results = equilibrium_threaded(
    dbf, comps, phases, conditions_list,
    max_workers=4,
    model=models,
    phase_records=phase_records
)

# Access results
for i, result in enumerate(results):
    if result is not None:
        print(f"Condition {i}: GM = {result.GM.values.flat[0]:.2f} J/mol")
```

### ESPEI-Type Workflow

```python
import numpy as np
from pycalphad import Database, equilibrium_threaded
from pycalphad.core.utils import instantiate_models
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
import pycalphad.variables as v

# Setup
dbf = Database('database.tdb')
comps = ['AL', 'FE', 'VA']
phases = ['LIQUID', 'FCC_A1', 'BCC_A2']

# Pre-build (do this once per parameter set in ESPEI optimization)
models = instantiate_models(dbf, comps, phases)
state_vars = sorted([v.T, v.P, v.N], key=str)
phase_records = PhaseRecordFactory(dbf, comps, state_vars, models)

# Generate many conditions (simulating experimental data points)
temperatures = np.linspace(800, 2000, 50)
compositions = np.linspace(0.1, 0.9, 20)

conditions_list = []
for T in temperatures:
    for x_al in compositions:
        conditions_list.append({
            v.T: float(T),
            v.P: 101325,
            v.X('AL'): float(x_al)
        })

# Calculate 1000 equilibria in parallel
results = equilibrium_threaded(
    dbf, comps, phases, conditions_list,
    max_workers=8,  # Use 8 CPU cores
    model=models,
    phase_records=phase_records
)

# Process results (calculate error, etc.)
for result in results:
    if result is not None:
        # Extract properties and compare to experimental data
        gm = result.GM.values.flat[0]
        phases_present = result.Phase.values
        # ... calculate error metrics
```

### Batched Conditions

```python
from pycalphad import equilibrium_batch_threaded
import pycalphad.variables as v

# Instead of a single equilibrium call with batched conditions:
# eq = equilibrium(dbf, comps, phases, {v.T: [1000, 1200, 1400], ...})

# Use the threaded version:
results = equilibrium_batch_threaded(
    dbf, comps, phases,
    conditions={v.T: [1000, 1200, 1400], v.P: 101325, v.X('AL'): 0.5},
    max_workers=3,
    model=models,
    phase_records=phase_records
)
```

## Performance Expectations

### With GIL Disabled (PYTHON_GIL=0)

Expected speedup on a typical ESPEI workflow with 100-1000 equilibrium calculations:

| Workers | Expected Speedup | Efficiency |
|---------|------------------|------------|
| 1       | 1.0x (baseline)  | 100%       |
| 2       | 1.8x - 1.9x      | 90-95%     |
| 4       | 3.5x - 3.8x      | 85-95%     |
| 8       | 6.0x - 7.5x      | 75-90%     |
| 16      | 8.0x - 12x       | 50-75%     |

Efficiency decreases with more threads due to:
- Memory bandwidth saturation
- Cache contention
- Overhead from thread management

### With GIL Enabled (Default Python)

Minimal to no speedup expected. May even be slower due to thread scheduling overhead.

### Optimal Configuration

- **CPU-bound calculations**: Use `max_workers` equal to the number of physical CPU cores
- **Memory-bound calculations**: Use fewer workers (4-8) to avoid memory bandwidth saturation
- **Small calculations**: Use `max_workers=1` or call `equilibrium()` directly to avoid threading overhead

## Testing

### Run the Test Suite

```bash
# With GIL disabled (recommended)
PYTHON_GIL=0 python test_espei_parallel.py

# With GIL enabled (for comparison)
python test_espei_parallel.py
```

### Test Coverage

The test suite includes:

1. **Correctness Test**: Verifies that parallel results match serial execution
2. **ESPEI Workflow Test**: Demonstrates realistic parameter optimization workflow
3. **Thread Safety Stress Test**: Tests with 100+ conditions and high thread count
4. **Performance Scaling Test**: Measures speedup with different problem sizes

### Expected Output

```
======================================================================
Python Free Threading Test Suite for pycalphad
ESPEI-Type Parallel Equilibrium Calculations
======================================================================

Python version: 3.13.x
NumPy version: 2.x.x
✓ GIL is disabled - free threading active!

======================================================================
TEST 1: Correctness - Serial vs Parallel Execution
======================================================================

Building models and phase records...

Running serial calculations...
Serial execution time: 1.234 seconds

Running parallel calculations (4 workers)...
Parallel execution time: 0.345 seconds

Comparing results...
  Condition 0: ✓ MATCH (GM=-96088.07 J/mol)
  Condition 1: ✓ MATCH (GM=-89234.56 J/mol)
  ...

✓ All results match! Speedup: 3.58x

...
```

## Troubleshooting

### GIL is Enabled Warning

If you see "WARNING: GIL is enabled", you need to disable it:

```bash
export PYTHON_GIL=0
python your_script.py
```

Or use the free-threaded Python build:

```bash
python3.13t your_script.py
```

### Import Error

If you get `ImportError: cannot import name 'equilibrium_threaded'`, check:

1. Python version is 3.11 or higher
2. pycalphad is installed correctly
3. `concurrent.futures` is available

### No Speedup Observed

If parallel execution is not faster:

1. **Check if GIL is disabled**: Use `sys._is_gil_enabled()` to verify
2. **Pre-build models and phase records**: This is critical for performance
3. **Check thread count**: Too many threads can hurt performance
4. **Problem size**: Small problems have too much overhead to benefit from parallelization

### Results Don't Match Serial Execution

This could indicate a thread safety issue. Please report this as a bug with:

1. Minimal reproducible example
2. Python version and package versions
3. Output showing the mismatch

## Integration with ESPEI

To use parallel equilibrium calculations in ESPEI:

1. Install pycalphad with this feature
2. Modify ESPEI's error calculation functions to use `equilibrium_threaded()`
3. Pre-build models and phase records at the start of each optimization iteration
4. Pass the condition list to `equilibrium_threaded()` instead of calling `equilibrium()` in a loop

Example integration point in ESPEI:

```python
# In espei/error_functions/zpf_error.py (pseudocode)

def calculate_zpf_error(dbf, comps, phases, datasets, params):
    # Build models with current parameters
    models = instantiate_models(dbf, comps, phases, params)
    phase_records = PhaseRecordFactory(dbf, comps, state_vars, models)

    # Collect all conditions
    conditions_list = [
        extract_conditions(dataset)
        for dataset in datasets
    ]

    # Parallel calculation
    from pycalphad import equilibrium_threaded
    results = equilibrium_threaded(
        dbf, comps, phases, conditions_list,
        max_workers=8,
        model=models,
        phase_records=phase_records
    )

    # Calculate error from results
    return sum(calculate_error(res, dataset)
               for res, dataset in zip(results, datasets))
```

## Known Limitations

1. **Python 3.13 Required**: Free threading is experimental in Python 3.13
2. **NumPy Thread Safety**: Requires NumPy 2.0+ for full thread safety
3. **Memory Usage**: Each thread requires some memory overhead
4. **Cython Code**: Some Cython code may not be fully optimized for GIL-free execution
5. **Debugging**: Parallel code can be harder to debug than serial code

## Future Work

### Phase 2: Low-Level Parallelization

The current implementation parallelizes at the high level (multiple calls to `equilibrium()`). Phase 2 would parallelize within a single `equilibrium()` call by modifying the Cython solver loop:

- Modify `_solve_eq_at_conditions()` in `pycalphad/core/eqsolver.pyx`
- Parallelize the condition point iteration loop
- Transparently parallelize existing `equilibrium()` API

This would require:
- Significant Cython code refactoring
- Thread safety analysis of all data structures
- More complex testing and debugging

### Additional Optimizations

- Tune chunk sizes for optimal load balancing
- Implement work stealing for uneven calculation times
- Add memory pool to reduce allocation overhead
- Profile and optimize hot paths in Cython code

## Contributing

To contribute to this feature:

1. Test on different platforms and Python versions
2. Report performance results and benchmarks
3. Report any thread safety issues or bugs
4. Suggest API improvements
5. Help with ESPEI integration

## References

- [PEP 703: Making the Global Interpreter Lock Optional](https://peps.python.org/pep-0703/)
- [Python 3.13 Free Threading Documentation](https://docs.python.org/3.13/howto/free-threading-python.html)
- [NumPy Free Threading Support](https://numpy.org/devdocs/dev/depending_on_numpy.html#free-threading-support)
- [concurrent.futures Documentation](https://docs.python.org/3/library/concurrent.futures.html)

## Contact

For questions or issues related to this feature:

- Open an issue on the pycalphad GitHub repository
- Join the pycalphad community discussions
- Contact the pycalphad development team

## License

This code is part of pycalphad and is licensed under the MIT License.
