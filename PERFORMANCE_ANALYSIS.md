# PyCalphad Performance Analysis

## Executive Summary

This document presents a comprehensive performance analysis of PyCalphad, focusing on high-throughput CALPHAD workflows common in applications like ESPEI and kawin. Through profiling and benchmarking, we identified key bottlenecks and optimization opportunities.

**Key Findings:**
- **4.69x speedup** achieved using Workspace API vs naive workflow (3.40ms vs 15.61ms per equilibrium)
- Model instantiation is a major bottleneck when not reused (~4.7ms per system)
- PhaseRecord compilation is expensive but well-cached (~35-40ms per phase, one-time cost)
- xarray Dataset creation adds significant overhead (~5ms per equilibrium in naive workflow)
- Solver performance is reasonable (~4ms per equilibrium)

## Profiling Results

### Performance Benchmarks (Al-Fe System, 3 Phases)

| Workflow Type | Time/Equilibrium | Setup Time | Speedup |
|--------------|------------------|------------|---------|
| Naive (recreate everything) | 15.61ms | N/A | 1.0x |
| Workspace (recommended) | 3.40ms | 12ms | 4.69x |
| Grid (100 points) | 3.82ms | included | 4.09x |

### Object Creation Costs

| Operation | Time | Notes |
|-----------|------|-------|
| Model instantiation (3 phases) | 4.73ms | Per system |
| PhaseRecordFactory creation | 0.03ms | Lightweight wrapper |
| PhaseRecord compilation | 34-41ms | Per phase, heavily cached |

### Top Bottlenecks (cProfile, 50 equilibria)

1. **Model instantiation** (0.442s, 30.6% of total time)
   - Location: `pycalphad/core/utils.py:354(instantiate_models)`
   - Called: 50 times (once per equilibrium in naive workflow)
   - **Optimization: Reuse models across calculations**

2. **xarray Dataset operations** (0.257s, 17.8% of total time)
   - Location: `xarray/core/dataset.py:371(__init__)`
   - Includes coordinate creation, index casting
   - **Optimization: Minimize Dataset recreations, use Workspace**

3. **PhaseRecord compilation** (0.177s, 12.3% of total time)
   - Location: `codegen/phase_record_factory.py:48(get_phase_property)`
   - Already uses `@lru_cache()`, so first-call cost only
   - **Already optimized**

4. **Model building** (0.247s, 17.1% of total time)
   - Location: `model.py:524(build_phase)`
   - Part of model instantiation
   - **Optimization: Reuse models**

5. **Equilibrium solver** (0.213s, 14.8% of total time)
   - Location: `solver.py:147(solve)`
   - This is the actual computational work
   - **Cannot be easily optimized without algorithmic changes**

## Current Caching Mechanisms

PyCalphad already implements several caching strategies:

### 1. PhaseRecordFactory Caching
All PhaseRecordFactory methods use `@lru_cache()`:
- `get_phase_constraints()`
- `get_phase_formula_moles_element()`
- `get_phase_property()` - **Most important for performance**
- `get()` - PhaseRecord creation

### 2. Function Compilation Caching
- `build_functions()` in `sympydiff_utils.py` uses `@cacheit` (unbounded LRU cache)
- LLVM-compiled property functions are cached
- Common Subexpression Elimination (CSE) optimization applied

### 3. Point Sampling Caching
- `_sample_phase_constitution()` uses `@cacheit`
- Caches sampled composition points for phases

### 4. Workspace Reactive Caching
- Automatically tracks dependencies between fields
- Only recomputes when dependencies change
- Reuses models, phase_records, and equilibrium results

## Optimization Strategies for High-Throughput Workflows

### Pattern 1: Same Components/Phases, Varying Conditions (RECOMMENDED)

This is the most common pattern in ESPEI and kawin. **Use the Workspace API:**

```python
from pycalphad import Database, Workspace
import pycalphad.variables as v

# Initialize once
dbf = Database('my_database.tdb')
comps = ['AL', 'FE', 'VA']
phases = ['LIQUID', 'FCC_A1', 'BCC_A2']

# Create workspace with initial conditions
wks = Workspace(
    database=dbf,
    components=comps,
    phases=phases,
    conditions={v.T: 1000, v.P: 101325, v.X('FE'): 0.5}
)

# Initial equilibrium (triggers compilation, ~50ms setup)
eq = wks.eq

# Update conditions for subsequent calculations (3-4ms each)
for temperature in temperature_range:
    wks.conditions[v.T] = temperature  # Fast update
    eq = wks.eq  # Triggers recalculation only
```

**Performance:**
- First equilibrium: ~50ms (includes LLVM compilation)
- Subsequent equilibria: ~3.4ms each
- **4.69x faster than naive approach**

### Pattern 2: Manual Object Reuse (Advanced)

For maximum control, reuse models and phase_records manually:

```python
from pycalphad import Database, equilibrium
from pycalphad.core.utils import instantiate_models
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
import pycalphad.variables as v

dbf = Database('my_database.tdb')
comps = ['AL', 'FE', 'VA']
phases = ['LIQUID', 'FCC_A1', 'BCC_A2']

# Create once, reuse everywhere
models = instantiate_models(dbf, comps, phases)  # ~5ms

# Note: PhaseRecords are created lazily and cached internally
# You typically don't need to create PhaseRecordFactory manually
# unless you need fine-grained control

# Run calculations
for temp in temperature_range:
    # Pass models to avoid recreation (but still creates new Workspace internally)
    # This is less optimal than Pattern 1
    eq = equilibrium(
        dbf, comps, phases,
        {v.T: temp, v.P: 101325, v.X('FE'): 0.5},
        model=models  # Reuse models
    )
```

**Note:** This is less efficient than Pattern 1 because it still creates a new Workspace for each call.

### Pattern 3: Batch/Grid Calculations

For grid calculations (phase diagrams, property maps):

```python
import numpy as np
from pycalphad import equilibrium
import pycalphad.variables as v

dbf = Database('my_database.tdb')
comps = ['AL', 'FE', 'VA']
phases = ['LIQUID', 'FCC_A1', 'BCC_A2']

# Define grid
temperatures = np.linspace(500, 2000, 50)
compositions = np.linspace(0.1, 0.9, 50)

# Single call computes entire grid
eq = equilibrium(
    dbf, comps, phases,
    {v.T: temperatures, v.P: 101325, v.X('FE'): compositions}
)

# Result is 50x50 grid of equilibria
# Time: ~3.8ms per equilibrium point
```

**Performance:**
- Efficient for grids due to vectorization
- ~3.8ms per equilibrium point
- Workspace is created once for entire grid

## Identified Optimization Opportunities

### 1. Model Instantiation (HIGH PRIORITY)

**Problem:** Models are recreated for every equilibrium call in naive workflows.

**Current Cost:** ~4.7ms per system (3 phases)

**Solution:**
- ✅ **Already implemented in Workspace API** - users should use it
- Document best practices more prominently
- Consider caching at the `instantiate_models()` level for backward compatibility

**Estimated Speedup:** 4-5x for workflows that reuse components/phases

### 2. xarray Dataset Creation Overhead (MEDIUM PRIORITY)

**Problem:** Dataset creation and coordinate handling adds ~5ms overhead per equilibrium.

**Current Cost:** ~17.8% of total time in naive workflow

**Solutions:**
- Return lightweight `LightDataset` objects when `to_xarray=False`
- Batch multiple equilibria before converting to xarray
- Cache coordinate objects for repeated grid structures

**Estimated Speedup:** 1.2-1.5x for batch operations

**Implementation Complexity:** Medium - requires API design decisions

### 3. Import Overhead (LOW PRIORITY)

**Problem:** cftimeindex imports happen during Dataset creation (0.135s for 150 imports)

**Current Cost:** ~0.9ms per import, happens frequently

**Solutions:**
- Move imports to module level
- Lazy import only when needed
- Use faster datetime handling

**Estimated Speedup:** 1.1-1.2x

**Implementation Complexity:** Low

### 4. Parallelization Opportunities (FUTURE WORK)

**Problem:** Sequential equilibrium calculations don't utilize multiple cores.

**Current State:** No parallelization in core equilibrium solver

**Solutions:**
- Parallel equilibrium calculations for independent points
- Thread pool for grid calculations
- GPU acceleration for property evaluation (LLVM already used)

**Estimated Speedup:** Near-linear with core count for embarrassingly parallel workloads

**Implementation Complexity:** High - requires thread safety audit

## Recommendations by Use Case

### For ESPEI Users (Parameter Optimization)

ESPEI runs thousands of equilibrium calculations with:
- Same database, components, phases
- Varying conditions (T, P, compositions)
- Need for reproducibility

**Recommendations:**
1. **Use Workspace API** for iterative optimization
2. **Disable xarray conversion** when not needed (`to_xarray=False`)
3. **Cache Database objects** - avoid re-parsing TDB files
4. **Consider batching** - group similar conditions together

**Expected Performance:**
- Current: ~15ms per equilibrium (if not using Workspace)
- Optimized: ~3-4ms per equilibrium
- **3-5x speedup possible**

### For kawin Users (Precipitation Simulations)

kawin runs time-stepping simulations with:
- Same system throughout simulation
- Changing conditions (T, compositions) over time
- Many repeated calls

**Recommendations:**
1. **Use Workspace API** - perfect fit for this use case
2. **Reuse Workspace instance** across time steps
3. **Only update changed conditions** (temperature, composition)
4. **Consider adaptive step sizing** to minimize equilibrium calls

**Expected Performance:**
- With Workspace: ~3-4ms per equilibrium
- Hundreds to thousands of time steps feasible

### For Mapping/Phase Diagram Calculations

Mapping constructs phase diagrams by:
- Adaptive stepping along boundaries
- Many equilibrium calculations
- Intelligent starting point reuse

**Recommendations:**
1. **Grid calculations** are already optimized (~3.8ms/point)
2. **Workspace already used internally** in mapping strategies
3. **Focus on adaptive stepping algorithms** - minimize equilibrium calls
4. **Starting point quality** is more important than per-calculation speed

**Expected Performance:**
- Current implementation is well-optimized
- Focus on algorithmic improvements (better starting points, fewer iterations)

## Detailed Code Analysis

### Critical Path Analysis

For a typical equilibrium calculation:

```
equilibrium()
├─ Workspace.__init__() [EXPENSIVE if models not provided]
│  ├─ instantiate_models() [4.7ms] ← MAJOR BOTTLENECK
│  │  └─ Model.__init__() × N phases [~1.5ms each]
│  │     └─ build_phase() [symbolic manipulation]
│  └─ PhaseRecordFactory() [0.03ms]
│     └─ (lazy creation, cached)
│
├─ Workspace.recompute() [Main computation]
│  ├─ calculate() [Property sampling]
│  │  ├─ _sample_phase_constitution() [cached]
│  │  ├─ PhaseRecord creation [35-40ms first time, cached]
│  │  │  ├─ build_functions() [LLVM compilation, cached]
│  │  │  └─ get_phase_property() [@lru_cache]
│  │  └─ _compute_phase_values() [Fast, vectorized]
│  │
│  ├─ starting_point() [Find initial guess]
│  │  └─ lower_convex_hull() [~1ms]
│  │
│  └─ Solver.solve() [4.3ms per equilibrium]
│     └─ Iterative optimization (typically 1-3 iterations)
│
└─ Dataset conversion [5ms] ← OVERHEAD
   └─ xarray.Dataset.__init__()
      └─ Coordinate/index creation
```

### Caching Effectiveness

Current cache hit rates (estimated from profiling):

- `PhaseRecord.get()`: Near 100% after first phase creation
- `_sample_phase_constitution()`: High hit rate for repeated conditions
- `build_functions()`: 100% hit rate after first compilation
- Model instances: 0% hit rate without Workspace (recreated every time)

**Impact of Workspace:**
- Model reuse: 100% hit rate after first use
- PhaseRecord reuse: 100% hit rate
- Equilibrium result caching: Conditional on whether conditions changed

## Benchmarking Script

A complete profiling script is available in `profile_performance.py`. Run it to analyze performance on your system:

```bash
python profile_performance.py
```

The script generates:
- Performance comparisons between naive and optimized workflows
- Object creation cost analysis
- Detailed cProfile output (`pycalphad_profile.prof`)

Analyze detailed profile with:
```bash
python -m pstats pycalphad_profile.prof
```

## Conclusion

PyCalphad's performance is limited primarily by **improper usage patterns** rather than algorithmic inefficiencies. The Workspace API provides excellent performance for high-throughput workflows (~3-4ms per equilibrium), achieving a 4.69x speedup over naive approaches.

**Primary Recommendations:**

1. **Use Workspace API** for all high-throughput workflows ✓
2. **Reuse Workspace instances** when components/phases don't change ✓
3. **Documentation improvements** - make best practices more visible
4. **Consider API enhancements** - make it harder to use inefficiently
5. **Future work** - parallelization for multi-core systems

The existing caching infrastructure (PhaseRecord compilation, function caching) is well-designed and effective. The main opportunity for user-facing performance improvements is better education and API design to guide users toward efficient patterns.

## Additional Resources

- [Workspace API Documentation](https://pycalphad.org/docs/)
- [ESPEI Performance Guide](https://espei.org/)
- [kawin Documentation](https://kawin.readthedocs.io/)

## Appendix: Profiling Data

### Full cProfile Output (Top 30 Functions)

```
         1455065 function calls (1344181 primitive calls) in 1.443 seconds

   Ordered by: cumulative time
   List reduced from 712 to 30 due to restriction <30>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       50    0.001    0.000    1.440    0.029 equilibrium.py:15(equilibrium)
      100    0.001    0.000    0.666    0.007 workspace.py:279(__get__)
       50    0.015    0.000    0.666    0.013 workspace.py:341(recompute)
       50    0.001    0.000    0.515    0.010 workspace.py:316(__init__)
       50    0.001    0.000    0.452    0.009 workspace.py:235(__set__)
       50    0.001    0.000    0.442    0.009 utils.py:354(instantiate_models)
      150    0.020    0.000    0.439    0.003 model.py:193(__init__)
       50    0.015    0.000    0.380    0.008 calculate.py:362(calculate)
       50    0.000    0.000    0.257    0.005 light_dataset.py:58(get_dataset)
      150    0.004    0.000    0.247    0.002 model.py:524(build_phase)
       50    0.067    0.001    0.213    0.004 solver.py:147(solve)
      357    0.048    0.000    0.177    0.000 phase_record_factory.py:48(get_phase_property)
      150    0.013    0.000    0.177    0.001 phase_record_factory.py:65(get)
```

Key observations:
- `instantiate_models()`: 0.442s / 50 calls = 8.8ms per call
- `Model.__init__()`: 0.439s / 150 calls = 2.9ms per model (3 phases × 50)
- `Solver.solve()`: 0.213s / 50 calls = 4.26ms per equilibrium
- `get_phase_property()`: 0.177s / 357 calls = 0.5ms per call (includes LLVM compilation on first call)
