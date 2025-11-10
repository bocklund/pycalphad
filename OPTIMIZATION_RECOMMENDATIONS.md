# PyCalphad Optimization Recommendations

## Summary of Findings

Performance analysis identified that PyCalphad can achieve **4.69x speedup** (15.61ms → 3.40ms per equilibrium) when using the Workspace API properly. The main bottleneck is **model recreation** in naive workflows, not algorithmic inefficiency.

## Priority-Ranked Optimization Opportunities

### 🔴 HIGH PRIORITY: Documentation & User Education

**Problem:** Users unknowingly use inefficient patterns by calling `equilibrium()` in loops without reusing objects.

**Impact:** 4-5x performance penalty

**Effort:** Low (documentation updates)

**Actions:**
1. Add prominent "Performance Guide" to documentation
2. Show anti-patterns vs. recommended patterns
3. Add performance warnings to `equilibrium()` docstring
4. Create tutorial notebooks for ESPEI/kawin-style workflows

**Example addition to `equilibrium()` docstring:**

```python
"""
...
Performance Notes
-----------------
For high-throughput workflows (many equilibrium calls with the same components/phases):

❌ SLOW (recreates models every time):
    for temp in temperatures:
        eq = equilibrium(dbf, comps, phases, {v.T: temp, ...})

✓ FAST (use Workspace API):
    wks = Workspace(dbf, comps, phases, {v.T: temperatures[0], ...})
    eq = wks.eq
    for temp in temperatures[1:]:
        wks.conditions[v.T] = temp
        eq = wks.eq

This can provide 4-5x speedup for typical workflows.
"""
```

### 🟡 MEDIUM PRIORITY: API Improvements

#### 1. Add Performance Warnings

**Problem:** No feedback when users use inefficient patterns

**Solution:** Detect and warn about performance anti-patterns

```python
# In equilibrium() function
import warnings

def equilibrium(dbf, comps, phases, conditions, ...):
    # Detect if we're in a hot loop (called many times without model reuse)
    if model is None and hasattr(equilibrium, '_call_count'):
        equilibrium._call_count += 1
        if equilibrium._call_count > 10:
            warnings.warn(
                "equilibrium() called many times without model reuse. "
                "Consider using the Workspace API for better performance. "
                "See: https://pycalphad.org/docs/performance",
                PerformanceWarning,
                stacklevel=2
            )
    elif model is None:
        equilibrium._call_count = 1
    ...
```

**Impact:** Educates users in real-time
**Effort:** Low
**Risk:** Low (only warnings)

#### 2. Convenience Function for Batch Equilibria

**Problem:** Users write loops instead of using batch mode

**Solution:** Add helper for sequential calculations with state reuse

```python
def equilibrium_sequence(dbf, comps, phases, conditions_list, **kwargs):
    """
    Compute equilibria for a sequence of conditions efficiently.

    This is optimized for the common pattern of varying one or more conditions
    while keeping the system (components, phases) the same.

    Parameters
    ----------
    dbf : Database
    comps : list
    phases : list
    conditions_list : list of dict
        List of condition dictionaries, one per equilibrium to compute

    Returns
    -------
    list of Dataset
        Equilibrium results for each condition set

    Examples
    --------
    >>> conditions = [
    ...     {v.T: 1000, v.P: 101325, v.X('FE'): 0.1},
    ...     {v.T: 1100, v.P: 101325, v.X('FE'): 0.2},
    ...     {v.T: 1200, v.P: 101325, v.X('FE'): 0.3},
    ... ]
    >>> results = equilibrium_sequence(dbf, ['AL', 'FE', 'VA'],
    ...                                ['LIQUID', 'FCC_A1'], conditions)

    Notes
    -----
    This is 4-5x faster than calling equilibrium() in a loop because it
    reuses models and compiled phase records.
    """
    if len(conditions_list) == 0:
        return []

    # Initialize workspace with first condition
    wks = Workspace(
        database=dbf,
        components=comps,
        phases=phases,
        conditions=conditions_list[0],
        **kwargs
    )

    results = [wks.eq]

    # Update conditions for remaining equilibria
    for conditions in conditions_list[1:]:
        for key, value in conditions.items():
            wks.conditions[key] = value
        results.append(wks.eq)

    return results
```

**Impact:** Makes efficient pattern easier to use
**Effort:** Low (50-100 lines)
**Risk:** Low (new API, doesn't break existing code)

#### 3. Model/PhaseRecord Caching at Module Level (Optional)

**Problem:** Even with manual model reuse, users must manage objects

**Solution:** Add optional global cache keyed by (database_hash, components, phases)

```python
from functools import lru_cache
import hashlib

def _database_hash(dbf):
    """Create a hash of database content for caching."""
    # This is a simplified version - real implementation needs careful design
    return hash(tuple(sorted(dbf._parameters.values())))

@lru_cache(maxsize=8)  # Cache up to 8 different systems
def get_cached_models(database_id, components_tuple, phases_tuple, parameters_tuple=None):
    """
    Get or create models for a system.

    This function caches models to avoid recreation when the same
    system is used repeatedly across different equilibrium() calls.

    Note: This is an internal optimization. Users should prefer
    the Workspace API for explicit control.
    """
    # Reconstruct from tuples
    from pycalphad.core.utils import instantiate_models
    models = instantiate_models(dbf, components, phases, parameters=parameters)
    return models

def equilibrium(dbf, comps, phases, conditions, model=None, ...):
    if model is None and USE_MODEL_CACHE:  # opt-in flag
        db_id = id(dbf)  # or _database_hash(dbf)
        comps_tuple = tuple(sorted(comps))
        phases_tuple = tuple(sorted(phases))
        params_tuple = tuple(sorted(parameters.items())) if parameters else None
        model = get_cached_models(db_id, comps_tuple, phases_tuple, params_tuple)
    ...
```

**Impact:** Automatic 4-5x speedup for naive code
**Effort:** Medium (100-200 lines, testing)
**Risk:** Medium (cache invalidation complexity, memory usage)

**Recommendation:** Consider for v1.0 or v2.0, but document Workspace API as primary solution first.

### 🟢 LOW PRIORITY: Implementation Optimizations

#### 1. Reduce xarray Dataset Creation Overhead

**Problem:** Dataset creation adds ~5ms per equilibrium due to coordinate handling

**Current hotspots:**
- `safe_cast_to_index()`: 0.167s / 900 calls = 0.19ms each
- `_maybe_cast_to_cftimeindex()`: 0.140s for imports
- Coordinate creation: 0.219s

**Solutions:**

**Option A:** Defer Dataset conversion
```python
class EquilibriumResult:
    """Lightweight equilibrium result that converts to Dataset on demand."""
    def __init__(self, light_dataset):
        self._light_ds = light_dataset
        self._xr_dataset = None

    def to_xarray(self):
        if self._xr_dataset is None:
            self._xr_dataset = self._light_ds.get_dataset()
        return self._xr_dataset

    def __getattr__(self, name):
        # Auto-convert on attribute access for backward compatibility
        return getattr(self.to_xarray(), name)

# In equilibrium():
if to_xarray == 'lazy':  # New option
    return EquilibriumResult(properties)
elif to_xarray:
    return properties.get_dataset()
else:
    return properties
```

**Option B:** Cache coordinate objects
```python
@lru_cache(maxsize=128)
def _get_coordinates_cached(coord_keys_tuple, coord_values_tuple):
    """Cache coordinate creation for repeated grid structures."""
    coord_dict = dict(zip(coord_keys_tuple, coord_values_tuple))
    return create_coords_with_default_indexes(coord_dict)

# Use in LightDataset.get_dataset()
```

**Impact:** 1.2-1.5x speedup for batch operations
**Effort:** Medium
**Risk:** Medium (API compatibility)

#### 2. Optimize Import Statements

**Problem:** `cftimeindex` imported 150 times during profiling (0.134s total)

**Solution:**
```python
# At module level in xarray coordinate creation
try:
    import cftime
    HAS_CFTIME = True
except ImportError:
    HAS_CFTIME = False

# In coordinate creation code
if HAS_CFTIME and needs_cftime:  # Only import if actually needed
    ...
```

**Impact:** 1.1-1.2x speedup
**Effort:** Low (but requires changes to xarray or PyCalphad's xarray usage)
**Risk:** Low

#### 3. Parallel Equilibrium Calculations (Future Work)

**Problem:** No parallelization for independent equilibrium points

**Use Case:** Grid calculations, mapping strategies

**Solution:** Add optional parallelization

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def equilibrium(dbf, comps, phases, conditions,
                parallel=False, n_jobs=None, ...):
    """
    ...
    parallel : bool or str, optional
        Enable parallel computation for grid calculations.
        - False (default): Sequential computation
        - 'thread': Use ThreadPoolExecutor (for I/O-bound)
        - 'process': Use ProcessPoolExecutor (for CPU-bound)
        Note: Only applies when conditions form a grid.
    n_jobs : int, optional
        Number of parallel workers. Default: cpu_count()
    """

    if parallel and is_grid_calculation(conditions):
        # Flatten grid into list of individual conditions
        cond_list = flatten_grid_conditions(conditions)

        executor_class = (ThreadPoolExecutor if parallel == 'thread'
                         else ProcessPoolExecutor)

        with executor_class(max_workers=n_jobs) as executor:
            # Each worker reuses workspace
            futures = [
                executor.submit(compute_single_equilibrium,
                              dbf, comps, phases, cond)
                for cond in cond_list
            ]
            results = [f.result() for f in futures]

        return combine_results_to_grid(results, conditions)
    else:
        # Standard sequential computation
        ...
```

**Impact:** Near-linear speedup with core count (e.g., 8x on 8 cores)
**Effort:** High (requires thread safety audit, testing)
**Risk:** High (potential for subtle bugs, GIL contention)

**Recommendation:** Post-1.0 feature, requires careful design

## Recommendations for Downstream Projects

### ESPEI

**Current Bottleneck:** Likely equilibrium calls during residual evaluation

**Recommendations:**
1. **Use Workspace API** in likelihood functions
2. **Batch equilibria** with same system (group by components/phases)
3. **Cache Database objects** - avoid re-parsing TDB files
4. **Profile with this script** to identify application-specific bottlenecks

**Example ESPEI optimization:**
```python
class OptimizedLikelihood:
    def __init__(self, dbf, datasets):
        self.dbf = dbf
        # Group datasets by system (comps, phases)
        self.grouped_datasets = self.group_by_system(datasets)
        # Pre-create workspaces for each system
        self.workspaces = {}
        for system_key, data in self.grouped_datasets.items():
            comps, phases = system_key
            # Initialize with first condition from dataset
            initial_cond = data[0]['conditions']
            self.workspaces[system_key] = Workspace(
                self.dbf, comps, phases, initial_cond
            )

    def compute_residuals(self, parameters):
        # Update database with new parameters
        self.dbf.update_parameters(parameters)

        residuals = []
        for system_key, data in self.grouped_datasets.items():
            wks = self.workspaces[system_key]
            for datapoint in data:
                # Update conditions (fast)
                for key, val in datapoint['conditions'].items():
                    wks.conditions[key] = val
                # Compute equilibrium (3-4ms)
                eq = wks.eq
                # Calculate residual
                residuals.append(self.residual(eq, datapoint))

        return residuals
```

### kawin

**Current Bottleneck:** Time-stepping simulations

**Recommendations:**
1. **Reuse single Workspace** across all time steps
2. **Adaptive time stepping** to minimize equilibrium calls
3. **Cache recent equilibria** for interpolation
4. **Consider lookup tables** for very frequent calls

**Example kawin optimization:**
```python
class PrecipitationSimulation:
    def __init__(self, dbf, components, phases):
        self.wks = Workspace(
            database=dbf,
            components=components,
            phases=phases,
            conditions=self.initial_conditions()
        )
        # Cache for interpolation
        self.eq_cache = []  # List of (time, conditions, equilibrium)

    def time_step(self, dt):
        # Update conditions based on simulation state
        new_T = self.temperature_profile(self.time + dt)
        new_X = self.composition_evolution(self.time + dt)

        # Check if we need new equilibrium or can interpolate
        if self.can_interpolate(new_T, new_X):
            eq = self.interpolate_equilibrium(new_T, new_X)
        else:
            # Update workspace conditions (fast)
            self.wks.conditions[v.T] = new_T
            self.wks.conditions[v.X('AL')] = new_X
            # Compute equilibrium (3-4ms)
            eq = self.wks.eq
            # Cache result
            self.eq_cache.append((self.time + dt, (new_T, new_X), eq))

        # Update simulation state
        self.update_state(eq, dt)
        self.time += dt
```

## Implementation Priority

### Phase 1 (Immediate - Documentation)
1. Add performance section to main documentation
2. Update `equilibrium()` docstring with performance notes
3. Create example notebooks for high-throughput workflows
4. Add "Performance" page to docs with profiling results

### Phase 2 (Short-term - Low-hanging fruit)
1. Add `PerformanceWarning` for detected anti-patterns
2. Implement `equilibrium_sequence()` helper function
3. Optimize import statements
4. Add profiling script to repository

### Phase 3 (Medium-term - API enhancements)
1. Lazy Dataset conversion option
2. Coordinate caching
3. Optional global model cache (with careful design)
4. Batch equilibrium API

### Phase 4 (Long-term - Advanced features)
1. Parallel grid calculations
2. GPU acceleration for property evaluation
3. Compiled equilibrium solver (Cython/Numba)
4. Incremental equilibrium updates (warm starts)

## Testing & Validation

All optimizations should maintain:
- ✓ Numerical accuracy (results match within tolerance)
- ✓ API compatibility (existing code continues to work)
- ✓ Thread safety (if applicable)
- ✓ Memory efficiency (avoid memory leaks)

Recommended test suite:
```python
def test_optimization_correctness():
    """Verify optimized and naive paths give same results."""
    # Compare results
    eq_naive = equilibrium(dbf, comps, phases, conds)

    wks = Workspace(dbf, comps, phases, conds)
    eq_optimized = wks.eq

    np.testing.assert_allclose(eq_naive.GM, eq_optimized.GM, rtol=1e-10)
    # ... test all properties

def test_performance_improvement():
    """Verify optimization actually improves performance."""
    import time

    # Naive
    start = time.perf_counter()
    for cond in conditions_list:
        eq = equilibrium(dbf, comps, phases, cond)
    naive_time = time.perf_counter() - start

    # Optimized
    wks = Workspace(dbf, comps, phases, conditions_list[0])
    start = time.perf_counter()
    for cond in conditions_list[1:]:
        for key, val in cond.items():
            wks.conditions[key] = val
        eq = wks.eq
    optimized_time = time.perf_counter() - start

    speedup = naive_time / optimized_time
    assert speedup > 3.0, f"Expected >3x speedup, got {speedup:.2f}x"
```

## Conclusion

PyCalphad's performance bottlenecks are primarily **usage patterns** rather than algorithmic issues. The Workspace API already provides excellent performance (~3-4ms per equilibrium) - users just need to be guided toward it.

**Immediate actions with highest ROI:**
1. ✅ Document Workspace API performance benefits
2. ✅ Add examples for ESPEI/kawin-like workflows
3. ✅ Include profiling script in repository
4. Add performance warnings for anti-patterns

These documentation and education efforts can provide 4-5x performance improvements for many users with minimal development effort.
