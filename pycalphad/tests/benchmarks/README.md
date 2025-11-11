# PyCalphad Performance Benchmarks

This directory contains performance benchmarks for verifying that the C++ migration maintains or improves performance over the original Cython implementation.

## Running Benchmarks

### Quick Run

Run a single benchmark suite:
```bash
uv run python pycalphad/tests/benchmarks/benchmark_equilibrium.py
```

### Comprehensive Benchmarking

Run all benchmarks and save results:
```bash
# Run current implementation
uv run python -m pycalphad.tests.benchmarks.benchmark_equilibrium > baseline_results.txt

# After C++ migration, compare
uv run python -m pycalphad.tests.benchmarks.benchmark_equilibrium > cpp_results.txt
diff baseline_results.txt cpp_results.txt
```

## Benchmark Suites

### benchmark_equilibrium.py

Full equilibrium calculations exercising all core modules:
- **Binary temperature scans** - Track phase boundaries across temperature range
- **Binary composition scans** - Test phase boundary tracking in composition space
- **Ternary calculations** - More complex phase space navigation

**Metrics**:
- Total execution time
- Time per equilibrium point
- Points per second throughput

**Typical Performance** (reference: i7-9750H, single-threaded):
- Binary temperature scan (50 points): ~2-5 seconds
- Binary composition scan (25 points): ~1-3 seconds

## Interpreting Results

### Acceptable Performance

C++ implementation should meet these criteria:
- **Time per point**: Within ±5% of Cython baseline
- **Total throughput**: No worse than 95% of Cython
- **Memory usage**: No more than 110% of Cython

### Performance Regression

If C++ is significantly slower:
1. Check for debug builds (should use `-O3` optimization)
2. Verify proper inlining of hot-path functions
3. Profile to identify bottlenecks
4. Compare assembly output for critical loops

### Performance Improvement

C++ may be faster due to:
- Better compiler optimization opportunities
- Reduced Python/C boundary crossings
- More efficient memory layouts
- Elimination of GIL constraints (for future parallelization)

## Adding New Benchmarks

To add a new benchmark:

1. Create `benchmark_<module>.py` in this directory
2. Implement benchmark functions with clear names
3. Include warmup iterations to stabilize timings
4. Document expected performance characteristics
5. Add to this README

## Continuous Performance Tracking

### CI Integration

Benchmarks can be integrated into CI to catch regressions:

```yaml
# .github/workflows/benchmarks.yaml
- name: Run benchmarks
  run: |
    uv run python -m pycalphad.tests.benchmarks.benchmark_equilibrium
```

### Performance History

Track performance over time:
```bash
# Save results with git hash
git rev-parse HEAD > benchmark_$(date +%Y%m%d).txt
uv run python -m pycalphad.tests.benchmarks.benchmark_equilibrium >> benchmark_$(date +%Y%m%d).txt
```

## Platform Considerations

### Linux
- Most stable timing results
- Use `perf` for detailed profiling: `perf record -g python benchmark.py`

### macOS
- Good timing stability
- Use Instruments for profiling

### Windows
- More variable timing due to OS scheduling
- Run multiple times and take median
- Disable Windows Defender for benchmark directories

## Profiling Tools

### Python Profilers
```bash
# Line profiler
uv run python -m line_profiler benchmark_equilibrium.py

# cProfile
uv run python -m cProfile -o profile.stats benchmark_equilibrium.py
```

### C++ Profilers
```bash
# Valgrind callgrind (Linux)
valgrind --tool=callgrind python benchmark.py

# perf (Linux)
perf record -g python benchmark.py
perf report

# Instruments (macOS)
instruments -t "Time Profiler" python benchmark.py
```

## Memory Profiling

Check for memory leaks:
```bash
# Python memory profiler
uv run python -m memory_profiler benchmark_equilibrium.py

# Valgrind memcheck (Linux)
valgrind --leak-check=full python benchmark_equilibrium.py
```

## See Also

- `../TESTING.md` - Overall testing strategy
- `../test_cpp_migration_utils.py` - Utilities for correctness verification
- `../../cpp/README.md` - C++ implementation documentation
