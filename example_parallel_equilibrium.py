#!/usr/bin/env python
"""
Simple example demonstrating parallel equilibrium calculations using free threading.

This example shows how to use the new equilibrium_threaded() function to
parallelize equilibrium calculations across multiple temperature and composition
points.

Run with:
    PYTHON_GIL=0 python example_parallel_equilibrium.py

Requirements:
    - Python 3.13+ with free threading support
    - pycalphad with parallel_equilibrium module
"""

import sys
import time
import numpy as np

print("=" * 70)
print("Parallel Equilibrium Calculation Example")
print("=" * 70)
print(f"\nPython version: {sys.version}")

# Check if GIL is disabled
try:
    if sys._is_gil_enabled():
        print("\n⚠️  WARNING: GIL is enabled")
        print("For best performance, run with: PYTHON_GIL=0 python example_parallel_equilibrium.py")
    else:
        print("\n✓ GIL is disabled - free threading active!")
except AttributeError:
    print("\n⚠️  Cannot determine GIL status (Python < 3.13)")

print("\n")

# Import pycalphad
try:
    from pycalphad import Database, equilibrium, equilibrium_threaded
    from pycalphad.core.utils import instantiate_models
    from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
    import pycalphad.variables as v
except ImportError as e:
    print(f"Error importing pycalphad: {e}")
    print("Please install pycalphad in development mode:")
    print("  pip install -e .")
    sys.exit(1)

# ============================================================================
# Example 1: Simple Parallel Calculation
# ============================================================================

print("Example 1: Simple parallel calculation at different temperatures")
print("-" * 70)

# Load database
try:
    dbf = Database('pycalphad/tests/databases/alfe.tdb')
except:
    print("Error: Could not load database. Please run from pycalphad root directory.")
    sys.exit(1)

comps = ['AL', 'FE', 'VA']
phases = ['LIQUID', 'FCC_A1', 'BCC_A2']

print(f"\nSystem: {'-'.join(comps[:-1])}")
print(f"Phases: {', '.join(phases)}")

# Define conditions at different temperatures
temperatures = [1000, 1200, 1400, 1600, 1800]
conditions_list = [
    {v.T: T, v.P: 101325, v.X('AL'): 0.5}
    for T in temperatures
]

print(f"Calculating equilibrium at {len(conditions_list)} temperatures...")
print(f"Temperatures: {temperatures}")
print(f"Composition: X(AL) = 0.5")

# Pre-build models and phase records (important for performance!)
print("\nPre-building models and phase records...")
models = instantiate_models(dbf, comps, phases)
state_vars = sorted([v.T, v.P, v.N], key=str)
phase_records = PhaseRecordFactory(dbf, comps, state_vars, models)

# Serial calculation
print("\nSerial calculation:")
start = time.perf_counter()
serial_results = []
for cond in conditions_list:
    result = equilibrium(dbf, comps, phases, cond,
                        model=models, phase_records=phase_records)
    serial_results.append(result)
serial_time = time.perf_counter() - start
print(f"  Time: {serial_time:.3f} seconds")

# Parallel calculation
print("\nParallel calculation (4 workers):")
start = time.perf_counter()
parallel_results = equilibrium_threaded(
    dbf, comps, phases, conditions_list,
    max_workers=4,
    model=models,
    phase_records=phase_records
)
parallel_time = time.perf_counter() - start
print(f"  Time: {parallel_time:.3f} seconds")

# Display results
speedup = serial_time / parallel_time if parallel_time > 0 else 0
print(f"\n✓ Speedup: {speedup:.2f}x")

print("\nResults:")
print(f"{'T (K)':<10} {'GM (J/mol)':<15} {'Phases Present':<30}")
print("-" * 70)
for i, (T, result) in enumerate(zip(temperatures, parallel_results)):
    if result is not None:
        gm = result.GM.values.flat[0]
        phases_present = [p for p in result.Phase.values.flat if p != '']
        phases_str = ', '.join(phases_present)
        print(f"{T:<10} {gm:<15.2f} {phases_str:<30}")

# ============================================================================
# Example 2: 2D Grid (Temperature × Composition)
# ============================================================================

print("\n" + "=" * 70)
print("Example 2: 2D grid calculation (Temperature × Composition)")
print("-" * 70)

# Generate a 2D grid of conditions
n_temps = 10
n_comps = 10
temperatures_2d = np.linspace(1000, 2000, n_temps)
compositions_2d = np.linspace(0.2, 0.8, n_comps)

conditions_2d = []
for T in temperatures_2d:
    for x_al in compositions_2d:
        conditions_2d.append({
            v.T: float(T),
            v.P: 101325,
            v.X('AL'): float(x_al)
        })

print(f"\nTotal calculations: {len(conditions_2d)} ({n_temps} temps × {n_comps} compositions)")
print(f"Temperature range: {temperatures_2d[0]:.0f} - {temperatures_2d[-1]:.0f} K")
print(f"Composition range: {compositions_2d[0]:.2f} - {compositions_2d[-1]:.2f} X(AL)")

# Compare different thread counts
worker_counts = [1, 2, 4, 8]
print(f"\nBenchmarking with different thread counts...")
print(f"{'Workers':<10} {'Time (s)':<12} {'Speedup':<10} {'Efficiency':<12} {'Throughput':<15}")
print("-" * 70)

baseline_time = None
for n_workers in worker_counts:
    start = time.perf_counter()
    results = equilibrium_threaded(
        dbf, comps, phases, conditions_2d,
        max_workers=n_workers,
        model=models,
        phase_records=phase_records
    )
    elapsed = time.perf_counter() - start

    if baseline_time is None:
        baseline_time = elapsed

    speedup = baseline_time / elapsed if elapsed > 0 else 0
    efficiency = (speedup / n_workers) * 100
    successful = sum(1 for r in results if r is not None)
    throughput = successful / elapsed if elapsed > 0 else 0

    print(f"{n_workers:<10} {elapsed:<12.3f} {speedup:<10.2f}x "
          f"{efficiency:<12.1f}% {throughput:<15.1f} calc/s")

# ============================================================================
# Example 3: Extracting Data for Plotting
# ============================================================================

print("\n" + "=" * 70)
print("Example 3: Extract data for phase diagram plotting")
print("-" * 70)

# Calculate along a composition line at fixed temperature
T_fixed = 1500
compositions_line = np.linspace(0.1, 0.9, 20)

conditions_line = [
    {v.T: T_fixed, v.P: 101325, v.X('AL'): float(x_al)}
    for x_al in compositions_line
]

print(f"\nCalculating at T = {T_fixed} K")
print(f"Composition range: {compositions_line[0]:.2f} - {compositions_line[-1]:.2f} X(AL)")
print(f"Number of points: {len(conditions_line)}")

# Calculate
results_line = equilibrium_threaded(
    dbf, comps, phases, conditions_line,
    max_workers=4,
    model=models,
    phase_records=phase_records
)

# Extract data
print("\nResults:")
print(f"{'X(AL)':<10} {'GM (J/mol)':<15} {'Phases':<30}")
print("-" * 70)

for x_al, result in zip(compositions_line, results_line):
    if result is not None:
        gm = result.GM.values.flat[0]
        phases_present = [p for p in result.Phase.values.flat if p != '']
        phases_str = ', '.join(phases_present)
        print(f"{x_al:<10.3f} {gm:<15.2f} {phases_str:<30}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("""
This example demonstrated:

1. Basic parallel equilibrium calculations with equilibrium_threaded()
2. Performance comparison between serial and parallel execution
3. 2D grid calculations (temperature × composition)
4. Scaling with different thread counts
5. Extracting data for phase diagram plotting

Key takeaways:
- Pre-build models and phase records for best performance
- Use max_workers equal to CPU core count (typically 4-8)
- Free threading (PYTHON_GIL=0) is required for significant speedup
- Ideal for ESPEI-type workflows with many independent equilibria

For more information, see FREE_THREADING_README.md
""")
