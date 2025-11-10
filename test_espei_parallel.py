#!/usr/bin/env python
"""
Test script for parallel equilibrium calculations using Python 3.13 free threading.

This script demonstrates how to use thread-based parallelization for ESPEI-type
workflows that require computing many independent equilibrium calculations.

Requirements:
    - Python 3.13+ with free threading support
    - pycalphad with parallel_equilibrium module
    - Run with: PYTHON_GIL=0 python test_espei_parallel.py

Author: pycalphad development team
Date: 2025-11-10
"""

import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional
import numpy as np
from numpy.testing import assert_allclose

# Check Python version
if sys.version_info < (3, 13):
    print(f"WARNING: This script is designed for Python 3.13+")
    print(f"Current version: {sys.version}")
    print(f"Free threading features may not be available.\n")

# Check if GIL is disabled
try:
    gil_enabled = sys._is_gil_enabled()
    if gil_enabled:
        print("WARNING: GIL is enabled. For best performance, run with:")
        print("  PYTHON_GIL=0 python test_espei_parallel.py")
        print("  or use python3.13t (free-threaded build)\n")
    else:
        print("✓ GIL is disabled - free threading active!\n")
except AttributeError:
    print("WARNING: Cannot determine GIL status (pre-3.13)\n")

from pycalphad import Database, equilibrium
from pycalphad.core.utils import instantiate_models
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
import pycalphad.variables as v

warnings.filterwarnings('ignore')


# ============================================================================
# Parallel Equilibrium Implementation (Inline for Testing)
# ============================================================================

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
):
    """
    Calculate equilibrium for multiple independent condition sets using threads.

    This function leverages Python 3.13+ free threading to parallelize equilibrium
    calculations across different condition sets without GIL contention.

    Parameters
    ----------
    dbf : Database
        Thermodynamic database
    comps : list of str
        Component names
    phases : list of str
        Phase names
    conditions_list : list of dict
        List of condition dictionaries
    max_workers : int, optional
        Maximum number of threads (defaults to CPU count)
    model : Model or dict, optional
        Pre-built phase models
    phase_records : PhaseRecordFactory, optional
        Pre-built phase records (strongly recommended)
    verbose : bool
        Print calculation details
    **kwargs
        Additional arguments passed to equilibrium()

    Returns
    -------
    results : list
        Equilibrium results for each condition set
    """

    def _calc_single(idx_conds):
        """Helper to calculate single equilibrium point."""
        idx, conds = idx_conds
        try:
            result = equilibrium(
                dbf, comps, phases, conds,
                model=model,
                phase_records=phase_records,
                verbose=verbose,
                **kwargs
            )
            return (idx, result, None)
        except Exception as e:
            return (idx, None, str(e))

    # Use ThreadPoolExecutor for parallel execution
    indexed_conditions = list(enumerate(conditions_list))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results_with_indices = list(executor.map(_calc_single, indexed_conditions))

    # Sort by original index and extract results
    results_with_indices.sort(key=lambda x: x[0])
    results = []
    errors = []

    for idx, result, error in results_with_indices:
        if error is not None:
            errors.append((idx, error))
        results.append(result)

    if errors:
        print(f"\nWarning: {len(errors)} calculations failed:")
        for idx, error in errors[:5]:  # Show first 5 errors
            print(f"  Condition {idx}: {error}")

    return results


# ============================================================================
# Test Functions
# ============================================================================

def test_serial_vs_parallel_correctness():
    """
    Test that parallel execution produces identical results to serial execution.
    """
    print("=" * 70)
    print("TEST 1: Correctness - Serial vs Parallel Execution")
    print("=" * 70)

    # Load database
    dbf_path = 'pycalphad/tests/databases/alfe.tdb'
    try:
        dbf = Database(dbf_path)
    except:
        print(f"ERROR: Could not load database from {dbf_path}")
        print("Please run this script from the pycalphad root directory.")
        return False

    comps = ['AL', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2']

    # Define test conditions
    conditions_list = [
        {v.T: 1000, v.P: 101325, v.X('AL'): 0.3},
        {v.T: 1200, v.P: 101325, v.X('AL'): 0.5},
        {v.T: 1400, v.P: 101325, v.X('AL'): 0.7},
        {v.T: 1600, v.P: 101325, v.X('AL'): 0.4},
        {v.T: 1800, v.P: 101325, v.X('AL'): 0.6},
    ]

    # Pre-build models and phase records
    print("\nBuilding models and phase records...")
    models = instantiate_models(dbf, comps, phases)
    state_vars = sorted([v.T, v.P, v.N], key=str)
    phase_records = PhaseRecordFactory(dbf, comps, state_vars, models)

    # Serial execution
    print("\nRunning serial calculations...")
    start = time.perf_counter()
    serial_results = []
    for cond in conditions_list:
        result = equilibrium(dbf, comps, phases, cond,
                           model=models, phase_records=phase_records)
        serial_results.append(result)
    serial_time = time.perf_counter() - start
    print(f"Serial execution time: {serial_time:.3f} seconds")

    # Parallel execution
    print("\nRunning parallel calculations (4 workers)...")
    start = time.perf_counter()
    parallel_results = equilibrium_threaded(
        dbf, comps, phases, conditions_list,
        max_workers=4,
        model=models,
        phase_records=phase_records
    )
    parallel_time = time.perf_counter() - start
    print(f"Parallel execution time: {parallel_time:.3f} seconds")

    # Compare results
    print("\nComparing results...")
    all_match = True
    for i, (serial, parallel) in enumerate(zip(serial_results, parallel_results)):
        if parallel is None:
            print(f"  Condition {i}: FAILED (parallel calculation error)")
            all_match = False
            continue

        try:
            assert_allclose(serial.GM.values, parallel.GM.values, rtol=1e-6)
            assert_allclose(serial.MU.values, parallel.MU.values, rtol=1e-6)
            assert_allclose(serial.NP.values, parallel.NP.values, rtol=1e-6)
            print(f"  Condition {i}: ✓ MATCH (GM={serial.GM.values.flat[0]:.2f} J/mol)")
        except AssertionError as e:
            print(f"  Condition {i}: ✗ MISMATCH")
            print(f"    Serial GM: {serial.GM.values.flat[0]}")
            print(f"    Parallel GM: {parallel.GM.values.flat[0]}")
            all_match = False

    if all_match:
        speedup = serial_time / parallel_time if parallel_time > 0 else 0
        print(f"\n✓ All results match! Speedup: {speedup:.2f}x")
        return True
    else:
        print("\n✗ Some results do not match!")
        return False


def test_espei_workflow():
    """
    Test a realistic ESPEI-type workflow with many equilibrium calculations.

    Simulates a parameter optimization scenario where we need to compute
    equilibria at many conditions and compare to experimental data.
    """
    print("\n" + "=" * 70)
    print("TEST 2: ESPEI-Type Workflow - Many Equilibrium Calculations")
    print("=" * 70)

    # Load database
    dbf_path = 'pycalphad/tests/databases/alfe.tdb'
    try:
        dbf = Database(dbf_path)
    except:
        print(f"ERROR: Could not load database from {dbf_path}")
        return False

    comps = ['AL', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'HCP_A3']

    # Generate many conditions (simulating experimental data points)
    n_temperatures = 20
    n_compositions = 10
    temperatures = np.linspace(800, 2000, n_temperatures)
    compositions = np.linspace(0.1, 0.9, n_compositions)

    conditions_list = []
    for T in temperatures:
        for x_al in compositions:
            conditions_list.append({
                v.T: float(T),
                v.P: 101325,
                v.X('AL'): float(x_al)
            })

    print(f"\nTotal equilibrium calculations: {len(conditions_list)}")
    print(f"Conditions: T=[{temperatures[0]:.0f}, {temperatures[-1]:.0f}]K, "
          f"X(AL)=[{compositions[0]:.2f}, {compositions[-1]:.2f}]")

    # Pre-build models and phase records (done once in ESPEI)
    print("\nBuilding models and phase records...")
    models = instantiate_models(dbf, comps, phases)
    state_vars = sorted([v.T, v.P, v.N], key=str)
    phase_records = PhaseRecordFactory(dbf, comps, state_vars, models)

    # Benchmark different thread counts
    worker_counts = [1, 2, 4, 8]
    results = {}

    for n_workers in worker_counts:
        print(f"\nRunning with {n_workers} worker(s)...")
        start = time.perf_counter()

        calc_results = equilibrium_threaded(
            dbf, comps, phases, conditions_list,
            max_workers=n_workers,
            model=models,
            phase_records=phase_records
        )

        elapsed = time.perf_counter() - start
        results[n_workers] = elapsed

        successful = sum(1 for r in calc_results if r is not None)
        print(f"  Time: {elapsed:.3f} seconds")
        print(f"  Successful: {successful}/{len(conditions_list)}")
        print(f"  Throughput: {successful/elapsed:.1f} calculations/second")

    # Print summary
    print("\n" + "-" * 70)
    print("Performance Summary")
    print("-" * 70)
    baseline_time = results[1]

    print(f"{'Workers':<10} {'Time (s)':<12} {'Speedup':<10} {'Efficiency':<12}")
    print("-" * 70)
    for n_workers in worker_counts:
        elapsed = results[n_workers]
        speedup = baseline_time / elapsed
        efficiency = (speedup / n_workers) * 100

        print(f"{n_workers:<10} {elapsed:<12.3f} {speedup:<10.2f}x {efficiency:<12.1f}%")

    return True


def test_thread_safety_stress():
    """
    Stress test thread safety with many concurrent calculations.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Thread Safety Stress Test")
    print("=" * 70)

    # Load database
    dbf_path = 'pycalphad/tests/databases/alfe.tdb'
    try:
        dbf = Database(dbf_path)
    except:
        print(f"ERROR: Could not load database from {dbf_path}")
        return False

    comps = ['AL', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1']

    # Generate many random conditions
    n_conditions = 100
    print(f"\nGenerating {n_conditions} random conditions...")

    np.random.seed(42)
    conditions_list = []
    for _ in range(n_conditions):
        T = np.random.uniform(1000, 2000)
        x_al = np.random.uniform(0.1, 0.9)
        conditions_list.append({
            v.T: float(T),
            v.P: 101325,
            v.X('AL'): float(x_al)
        })

    # Pre-build
    models = instantiate_models(dbf, comps, phases)
    state_vars = sorted([v.T, v.P, v.N], key=str)
    phase_records = PhaseRecordFactory(dbf, comps, state_vars, models)

    # Run with high thread count
    max_workers = 16
    print(f"\nRunning {n_conditions} calculations with {max_workers} workers...")

    start = time.perf_counter()
    results = equilibrium_threaded(
        dbf, comps, phases, conditions_list,
        max_workers=max_workers,
        model=models,
        phase_records=phase_records
    )
    elapsed = time.perf_counter() - start

    successful = sum(1 for r in results if r is not None)
    print(f"\nResults:")
    print(f"  Total calculations: {len(conditions_list)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {len(conditions_list) - successful}")
    print(f"  Time: {elapsed:.3f} seconds")
    print(f"  Throughput: {successful/elapsed:.1f} calculations/second")

    if successful >= len(conditions_list) * 0.9:  # 90% success rate
        print("\n✓ Stress test passed!")
        return True
    else:
        print("\n✗ Too many failures in stress test")
        return False


def test_performance_scaling():
    """
    Test performance scaling with different problem sizes and thread counts.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Performance Scaling Analysis")
    print("=" * 70)

    # Load database
    dbf_path = 'pycalphad/tests/databases/alfe.tdb'
    try:
        dbf = Database(dbf_path)
    except:
        print(f"ERROR: Could not load database from {dbf_path}")
        return False

    comps = ['AL', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2']

    # Pre-build
    models = instantiate_models(dbf, comps, phases)
    state_vars = sorted([v.T, v.P, v.N], key=str)
    phase_records = PhaseRecordFactory(dbf, comps, state_vars, models)

    # Test different problem sizes
    problem_sizes = [10, 50, 100]
    n_workers = 4

    print(f"\nTesting scaling with {n_workers} workers...")
    print(f"{'Problem Size':<15} {'Time (s)':<12} {'Throughput (calc/s)':<20}")
    print("-" * 70)

    for n_conditions in problem_sizes:
        # Generate conditions
        temperatures = np.linspace(1000, 2000, n_conditions)
        conditions_list = [
            {v.T: float(T), v.P: 101325, v.X('AL'): 0.5}
            for T in temperatures
        ]

        # Run calculation
        start = time.perf_counter()
        results = equilibrium_threaded(
            dbf, comps, phases, conditions_list,
            max_workers=n_workers,
            model=models,
            phase_records=phase_records
        )
        elapsed = time.perf_counter() - start

        successful = sum(1 for r in results if r is not None)
        throughput = successful / elapsed

        print(f"{n_conditions:<15} {elapsed:<12.3f} {throughput:<20.1f}")

    print("\n✓ Scaling test complete!")
    return True


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all tests."""
    print("=" * 70)
    print("Python Free Threading Test Suite for pycalphad")
    print("ESPEI-Type Parallel Equilibrium Calculations")
    print("=" * 70)
    print(f"\nPython version: {sys.version}")
    print(f"NumPy version: {np.__version__}")

    try:
        import concurrent.futures
        print(f"concurrent.futures: Available")
    except ImportError:
        print(f"concurrent.futures: NOT AVAILABLE")

    print("\n")

    # Run tests
    tests = [
        ("Correctness Test", test_serial_vs_parallel_correctness),
        ("ESPEI Workflow Test", test_espei_workflow),
        ("Thread Safety Stress Test", test_thread_safety_stress),
        ("Performance Scaling Test", test_performance_scaling),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"\n✗ {test_name} FAILED with exception:")
            print(f"  {type(e).__name__}: {e}")
            results[test_name] = False
            import traceback
            traceback.print_exc()

    # Print final summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:<40} {status}")

    total_tests = len(results)
    passed_tests = sum(results.values())
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
