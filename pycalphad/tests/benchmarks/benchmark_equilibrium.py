"""
Equilibrium calculation benchmarks.

These benchmarks measure the performance of the full equilibrium solver,
which exercises all core Cython modules: hyperplane, phase_rec, composition_set,
eqsolver, and minimizer.
"""
import time
import numpy as np
from pycalphad import Database, equilibrium, variables as v


def benchmark_binary_equilibrium(dbf, num_points=50):
    """
    Benchmark binary equilibrium calculation.

    Parameters
    ----------
    dbf : Database
        Thermodynamic database
    num_points : int
        Number of temperature points to calculate

    Returns
    -------
    dict
        Benchmark results with timing information
    """
    comps = ['AL', 'FE', 'VA']
    phases = list(dbf.phases.keys())

    # Setup conditions
    temps = np.linspace(300, 2500, num_points)
    x_al = 0.5

    # Warmup
    equilibrium(dbf, comps, phases, {v.T: 1000, v.X('AL'): x_al, v.P: 101325}, verbose=False)

    # Benchmark
    start_time = time.perf_counter()
    for temp in temps:
        eq_result = equilibrium(dbf, comps, phases,
                               {v.T: temp, v.X('AL'): x_al, v.P: 101325},
                               verbose=False)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    time_per_point = total_time / num_points

    return {
        'total_time': total_time,
        'time_per_point': time_per_point,
        'num_points': num_points,
        'points_per_second': num_points / total_time,
    }


def benchmark_binary_composition_scan(dbf, num_points=25):
    """
    Benchmark binary composition scan at constant temperature.

    This tests the solver's ability to track phase boundaries.
    """
    comps = ['AL', 'FE', 'VA']
    phases = list(dbf.phases.keys())

    # Setup conditions
    x_al_values = np.linspace(0.01, 0.99, num_points)
    temp = 1200

    # Warmup
    equilibrium(dbf, comps, phases, {v.T: temp, v.X('AL'): 0.5, v.P: 101325}, verbose=False)

    # Benchmark
    start_time = time.perf_counter()
    for x_al in x_al_values:
        eq_result = equilibrium(dbf, comps, phases,
                               {v.T: temp, v.X('AL'): x_al, v.P: 101325},
                               verbose=False)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    time_per_point = total_time / num_points

    return {
        'total_time': total_time,
        'time_per_point': time_per_point,
        'num_points': num_points,
        'points_per_second': num_points / total_time,
    }


def benchmark_ternary_equilibrium(dbf, num_points=20):
    """
    Benchmark ternary equilibrium calculation.

    Ternary systems are more computationally intensive due to
    increased phase space complexity.
    """
    comps = ['AL', 'NI', 'FE', 'VA']
    phases = list(dbf.phases.keys())

    # Setup conditions - simple temperature scan
    temps = np.linspace(500, 2000, num_points)
    x_al = 0.33
    x_ni = 0.33

    # Warmup
    equilibrium(dbf, comps, phases,
               {v.T: 1000, v.X('AL'): x_al, v.X('NI'): x_ni, v.P: 101325},
               verbose=False)

    # Benchmark
    start_time = time.perf_counter()
    for temp in temps:
        eq_result = equilibrium(dbf, comps, phases,
                               {v.T: temp, v.X('AL'): x_al, v.X('NI'): x_ni, v.P: 101325},
                               verbose=False)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    time_per_point = total_time / num_points

    return {
        'total_time': total_time,
        'time_per_point': time_per_point,
        'num_points': num_points,
        'points_per_second': num_points / total_time,
    }


def run_all_benchmarks():
    """
    Run all equilibrium benchmarks and print results.

    This is the main entry point for benchmark execution.
    """
    import os
    from pathlib import Path

    # Find test database
    test_db_dir = Path(__file__).parent.parent / 'databases'
    alfe_db = test_db_dir / 'alfe.tdb'

    if not alfe_db.exists():
        print(f"Warning: Could not find test database at {alfe_db}")
        print("Benchmarks require test databases to run.")
        return

    print("=" * 70)
    print("PyCalphad Equilibrium Benchmarks")
    print("=" * 70)
    print()

    # Load database
    dbf = Database(str(alfe_db))

    # Binary temperature scan
    print("1. Binary Temperature Scan (AL-FE)")
    print("-" * 70)
    results = benchmark_binary_equilibrium(dbf, num_points=50)
    print(f"  Total time:         {results['total_time']:.4f} s")
    print(f"  Time per point:     {results['time_per_point']*1000:.2f} ms")
    print(f"  Points per second:  {results['points_per_second']:.1f}")
    print()

    # Binary composition scan
    print("2. Binary Composition Scan (AL-FE at 1200K)")
    print("-" * 70)
    results = benchmark_binary_composition_scan(dbf, num_points=25)
    print(f"  Total time:         {results['total_time']:.4f} s")
    print(f"  Time per point:     {results['time_per_point']*1000:.2f} ms")
    print(f"  Points per second:  {results['points_per_second']:.1f}")
    print()

    print("=" * 70)
    print("Benchmarks complete!")
    print("=" * 70)


if __name__ == '__main__':
    run_all_benchmarks()
