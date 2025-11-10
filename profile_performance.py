"""
Profiling script for PyCalphad performance analysis.

This script simulates high-throughput CALPHAD workflows where the same
components and phases are used but with varying conditions (temperatures,
compositions, etc.). This is common in ESPEI and kawin workflows.
"""

import time
import cProfile
import pstats
import io
from pstats import SortKey
import numpy as np
from pycalphad import Database, equilibrium, calculate
from pycalphad.core.utils import instantiate_models
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.core.workspace import Workspace
import pycalphad.variables as v

# Try to import line_profiler if available
try:
    from line_profiler import LineProfiler
    HAS_LINE_PROFILER = True
except ImportError:
    HAS_LINE_PROFILER = False
    print("line_profiler not available. Install with: pip install line_profiler")

def get_test_database():
    """Create a simple test database for Al-Fe system."""
    # Using Al-Fe as a common test system
    dbf = Database("""
    ELEMENT AL FCC_A1 2.6982E+01 4.5773E+03 2.8322E+01 !
    ELEMENT FE BCC_A2 5.5847E+01 4.4890E+03 2.7280E+01 !
    ELEMENT VA VACUUM 0.0 0.0 0.0 !

    FUNCTION GHSERAL 298.15 -7976.15+137.093038*T-24.3671976*T*LN(T)
        -.001884662*T**2-8.77664E-07*T**3+74092*T**(-1); 700.00 Y
        -11276.24+223.048446*T-38.5844296*T*LN(T)+.018531982*T**2
        -5.764227E-06*T**3+74092*T**(-1); 933.47 Y
        -11278.378+188.684153*T-31.748192*T*LN(T)-1.230524E+28*T**(-9);
        2900.00 N !

    FUNCTION GHSERFE 298.15 +1225.7+124.134*T-23.5143*T*LN(T)
        -.00439752*T**2-5.8927E-08*T**3+77358.5*T**(-1); 1811.00 Y
        -25383.581+299.31255*T-46*T*LN(T)+2.29603E+31*T**(-9); 6000.00 N !

    TYPE_DEFINITION % SEQ * !
    DEFINE_SYSTEM_DEFAULT ELEMENT 2 !

    PHASE FCC_A1 %  2 1 1 !
    CONSTITUENT FCC_A1 :AL,FE:VA: !

    PHASE BCC_A2 %  2 1 3 !
    CONSTITUENT BCC_A2 :AL,FE:VA: !

    PHASE LIQUID % 1 1 !
    CONSTITUENT LIQUID :AL,FE: !

    $ FCC_A1 parameters
    PARAMETER G(FCC_A1,AL:VA;0) 298.15 +GHSERAL; 6000 N !
    PARAMETER G(FCC_A1,FE:VA;0) 298.15 +GHSERFE+1.3E-3; 6000 N !
    PARAMETER G(FCC_A1,AL,FE:VA;0) 298.15 -76066.1+18.6758*T; 6000 N !
    PARAMETER G(FCC_A1,AL,FE:VA;1) 298.15 +21167.4+1.3398*T; 6000 N !

    $ BCC_A2 parameters
    PARAMETER G(BCC_A2,AL:VA;0) 298.15 +GHSERAL+10083-4.813*T; 6000 N !
    PARAMETER G(BCC_A2,FE:VA;0) 298.15 +GHSERFE; 6000 N !
    PARAMETER G(BCC_A2,AL,FE:VA;0) 298.15 -122960+31.9888*T; 6000 N !

    $ LIQUID parameters
    PARAMETER G(LIQUID,AL;0) 298.15 +GHSERAL+11005.553-11.840873*T
        +7.9401E-20*T**7; 933.47 Y
        +10481.974-11.253974*T+1.231E+28*T**(-9); 6000 N !
    PARAMETER G(LIQUID,FE;0) 298.15 +GHSERFE+12040.17-6.55843*T
        -3.6751551E-21*T**7; 1811 Y
        -10838.83+291.302*T-46*T*LN(T); 6000 N !
    PARAMETER G(LIQUID,AL,FE;0) 298.15 -91976.5+22.1314*T; 6000 N !
    PARAMETER G(LIQUID,AL,FE;1) 298.15 -5672.58+4.8728*T; 6000 N !
    PARAMETER G(LIQUID,AL,FE;2) 298.15 +121.9; 6000 N !
    """)
    return dbf


def profile_workflow_naive(dbf, num_equilibria=100):
    """
    Naive workflow: recreate everything for each equilibrium call.
    This represents the worst-case scenario.
    """
    print(f"\n{'='*60}")
    print(f"NAIVE WORKFLOW: {num_equilibria} equilibria")
    print(f"{'='*60}")

    comps = ['AL', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2']

    # Vary temperature and composition
    temperatures = np.linspace(500, 2000, num_equilibria)
    compositions = np.linspace(0.1, 0.9, num_equilibria)

    start = time.perf_counter()

    for i in range(num_equilibria):
        conds = {v.T: temperatures[i], v.P: 101325, v.X('FE'): compositions[i]}
        eq = equilibrium(dbf, comps, phases, conds, verbose=False)

    elapsed = time.perf_counter() - start
    print(f"Total time: {elapsed:.3f}s")
    print(f"Time per equilibrium: {elapsed/num_equilibria*1000:.2f}ms")
    return elapsed


def profile_workflow_with_reuse(dbf, num_equilibria=100):
    """
    Optimized workflow: reuse models and phase records via Workspace.
    This is the recommended pattern for high-throughput workflows.
    """
    print(f"\n{'='*60}")
    print(f"OPTIMIZED WORKFLOW (Workspace with reuse): {num_equilibria} equilibria")
    print(f"{'='*60}")

    comps = ['AL', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2']

    # Vary temperature and composition
    temperatures = np.linspace(500, 2000, num_equilibria)
    compositions = np.linspace(0.1, 0.9, num_equilibria)

    # Initialize workspace with first condition
    print("Initializing workspace...")
    setup_start = time.perf_counter()
    initial_conds = {v.T: temperatures[0], v.P: 101325, v.X('FE'): compositions[0]}
    wks = Workspace(database=dbf, components=comps, phases=phases,
                    conditions=initial_conds, verbose=False)
    # Force initial equilibrium calculation
    _ = wks.eq
    setup_time = time.perf_counter() - setup_start
    print(f"Setup time: {setup_time:.3f}s")

    start = time.perf_counter()

    for i in range(1, num_equilibria):  # Start from 1 since we did 0 in setup
        # Update conditions (this should be fast)
        wks.conditions[v.T] = temperatures[i]
        wks.conditions[v.X('FE')] = compositions[i]
        # Access equilibrium (triggers recalculation)
        eq = wks.eq

    elapsed = time.perf_counter() - start
    total_time = elapsed + (setup_time if num_equilibria > 1 else 0)
    print(f"Total time (excluding setup): {elapsed:.3f}s")
    print(f"Time per equilibrium: {elapsed/(num_equilibria-1)*1000:.2f}ms")
    return elapsed, setup_time




def profile_calculate_vs_equilibrium(dbf):
    """
    Compare calculate() vs equilibrium() performance.
    calculate() is often faster for sampling but doesn't find equilibria.
    """
    print(f"\n{'='*60}")
    print(f"CALCULATE vs EQUILIBRIUM comparison")
    print(f"{'='*60}")

    comps = ['AL', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2']

    # Test on a smaller grid to avoid memory issues
    T_range = np.linspace(500, 2000, 10)
    X_range = np.linspace(0.1, 0.9, 10)

    # equilibrium() - finds equilibrium at each point
    print("\nRunning equilibrium() on grid...")
    start = time.perf_counter()
    eq_result = equilibrium(dbf, comps, phases, {v.T: T_range, v.P: 101325, v.X('FE'): X_range})
    eq_time = time.perf_counter() - start
    print(f"equilibrium() time: {eq_time:.3f}s")
    print(f"Number of equilibria: {np.prod(eq_result.GM.shape)}")
    print(f"Time per equilibrium: {eq_time/np.prod(eq_result.GM.shape)*1000:.2f}ms")


def detailed_profiling(dbf, num_equilibria=50):
    """
    Run cProfile on the naive workflow to identify bottlenecks.
    """
    print(f"\n{'='*60}")
    print(f"DETAILED PROFILING (cProfile)")
    print(f"{'='*60}")

    comps = ['AL', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2']
    temperatures = np.linspace(500, 2000, num_equilibria)
    compositions = np.linspace(0.1, 0.9, num_equilibria)

    # Profile the naive equilibrium calculations (recreate each time)
    pr = cProfile.Profile()
    pr.enable()

    for i in range(num_equilibria):
        conds = {v.T: temperatures[i], v.P: 101325, v.X('FE'): compositions[i]}
        eq = equilibrium(dbf, comps, phases, conds, verbose=False)

    pr.disable()

    # Print statistics
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(30)  # Top 30 functions

    print("\nTop 30 functions by cumulative time:")
    print(s.getvalue())

    # Save full profile for later analysis
    pr.dump_stats('pycalphad_profile.prof')
    print("\nFull profile saved to: pycalphad_profile.prof")
    print("Analyze with: python -m pstats pycalphad_profile.prof")


def benchmark_object_creation(dbf, num_iterations=10):
    """
    Benchmark the cost of creating models and phase records.
    """
    print(f"\n{'='*60}")
    print(f"OBJECT CREATION BENCHMARKS")
    print(f"{'='*60}")

    comps = ['AL', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2']
    state_vars = {v.T: 1000, v.P: 101325}

    # Benchmark model instantiation
    print(f"\nModel instantiation ({num_iterations} iterations):")
    start = time.perf_counter()
    for _ in range(num_iterations):
        models = instantiate_models(dbf, comps, phases)
    model_time = (time.perf_counter() - start) / num_iterations
    print(f"  Average time: {model_time*1000:.2f}ms")

    # Benchmark PhaseRecordFactory creation
    print(f"\nPhaseRecordFactory creation ({num_iterations} iterations):")
    models = instantiate_models(dbf, comps, phases)
    start = time.perf_counter()
    for _ in range(num_iterations):
        phase_records = PhaseRecordFactory(dbf, comps, state_vars, models)
    prf_time = (time.perf_counter() - start) / num_iterations
    print(f"  Average time: {prf_time*1000:.2f}ms")

    # Benchmark individual PhaseRecord creation (with compilation)
    print(f"\nIndividual PhaseRecord creation (includes LLVM compilation):")
    phase_records = PhaseRecordFactory(dbf, comps, state_vars, models)
    for phase in phases:
        start = time.perf_counter()
        pr = phase_records[phase]
        pr_time = (time.perf_counter() - start)
        print(f"  {phase}: {pr_time*1000:.2f}ms")


def main():
    """Run all profiling benchmarks."""
    print("="*60)
    print("PyCalphad Performance Profiling")
    print("="*60)
    print("\nThis script profiles PyCalphad for high-throughput workflows")
    print("common in ESPEI, kawin, and other applications.\n")

    # Get test database
    dbf = get_test_database()

    # Run benchmarks
    num_eq = 50  # Number of equilibria for main tests

    # 1. Object creation costs
    benchmark_object_creation(dbf, num_iterations=10)

    # 2. Calculate vs Equilibrium
    profile_calculate_vs_equilibrium(dbf)

    # 3. Workflow comparisons
    naive_time = profile_workflow_naive(dbf, num_equilibria=num_eq)
    optimized_time, optimized_setup = profile_workflow_with_reuse(dbf, num_equilibria=num_eq)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Naive workflow:      {naive_time:.3f}s ({naive_time/num_eq*1000:.2f}ms/eq)")
    print(f"Workspace workflow:  {optimized_time:.3f}s ({optimized_time/(num_eq-1)*1000:.2f}ms/eq) [setup: {optimized_setup:.3f}s]")
    print(f"\nSpeedup (naive vs workspace): {naive_time/optimized_time:.2f}x")

    # 4. Detailed profiling
    detailed_profiling(dbf, num_equilibria=num_eq)

    print(f"\n{'='*60}")
    print("Profiling complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
