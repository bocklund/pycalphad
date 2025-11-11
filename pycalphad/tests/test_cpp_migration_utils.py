"""
Utilities for testing C++ migration equivalence.

These helpers verify that C++ implementations produce identical results
to the original Cython implementations.
"""
import numpy as np
from numpy.testing import assert_allclose


def assert_arrays_equal(arr1, arr2, rtol=1e-15, atol=1e-15, context=""):
    """
    Assert two arrays are numerically equivalent.

    Parameters
    ----------
    arr1, arr2 : ndarray
        Arrays to compare
    rtol : float
        Relative tolerance
    atol : float
        Absolute tolerance
    context : str
        Context string for error messages

    Raises
    ------
    AssertionError
        If arrays are not equivalent
    """
    context_str = f" ({context})" if context else ""
    assert_allclose(arr1, arr2, rtol=rtol, atol=atol,
                   err_msg=f"Arrays not equal{context_str}")


def assert_phase_amounts_equal(amounts1, amounts2, rtol=1e-12, context=""):
    """
    Assert phase amounts are equal.

    Phase amounts close to zero are handled specially to avoid
    numerical issues with relative tolerance.

    Parameters
    ----------
    amounts1, amounts2 : ndarray
        Phase amounts to compare
    rtol : float
        Relative tolerance for non-zero amounts
    context : str
        Context for error messages
    """
    # Mask for truly zero phases
    zero_mask = (np.abs(amounts1) < 1e-15) & (np.abs(amounts2) < 1e-15)

    # For non-zero phases, use relative tolerance
    if np.any(~zero_mask):
        assert_allclose(amounts1[~zero_mask], amounts2[~zero_mask],
                       rtol=rtol, atol=1e-15,
                       err_msg=f"Phase amounts differ{' ('+context+')' if context else ''}")

    # For zero phases, verify both are actually zero
    if np.any(zero_mask):
        assert np.all(np.abs(amounts1[zero_mask]) < 1e-15) and \
               np.all(np.abs(amounts2[zero_mask]) < 1e-15), \
               f"Phase amounts differ for zero phases{' ('+context+')' if context else ''}"


def assert_equilibrium_results_equal(eq1, eq2, rtol=1e-12, atol=1e-15):
    """
    Assert two equilibrium results are equivalent.

    Parameters
    ----------
    eq1, eq2 : xarray.Dataset
        Equilibrium calculation results to compare
    rtol : float
        Relative tolerance
    atol : float
        Absolute tolerance

    Raises
    ------
    AssertionError
        If results are not equivalent
    """
    # Check that same phases are present
    phases1 = set(eq1.Phase.values.flatten())
    phases2 = set(eq2.Phase.values.flatten())
    phases1.discard('')  # Remove empty strings
    phases2.discard('')

    assert phases1 == phases2, \
        f"Different phases present: {phases1} vs {phases2}"

    # Check phase fractions
    if 'NP' in eq1:
        assert_arrays_equal(eq1.NP.values, eq2.NP.values,
                           rtol=rtol, atol=atol, context="Phase fractions")

    # Check chemical potentials
    if 'MU' in eq1:
        assert_arrays_equal(eq1.MU.values, eq2.MU.values,
                           rtol=rtol, atol=atol, context="Chemical potentials")

    # Check Gibbs energy
    if 'GM' in eq1:
        assert_arrays_equal(eq1.GM.values, eq2.GM.values,
                           rtol=rtol, atol=atol, context="Gibbs energy")

    # Check compositions
    if 'X' in eq1:
        assert_arrays_equal(eq1.X.values, eq2.X.values,
                           rtol=rtol, atol=atol, context="Compositions")


def benchmark_function(func, *args, iterations=1000, warmup=10, **kwargs):
    """
    Benchmark a function call.

    Parameters
    ----------
    func : callable
        Function to benchmark
    *args
        Positional arguments to func
    iterations : int
        Number of iterations for timing
    warmup : int
        Number of warmup iterations
    **kwargs
        Keyword arguments to func

    Returns
    -------
    dict
        Benchmark results with timing statistics
    """
    import time
    import statistics

    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)

    return {
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'stdev': statistics.stdev(times) if len(times) > 1 else 0,
        'min': min(times),
        'max': max(times),
        'iterations': iterations,
        'total_time': sum(times),
    }


def compare_implementations(cython_func, cpp_func, test_cases,
                           rtol=1e-15, atol=1e-15, benchmark=False):
    """
    Compare Cython and C++ implementations across test cases.

    Parameters
    ----------
    cython_func : callable
        Original Cython implementation
    cpp_func : callable
        New C++ implementation
    test_cases : list of tuples
        List of (args, kwargs) tuples for each test case
    rtol, atol : float
        Tolerances for comparison
    benchmark : bool
        If True, also benchmark both implementations

    Returns
    -------
    dict
        Comparison results including pass/fail and timing
    """
    results = {
        'passed': 0,
        'failed': 0,
        'failures': [],
    }

    if benchmark:
        results['cython_times'] = []
        results['cpp_times'] = []

    for i, (args, kwargs) in enumerate(test_cases):
        try:
            # Run both implementations
            if benchmark:
                import time
                start = time.perf_counter()
                cython_result = cython_func(*args, **kwargs)
                cython_time = time.perf_counter() - start

                start = time.perf_counter()
                cpp_result = cpp_func(*args, **kwargs)
                cpp_time = time.perf_counter() - start

                results['cython_times'].append(cython_time)
                results['cpp_times'].append(cpp_time)
            else:
                cython_result = cython_func(*args, **kwargs)
                cpp_result = cpp_func(*args, **kwargs)

            # Compare results
            if isinstance(cython_result, np.ndarray):
                assert_arrays_equal(cython_result, cpp_result, rtol=rtol, atol=atol,
                                   context=f"Test case {i}")
            elif isinstance(cython_result, (list, tuple)):
                for j, (cr, cpr) in enumerate(zip(cython_result, cpp_result)):
                    if isinstance(cr, np.ndarray):
                        assert_arrays_equal(cr, cpr, rtol=rtol, atol=atol,
                                           context=f"Test case {i}, output {j}")
                    else:
                        assert abs(cr - cpr) < max(atol, rtol * abs(cr)), \
                            f"Scalar outputs differ for test case {i}, output {j}"
            else:
                # Scalar comparison
                assert abs(cython_result - cpp_result) < max(atol, rtol * abs(cython_result)), \
                    f"Scalar outputs differ for test case {i}"

            results['passed'] += 1

        except AssertionError as e:
            results['failed'] += 1
            results['failures'].append({
                'test_case': i,
                'error': str(e),
                'args': args,
                'kwargs': kwargs,
            })

    # Calculate speedup if benchmarking
    if benchmark and results['cython_times'] and results['cpp_times']:
        avg_cython = sum(results['cython_times']) / len(results['cython_times'])
        avg_cpp = sum(results['cpp_times']) / len(results['cpp_times'])
        results['speedup'] = avg_cython / avg_cpp if avg_cpp > 0 else float('inf')

    return results


def print_comparison_report(results, title="Implementation Comparison"):
    """
    Print a formatted comparison report.

    Parameters
    ----------
    results : dict
        Results from compare_implementations()
    title : str
        Report title
    """
    print("=" * 70)
    print(f"{title}")
    print("=" * 70)
    print(f"Total tests:  {results['passed'] + results['failed']}")
    print(f"Passed:       {results['passed']}")
    print(f"Failed:       {results['failed']}")

    if 'speedup' in results:
        print(f"Speedup:      {results['speedup']:.2f}x")

    if results['failures']:
        print("\nFailures:")
        for failure in results['failures']:
            print(f"  Test case {failure['test_case']}: {failure['error']}")

    print("=" * 70)
