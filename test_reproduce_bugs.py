"""
Reproduce and investigate the bugs found during fuzz testing
"""

import traceback
from pycalphad.io.tdb import _sympify_string

print("=" * 80)
print("REPRODUCING BUGS FOUND DURING FUZZ TESTING")
print("=" * 80)

bugs = [
    ("1/0", "Division by zero"),
    ("0/0", "Indeterminate form (0/0)"),
    ("ln(0)", "Logarithm of zero"),
]

for expr, description in bugs:
    print(f"\n{description}: {expr}")
    print("-" * 40)
    try:
        result = _sympify_string(expr)
        print(f"Result: {result}")
        print(f"Type: {type(result)}")
    except Exception as e:
        print(f"Exception: {type(e).__name__}: {e}")
        print("\nFull traceback:")
        traceback.print_exc()

# Additional edge cases to test
print("\n" + "=" * 80)
print("TESTING ADDITIONAL EDGE CASES")
print("=" * 80)

additional_tests = [
    "1.0/0.0",
    "T/0",
    "log(0)",
    "exp(T)/0",
    "0**0",
    "T**(-1)",
]

for expr in additional_tests:
    print(f"\nExpression: {expr}")
    try:
        result = _sympify_string(expr)
        print(f"  ✓ Result: {result}")
    except Exception as e:
        print(f"  ✗ {type(e).__name__}: {e}")
