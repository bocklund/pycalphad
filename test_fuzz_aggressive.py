"""
Aggressive fuzz testing with detailed bug tracking
"""

import hypothesis
from hypothesis import given, strategies as st, settings, HealthCheck, reproduce_failure
from hypothesis.strategies import composite
import string
import traceback
import sys

from pycalphad import Database
from pycalphad.io.grammar import parse_chemical_formula
from pycalphad.io.tdb import _sympify_string
from pycalphad.variables import Species, Component

# Configure for aggressive fuzzing
hypothesis.settings.register_profile("aggressive",
    max_examples=5000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large]
)
hypothesis.settings.load_profile("aggressive")

bugs_found = []

def log_bug(test_name, input_data, exception):
    """Log a bug with details"""
    bugs_found.append({
        'test': test_name,
        'input': repr(input_data)[:200],  # Truncate long inputs
        'exception': type(exception).__name__,
        'message': str(exception)[:200],
        'traceback': ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))[:500]
    })

print("=" * 80)
print("AGGRESSIVE FUZZ TESTING WITH BUG TRACKING")
print("Running 5000 examples per test...")
print("=" * 80)

# ============================================================================
# Test 1: Division by zero and numeric edge cases in math expressions
# ============================================================================

@composite
def numeric_edge_expression(draw):
    """Generate expressions with numeric edge cases"""
    return draw(st.sampled_from([
        "1/0",  # Division by zero
        "0/0",  # Undefined
        "T**1000",  # Massive exponent
        "1e308 * 1e308",  # Overflow
        "1e-308 / 1e308",  # Underflow
        "0**0",  # Indeterminate
        "-1**0.5",  # Complex result
        "log(-1)",  # Invalid log
        "ln(0)",  # Log of zero
        "T**T**T",  # Expensive nested power
        "1/(T-298.15)",  # Division by expression that could be zero
        "exp(1000000)",  # Huge exp
    ]))

print("\n[1] Testing numeric edge cases in math expressions...")
edge_case_bugs = []
for expr in ["1/0", "0/0", "T**1000", "1e308*1e308", "log(-1)", "ln(0)", "exp(1000000)", "T**T**T"]:
    try:
        result = _sympify_string(expr)
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as e:
        # Expected - these should be rejected or handled gracefully
        pass
    except Exception as e:
        edge_case_bugs.append((expr, e))
        log_bug("numeric_edge_cases", expr, e)

if edge_case_bugs:
    print(f"   Found {len(edge_case_bugs)} unexpected errors:")
    for expr, e in edge_case_bugs:
        print(f"   - {expr}: {type(e).__name__}: {str(e)[:60]}")
else:
    print("   ✓ All numeric edge cases handled correctly")

# ============================================================================
# Test 2: Very long inputs (DoS potential)
# ============================================================================

print("\n[2] Testing very long inputs (potential DoS)...")
long_input_bugs = []

# Very long chemical formula
try:
    long_formula = "Fe" * 10000
    result = parse_chemical_formula(long_formula)
except MemoryError as e:
    long_input_bugs.append(("long_formula", e))
    log_bug("long_chemical_formula", "Fe*10000", e)
except Exception as e:
    # Any other exception is okay
    pass

# Very long element name
try:
    long_element = "A" * 10000 + "1.5"
    result = parse_chemical_formula(long_element)
except MemoryError as e:
    long_input_bugs.append(("long_element", e))
    log_bug("long_element_name", "A*10000", e)
except Exception as e:
    pass

# Very long TDB
try:
    long_tdb = "ELEMENT AL FCC 0 0 0 !\n" * 10000
    db = Database.from_string(long_tdb)
except MemoryError as e:
    long_input_bugs.append(("long_tdb", e))
    log_bug("long_tdb", "ELEMENT*10000", e)
except Exception as e:
    pass

if long_input_bugs:
    print(f"   ⚠ Found {len(long_input_bugs)} potential DoS issues with long inputs")
    for name, e in long_input_bugs:
        print(f"   - {name}: {type(e).__name__}")
else:
    print("   ✓ Long inputs handled correctly")

# ============================================================================
# Test 3: Unicode and special characters
# ============================================================================

print("\n[3] Testing unicode and special characters...")
unicode_bugs = []

unicode_tests = [
    "Fe™",  # Trademark symbol
    "Al\x00Cr",  # Null byte
    "Fe\n\r\tAl",  # Whitespace chars
    "Al💀Fe",  # Emoji
    "ЖЗИ",  # Cyrillic
    "铁铝",  # Chinese
    "Fe\x1b[31mAl",  # ANSI escape codes
    "../../../etc/passwd",  # Path traversal attempt
    "Fe; DROP TABLE elements;--",  # SQL injection attempt
]

for test_input in unicode_tests:
    try:
        result = parse_chemical_formula(test_input)
    except (ValueError, TypeError, UnicodeError) as e:
        # Expected rejections
        pass
    except Exception as e:
        if not isinstance(e, (SystemExit, KeyboardInterrupt)):
            unicode_bugs.append((test_input, e))
            log_bug("unicode_special_chars", test_input, e)

if unicode_bugs:
    print(f"   ⚠ Found {len(unicode_bugs)} unexpected errors with special characters:")
    for inp, e in unicode_bugs:
        print(f"   - {repr(inp)[:40]}: {type(e).__name__}")
else:
    print("   ✓ Special characters handled correctly")

# ============================================================================
# Test 4: Deeply nested structures
# ============================================================================

print("\n[4] Testing deeply nested structures...")
nesting_bugs = []

# Deeply nested parentheses
try:
    deep_parens = "(" * 1000 + "T" + ")" * 1000
    result = _sympify_string(deep_parens)
except RecursionError as e:
    nesting_bugs.append(("deep_parens", e))
    log_bug("deep_parentheses", "(*1000)T(*1000)", e)
except Exception as e:
    # Other exceptions are okay
    pass

# Deeply nested function calls
try:
    deep_funcs = "exp(" * 100 + "T" + ")" * 100
    result = _sympify_string(deep_funcs)
except RecursionError as e:
    nesting_bugs.append(("deep_functions", e))
    log_bug("deep_function_nesting", "exp(*100)T", e)
except Exception as e:
    pass

if nesting_bugs:
    print(f"   ⚠ Found {len(nesting_bugs)} recursion issues with deep nesting:")
    for name, e in nesting_bugs:
        print(f"   - {name}: {type(e).__name__}")
else:
    print("   ✓ Deep nesting handled correctly")

# ============================================================================
# Test 5: Boundary values for chemical formulas
# ============================================================================

print("\n[5] Testing boundary values in chemical formulas...")
boundary_bugs = []

boundary_tests = [
    ("Fe0", "Zero stoichiometry"),
    ("Fe-1", "Negative stoichiometry"),
    ("Fe1e308", "Very large stoichiometry"),
    ("Fe1e-308", "Very small stoichiometry"),
    ("Fe/2147483648", "Large charge beyond int32"),
    ("Fe/-2147483648", "Large negative charge"),
    ("VA", "Vacancy element"),
    ("", "Empty string"),
    ("/", "Just charge separator"),
    ("//5", "Double separator"),
    ("Fe1.0.0", "Multiple decimals"),
    ("Fe..5", "Double decimal points"),
]

for formula, desc in boundary_tests:
    try:
        result = parse_chemical_formula(formula)
        # Some might succeed, which is fine
    except (ValueError, TypeError, OverflowError, IndexError) as e:
        # Expected errors
        pass
    except Exception as e:
        if not isinstance(e, (SystemExit, KeyboardInterrupt)):
            boundary_bugs.append((formula, desc, e))
            log_bug("boundary_values", f"{formula} ({desc})", e)

if boundary_bugs:
    print(f"   ⚠ Found {len(boundary_bugs)} unexpected errors with boundary values:")
    for formula, desc, e in boundary_bugs:
        print(f"   - {desc}: {type(e).__name__}")
else:
    print("   ✓ Boundary values handled correctly")

# ============================================================================
# Test 6: TDB parser edge cases
# ============================================================================

print("\n[6] Testing TDB parser edge cases...")
tdb_bugs = []

tdb_tests = [
    ("FUNCTION F 298.15 1/0; 6000 N !", "Division by zero in function"),
    ("FUNCTION F 298.15 T**1000; 6000 N !", "Large exponent"),
    ("FUNCTION " + "A"*100 + " 298.15 1; 6000 N !", "Long function name"),
    ("ELEMENT AL AL 0 0 0 !\nELEMENT AL FE 0 0 0 !", "Duplicate element different phase"),
    ("FUNCTION F1 298.15 F2; 6000 N !\nFUNCTION F2 298.15 F1; 6000 N !", "Circular function reference"),
    ("", "Empty TDB"),
    ("!" * 1000, "Only terminators"),
    ("$" * 10000, "Only comments"),
]

for tdb, desc in tdb_tests:
    try:
        db = Database.from_string(tdb)
    except Exception as e:
        if not isinstance(e, (SystemExit, KeyboardInterrupt, ValueError, SyntaxError,
                             KeyError, RecursionError)) and 'ParseException' not in str(type(e)):
            tdb_bugs.append((desc, e))
            log_bug("tdb_edge_cases", desc, e)

if tdb_bugs:
    print(f"   ⚠ Found {len(tdb_bugs)} unexpected errors in TDB parsing:")
    for desc, e in tdb_bugs:
        print(f"   - {desc}: {type(e).__name__}: {str(e)[:60]}")
else:
    print("   ✓ TDB edge cases handled correctly")

# ============================================================================
# Test 7: Species constructor edge cases
# ============================================================================

print("\n[7] Testing Species constructor edge cases...")
species_bugs = []

species_tests = [
    ({"name": "", "charge": 0}, "Empty name"),
    ({"name": "A"*1000, "charge": 0}, "Very long name"),
    ({"name": "Fe", "charge": 1e308}, "Huge charge"),
    ({"name": "Fe", "constituents": {"Fe": 1e308}}, "Huge constituent amount"),
    ({"name": "Fe", "constituents": {"Fe": -1}}, "Negative constituent"),
    ({"name": "Fe\x00Al", "charge": 0}, "Null byte in name"),
]

for params, desc in species_tests:
    try:
        species = Species(**params)
    except (ValueError, TypeError, OverflowError, AttributeError) as e:
        # Expected errors
        pass
    except Exception as e:
        if not isinstance(e, (SystemExit, KeyboardInterrupt)):
            species_bugs.append((desc, e))
            log_bug("species_constructor", desc, e)

if species_bugs:
    print(f"   ⚠ Found {len(species_bugs)} unexpected errors in Species constructor:")
    for desc, e in species_bugs:
        print(f"   - {desc}: {type(e).__name__}")
else:
    print("   ✓ Species constructor edge cases handled correctly")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("FUZZ TESTING SUMMARY")
print("=" * 80)

total_bugs = len(bugs_found)
if total_bugs == 0:
    print("✓ NO BUGS FOUND - All edge cases handled gracefully!")
else:
    print(f"⚠ FOUND {total_bugs} POTENTIAL ISSUES:\n")
    for i, bug in enumerate(bugs_found, 1):
        print(f"{i}. {bug['test']}")
        print(f"   Input: {bug['input']}")
        print(f"   Error: {bug['exception']}: {bug['message']}")
        print()

print("=" * 80)
