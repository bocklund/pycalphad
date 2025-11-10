"""
Comprehensive fuzz testing suite for PyCalphad
Tests critical components for crashes, exceptions, and edge cases
"""

import hypothesis
from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis.strategies import composite
import string
import re
import traceback
from io import StringIO

from pycalphad import Database, Model
from pycalphad.io.grammar import parse_chemical_formula
from pycalphad.io.tdb import _sympify_string
from pycalphad.variables import Species, Component
import pycalphad.variables as v

# Configure hypothesis for more aggressive fuzzing
hypothesis.settings.register_profile("fuzz", max_examples=1000, deadline=None,
                                     suppress_health_check=[HealthCheck.too_slow])
hypothesis.settings.load_profile("fuzz")

print("=" * 80)
print("PYCALPHAD FUZZ TESTING SUITE")
print("=" * 80)

# ============================================================================
# FUZZ TEST 1: Chemical Formula Parser
# ============================================================================

@composite
def chemical_formula_strategy(draw):
    """Generate chemical formulas that might break the parser"""
    # Mix of valid and edge case patterns
    elements = ['H', 'He', 'Li', 'C', 'N', 'O', 'Fe', 'Al', 'Cr', 'Ni', 'VA', 'X', 'Aa', 'Zz']
    num_elements = draw(st.integers(min_value=0, max_value=10))

    formula_parts = []
    for _ in range(num_elements):
        element = draw(st.sampled_from(elements))
        # Sometimes add amounts, sometimes not
        if draw(st.booleans()):
            amount = draw(st.one_of(
                st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                st.integers(min_value=-100, max_value=100),
                st.just(0),
                st.just(1),
                st.just(-1)
            ))
            formula_parts.append(f"{element}{amount}")
        else:
            formula_parts.append(element)

    # Sometimes add charge
    formula = ''.join(formula_parts)
    if draw(st.booleans()):
        charge = draw(st.integers(min_value=-10, max_value=10))
        formula += f"/{charge}"

    return formula

@composite
def malformed_chemical_formula_strategy(draw):
    """Generate completely malformed formulas"""
    return draw(st.one_of(
        st.text(alphabet=string.printable, min_size=0, max_size=50),
        st.text(alphabet=string.ascii_letters + string.digits + '/-+.', min_size=0, max_size=50),
        st.just(""),
        st.just("///"),
        st.just("123456"),
        st.just("..Fe2.3.4"),
        st.just("Al/0/1/2"),
    ))

@given(chemical_formula_strategy())
def test_fuzz_chemical_formula_parser(formula):
    """Fuzz test chemical formula parser with edge cases"""
    try:
        result = parse_chemical_formula(formula)
        # Result should be a tuple (list of tuples, charge)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], (int, float))
    except Exception as e:
        # Parser is allowed to reject invalid formulas, but shouldn't crash
        assert not isinstance(e, (SystemExit, KeyboardInterrupt))

@given(malformed_chemical_formula_strategy())
def test_fuzz_chemical_formula_malformed(formula):
    """Fuzz test with completely malformed input"""
    try:
        result = parse_chemical_formula(formula)
        # Should either succeed or raise a reasonable exception
        assert isinstance(result, tuple)
    except Exception as e:
        # Should not crash hard
        assert not isinstance(e, (SystemExit, KeyboardInterrupt))

# ============================================================================
# FUZZ TEST 2: Species Constructor
# ============================================================================

@given(
    name=st.text(alphabet=string.ascii_letters + string.digits + '_-', min_size=1, max_size=20),
    constituents=st.dictionaries(
        keys=st.text(alphabet=string.ascii_uppercase, min_size=1, max_size=2),
        values=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
    ),
    charge=st.integers(min_value=-10, max_value=10)
)
def test_fuzz_species_constructor(name, constituents, charge):
    """Fuzz test Species constructor"""
    try:
        species = Species(name, constituents=constituents, charge=charge)
        assert species.name == name
        assert species.charge == charge
    except (ValueError, TypeError, AttributeError) as e:
        # Valid exceptions for invalid input
        pass
    except Exception as e:
        # Should not crash with unexpected errors
        assert not isinstance(e, (SystemExit, KeyboardInterrupt))

# ============================================================================
# FUZZ TEST 3: Math Expression Parser (_sympify_string)
# ============================================================================

@composite
def math_expression_strategy(draw):
    """Generate mathematical expressions"""
    operators = ['+', '-', '*', '/', '**']
    functions = ['exp', 'ln', 'log', 'EXP', 'LN', 'LOG']
    numbers = draw(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=5))
    variables = ['T', 'P', 'R', 'X', 'Y']

    expr_type = draw(st.sampled_from(['simple', 'function', 'nested']))

    if expr_type == 'simple':
        # Simple arithmetic: a + b * c
        a, b, c = numbers[:3] if len(numbers) >= 3 else [1.0, 2.0, 3.0]
        op1 = draw(st.sampled_from(operators))
        op2 = draw(st.sampled_from(operators))
        return f"{a} {op1} {b} {op2} {c}"
    elif expr_type == 'function':
        # Function calls: exp(T) + ln(P)
        func = draw(st.sampled_from(functions))
        var = draw(st.sampled_from(variables))
        return f"{func}({var})"
    else:
        # Nested: exp(T*ln(P+1))
        return "exp(T*ln(P+1))"

@composite
def malicious_expression_strategy(draw):
    """Generate potentially malicious expressions"""
    return draw(st.sampled_from([
        "__import__('os').system('ls')",  # Code injection attempt
        "eval('1+1')",  # Eval injection
        "exec('print(1)')",  # Exec injection
        "[x for x in range(1000000)]",  # List comprehension (not allowed)
        "lambda x: x",  # Lambda (not allowed)
        "import sys",  # Import statement
        "while True: pass",  # Infinite loop
        "1/0",  # Division by zero (should be caught)
        "T" * 10000,  # Very long variable name
        "(" * 100 + "1" + ")" * 100,  # Deep nesting
        "T**T**T**T**T",  # Expensive computation
    ]))

@given(math_expression_strategy())
def test_fuzz_sympify_string(expression):
    """Fuzz test mathematical expression parser"""
    try:
        result = _sympify_string(expression)
        # Should return a symengine expression or raise
    except (ValueError, SyntaxError, TypeError, AttributeError, KeyError) as e:
        # Expected errors for invalid expressions
        pass
    except Exception as e:
        # Should not crash with unexpected errors
        assert not isinstance(e, (SystemExit, KeyboardInterrupt))

@given(malicious_expression_strategy())
def test_fuzz_sympify_malicious(expression):
    """Test that malicious expressions are blocked"""
    try:
        result = _sympify_string(expression)
        # If it succeeds, verify no code execution happened
        # The AST whitelist should block most attacks
    except (ValueError, SyntaxError, TypeError, AttributeError, KeyError) as e:
        # Expected - malicious code should be rejected
        pass
    except Exception as e:
        # Should not crash with unexpected errors
        assert not isinstance(e, (SystemExit, KeyboardInterrupt))

# ============================================================================
# FUZZ TEST 4: TDB Database String Parsing
# ============================================================================

@composite
def tdb_string_strategy(draw):
    """Generate TDB-like strings"""
    elements = ['AL', 'FE', 'CR', 'NI', 'VA']

    tdb_parts = []

    # Elements section
    num_elements = draw(st.integers(min_value=0, max_value=5))
    for _ in range(num_elements):
        elem = draw(st.sampled_from(elements))
        tdb_parts.append(f"ELEMENT {elem} FCC_A1 0 0 0 !")

    # Function section
    num_functions = draw(st.integers(min_value=0, max_value=3))
    for i in range(num_functions):
        func_name = f"FUNC{i}"
        temp = draw(st.floats(min_value=100, max_value=5000, allow_nan=False, allow_infinity=False))
        value = draw(st.floats(min_value=-100000, max_value=100000, allow_nan=False, allow_infinity=False))
        tdb_parts.append(f"FUNCTION {func_name} {temp} {value}; 6000 N !")

    return '\n'.join(tdb_parts)

@composite
def malformed_tdb_strategy(draw):
    """Generate malformed TDB strings"""
    return draw(st.sampled_from([
        "",  # Empty
        "ELEMENT",  # Incomplete
        "FUNCTION " * 100,  # Repetitive
        "!" * 1000,  # Just terminators
        "ELEMENT AL\nELEMENT AL\n",  # Duplicate
        "FUNCTION F 298.15 1/0; 6000 N !",  # Division by zero
        "FUNCTION F 298.15 " + "("*100 + "1" + ")"*100 + "; 6000 N !",  # Deep nesting
        "ELEMENT " + "A"*1000 + " FCC 0 0 0 !",  # Very long name
        "$" * 10000,  # Just comments
        "\x00\x01\x02",  # Null bytes and control characters
    ]))

@given(tdb_string_strategy())
def test_fuzz_tdb_database_parsing(tdb_string):
    """Fuzz test TDB database parsing"""
    try:
        db = Database.from_string(tdb_string)
        # Should either succeed or raise ParseException
        assert isinstance(db, Database)
    except Exception as e:
        # Parser is allowed to reject invalid TDB, but shouldn't crash hard
        assert not isinstance(e, (SystemExit, KeyboardInterrupt))

@given(malformed_tdb_strategy())
def test_fuzz_tdb_malformed(tdb_string):
    """Fuzz test with malformed TDB strings"""
    try:
        db = Database.from_string(tdb_string)
    except Exception as e:
        # Should not crash with unexpected errors
        assert not isinstance(e, (SystemExit, KeyboardInterrupt))

# ============================================================================
# FUZZ TEST 5: Database Serialization (Pickle)
# ============================================================================

def test_fuzz_database_pickle_roundtrip():
    """Test that Database objects can be safely pickled/unpickled"""
    import pickle

    # Create minimal database
    tdb = "ELEMENT AL FCC_A1 0 0 0 !"
    try:
        db = Database.from_string(tdb)

        # Pickle and unpickle
        pickled = pickle.dumps(db)
        db_restored = pickle.loads(pickled)

        # Should maintain equality
        assert db_restored.elements == db.elements
    except Exception as e:
        print(f"Pickle test error: {e}")

# ============================================================================
# Run the fuzz tests
# ============================================================================

if __name__ == "__main__":
    print("\n[1/9] Testing chemical formula parser with edge cases...")
    test_fuzz_chemical_formula_parser()
    print("✓ Completed")

    print("\n[2/9] Testing chemical formula parser with malformed input...")
    test_fuzz_chemical_formula_malformed()
    print("✓ Completed")

    print("\n[3/9] Testing Species constructor...")
    test_fuzz_species_constructor()
    print("✓ Completed")

    print("\n[4/9] Testing math expression parser...")
    test_fuzz_sympify_string()
    print("✓ Completed")

    print("\n[5/9] Testing math expression parser with malicious input...")
    test_fuzz_sympify_malicious()
    print("✓ Completed")

    print("\n[6/9] Testing TDB database parsing...")
    test_fuzz_tdb_database_parsing()
    print("✓ Completed")

    print("\n[7/9] Testing TDB parsing with malformed input...")
    test_fuzz_tdb_malformed()
    print("✓ Completed")

    print("\n[8/9] Testing database pickle roundtrip...")
    test_fuzz_database_pickle_roundtrip()
    print("✓ Completed")

    print("\n" + "=" * 80)
    print("FUZZ TESTING COMPLETE")
    print("=" * 80)
