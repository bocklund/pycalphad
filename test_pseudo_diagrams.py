"""
Test script to create and visualize pseudobinary/pseudoternary phase diagrams
"""
import numpy as np
import matplotlib.pyplot as plt
from pycalphad import Database, variables as v
from pycalphad.mapping import plot_binary, plot_ternary
from pycalphad.mapping.strategy.binary_strategy import BinaryStrategy
from pycalphad.mapping.strategy.ternary_strategy import TernaryStrategy
from pycalphad.core.utils import is_pseudobinary, is_pseudoternary

# Create a simple test database for pseudobinary system
PSEUDOBINARY_TDB = """
$ Test database for pseudobinary/ternary phase diagrams
$ System: Al-Mg-Si
ELEMENT AL   FCC_A1    26.982    4540.      28.30    !
ELEMENT MG   HCP_A3    24.305    4998.      32.68    !
ELEMENT SI   DIAMOND_A4 28.085   3217.      18.81    !
ELEMENT VA   VACUUM     0.0       0.0        0.0      !

FUNCTION GHSERAL 298.15
    -7976.15+137.093038*T-24.3671976*T*LN(T)
    -.001884662*T**2-8.77664E-07*T**3+74092*T**(-1); 700.00 Y
    -11276.24+223.048446*T-38.5844296*T*LN(T)
    +.018531982*T**2-5.764227E-06*T**3+74092*T**(-1); 933.47 Y
    -11278.378+188.684153*T-31.748192*T*LN(T)
    -1.230524E+28*T**(-9); 2900.00 N !

FUNCTION GHSERMG 298.15
    -8367.34+143.675547*T-26.1849782*T*LN(T)
    +4.858E-04*T**2-1.393669E-06*T**3+78950*T**(-1); 923.00 Y
    -14130.185+204.716215*T-34.3088*T*LN(T)
    +1.038192E+28*T**(-9); 3000.00 N !

FUNCTION GHSERSI 298.15
    -8162.609+137.236859*T-22.8317533*T*LN(T)
    -.001912904*T**2-3.552E-09*T**3+176667*T**(-1); 1687.00 Y
    -9457.642+167.281367*T-27.196*T*LN(T)
    -4.20369E+30*T**(-9); 3600.00 N !

$ Liquid phase
PHASE LIQUID:L % 1 1.0 !
CONSTITUENT LIQUID:L :AL,MG,SI: !
PARAMETER G(LIQUID,AL;0) 298.15 +GHSERAL+11005.029
    -11.841867*T+7.934E-20*T**7; 933.47 Y
    +GHSERAL+10482.382-11.253974*T+1.231E+28*T**(-9); 2900.00 N !
PARAMETER G(LIQUID,MG;0) 298.15 +GHSERMG+8202.243
    -8.83693*T-8.01759E-20*T**7; 923.00 Y
    +GHSERMG+8690.316-9.392158*T-1.03819E+28*T**(-9); 3000.00 N !
PARAMETER G(LIQUID,SI;0) 298.15 +GHSERSI+50696.36
    -30.099439*T+2.0931E-21*T**7; 1687.00 Y
    +GHSERSI+49828.165-29.559068*T+4.20369E+30*T**(-9); 3600.00 N !
PARAMETER G(LIQUID,AL,MG;0) 298.15 -2000; 6000 N !
PARAMETER G(LIQUID,AL,SI;0) 298.15 -11655.93+1.79993*T; 6000 N !
PARAMETER G(LIQUID,MG,SI;0) 298.15 -5000; 6000 N !

$ FCC phase (Al-rich)
PHASE FCC_A1 % 1 1.0 !
CONSTITUENT FCC_A1 :AL,MG,SI: !
PARAMETER G(FCC_A1,AL;0) 298.15 +GHSERAL; 2900.00 N !
PARAMETER G(FCC_A1,MG;0) 298.15 +GHSERMG+2600-.9*T; 3000.00 N !
PARAMETER G(FCC_A1,SI;0) 298.15 +GHSERSI+51000-21.8*T; 3600.00 N !
PARAMETER G(FCC_A1,AL,SI;0) 298.15 -3143.78+.39297*T; 6000 N !

$ HCP phase (Mg-rich)
PHASE HCP_A3 % 1 1.0 !
CONSTITUENT HCP_A3 :AL,MG,SI: !
PARAMETER G(HCP_A3,AL;0) 298.15 +GHSERAL+5481-1.8*T; 2900.00 N !
PARAMETER G(HCP_A3,MG;0) 298.15 +GHSERMG; 3000.00 N !
PARAMETER G(HCP_A3,SI;0) 298.15 +GHSERSI+49200-20.8*T; 3600.00 N !

$ Diamond (Si-rich)
PHASE DIAMOND_A4 % 1 1.0 !
CONSTITUENT DIAMOND_A4 :AL,MG,SI: !
PARAMETER G(DIAMOND_A4,AL;0) 298.15 +GHSERAL+30*T; 2900.00 N !
PARAMETER G(DIAMOND_A4,MG;0) 298.15 +GHSERMG+30*T; 3000.00 N !
PARAMETER G(DIAMOND_A4,SI;0) 298.15 +GHSERSI; 3600.00 N !
"""

def test_pseudobinary_with_elements():
    """
    Test a pseudobinary-like system using pure elements
    System: Al-Mg-Si with fixed Al content (effectively Mg-Si binary at fixed Al)
    """
    print("\n" + "="*80)
    print("TEST 1: Pseudobinary-like system (Al-Mg-Si with varying Mg-Si)")
    print("="*80)

    dbf = Database(PSEUDOBINARY_TDB)

    # This creates a pseudobinary by fixing one composition
    # Varying Mg from 0 to 1 at constant Al = 0.2
    # This is effectively a binary section through ternary space
    components = ['AL', 'MG', 'SI', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'HCP_A3', 'DIAMOND_A4']

    # Isopleth conditions: fix X(AL)=0.2, vary X(MG) and T
    conditions = {
        v.T: (600, 1200, 10),
        v.X('AL'): 0.2,
        v.X('MG'): (0, 0.8, 0.02),
        v.P: 101325,
        v.N: 1
    }

    print(f"Components: {components}")
    print(f"Phases: {phases}")
    print(f"Conditions: T=(600-1200K), X(AL)=0.2, X(MG)=(0-0.8)")

    # Create and run strategy
    strategy = BinaryStrategy(dbf, components, phases, conditions)

    # Check detection
    axis_vars = strategy.axis_vars
    print(f"Axis variables: {axis_vars}")
    print(f"Pure elements: {strategy.elements}")
    print(f"Is pseudobinary: {strategy.is_pseudobinary}")

    strategy.do_map()

    print(f"Mapping complete:")
    print(f"  - ZPF lines found: {len(strategy.zpf_lines)}")
    print(f"  - Nodes found: {len(strategy.node_queue.nodes)}")

    # Plot the diagram
    fig, ax = plt.subplots(figsize=(10, 8))
    ax = plot_binary(strategy, x=v.X('MG'), y=v.T, ax=ax)
    ax.set_title('Pseudobinary-like Phase Diagram: Al-Mg-Si at X(Al)=0.2')
    plt.tight_layout()
    plt.savefig('/tmp/pseudobinary_almgsi.png', dpi=150, bbox_inches='tight')
    print("Saved plot to: /tmp/pseudobinary_almgsi.png")
    plt.close()

    return strategy

def test_ternary_section():
    """
    Test a ternary section
    System: Al-Mg-Si ternary at constant temperature
    """
    print("\n" + "="*80)
    print("TEST 2: Ternary section (Al-Mg-Si at T=800K)")
    print("="*80)

    dbf = Database(PSEUDOBINARY_TDB)

    components = ['AL', 'MG', 'SI', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'HCP_A3', 'DIAMOND_A4']

    conditions = {
        v.T: 800,
        v.X('MG'): (0, 1, 0.05),
        v.X('SI'): (0, 1, 0.05),
        v.P: 101325,
        v.N: 1
    }

    print(f"Components: {components}")
    print(f"Phases: {phases}")
    print(f"Conditions: T=800K, X(MG)=(0-1), X(SI)=(0-1)")

    strategy = TernaryStrategy(dbf, components, phases, conditions)

    # Check detection
    axis_vars = strategy.axis_vars
    print(f"Axis variables: {axis_vars}")
    print(f"Pure elements: {strategy.elements}")
    print(f"Is pseudoternary: {strategy.is_pseudoternary}")

    strategy.do_map()

    print(f"Mapping complete:")
    print(f"  - ZPF lines found: {len(strategy.zpf_lines)}")
    print(f"  - Nodes found: {len(strategy.node_queue.nodes)}")

    # Plot the diagram
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='triangular')
    ax = plot_ternary(strategy, x=v.X('SI'), y=v.X('MG'), ax=ax)
    ax.set_title('Ternary Phase Diagram: Al-Mg-Si at T=800K')
    plt.tight_layout()
    plt.savefig('/tmp/ternary_almgsi.png', dpi=150, bbox_inches='tight')
    print("Saved plot to: /tmp/ternary_almgsi.png")
    plt.close()

    return strategy

def test_simple_binary():
    """
    Test a simple binary for comparison
    System: Al-Si binary
    """
    print("\n" + "="*80)
    print("TEST 3: Regular binary system (Al-Si)")
    print("="*80)

    dbf = Database(PSEUDOBINARY_TDB)

    components = ['AL', 'SI', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'DIAMOND_A4']

    conditions = {
        v.T: (600, 1800, 10),
        v.X('SI'): (0, 1, 0.02),
        v.P: 101325,
        v.N: 1
    }

    print(f"Components: {components}")
    print(f"Phases: {phases}")
    print(f"Conditions: T=(600-1800K), X(SI)=(0-1)")

    strategy = BinaryStrategy(dbf, components, phases, conditions)

    # Check detection
    axis_vars = strategy.axis_vars
    print(f"Axis variables: {axis_vars}")
    print(f"Pure elements: {strategy.elements}")
    print(f"Is pseudobinary: {strategy.is_pseudobinary}")

    strategy.do_map()

    print(f"Mapping complete:")
    print(f"  - ZPF lines found: {len(strategy.zpf_lines)}")
    print(f"  - Nodes found: {len(strategy.node_queue.nodes)}")

    # Plot the diagram
    fig, ax = plt.subplots(figsize=(10, 8))
    ax = plot_binary(strategy, x=v.X('SI'), y=v.T, ax=ax)
    ax.set_title('Binary Phase Diagram: Al-Si')
    plt.tight_layout()
    plt.savefig('/tmp/binary_alsi.png', dpi=150, bbox_inches='tight')
    print("Saved plot to: /tmp/binary_alsi.png")
    plt.close()

    return strategy

if __name__ == '__main__':
    print("\n" + "#"*80)
    print("# Testing Pseudobinary/Pseudoternary Phase Diagram Support")
    print("#"*80)

    try:
        # Run tests
        strategy1 = test_simple_binary()
        strategy2 = test_pseudobinary_with_elements()
        strategy3 = test_ternary_section()

        print("\n" + "="*80)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nGenerated plots:")
        print("  1. /tmp/binary_alsi.png - Regular Al-Si binary")
        print("  2. /tmp/pseudobinary_almgsi.png - Pseudobinary section at X(Al)=0.2")
        print("  3. /tmp/ternary_almgsi.png - Ternary section at T=800K")

    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
