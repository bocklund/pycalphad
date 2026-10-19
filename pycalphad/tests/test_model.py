"""
The test_model module contains unit tests for the Model object.
"""
from pycalphad import Database, Model, variables as v, equilibrium
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.codegen.sympydiff_utils import build_functions, build_constraint_functions
from pycalphad.tests.fixtures import select_database, load_database
from pycalphad.core.errors import DofError
from symengine import Piecewise
import copy
import os
import subprocess
import sys
import textwrap
import numpy as np
import pickle
import pytest


@select_database("alcrni.tdb")
def test_model_eq(load_database):
    "Model equality comparison."
    dbf = load_database()
    test_model = Model(dbf, ['AL', 'CR'], 'L12_FCC')
    assert test_model == test_model
    assert test_model == Model(dbf, ['AL', 'CR'], 'L12_FCC')
    assert not (test_model == Model(dbf, ['NI', 'CR'], 'L12_FCC'))
    # literals which don't have __dict__
    assert not (test_model == 42)
    assert not (test_model == None)
    assert not (42 == test_model)
    assert not (None == test_model)


@select_database("alcrni.tdb")
def test_model_ne(load_database):
    "Model inequality comparison."
    dbf = load_database()
    test_model = Model(dbf, ['AL', 'CR'], 'L12_FCC')
    assert not (test_model != test_model)
    assert not (test_model != Model(dbf, ['AL', 'CR'], 'L12_FCC'))
    assert test_model != Model(dbf, ['NI', 'CR'], 'L12_FCC')
    # literals which don't have __dict__
    assert test_model != 42
    assert test_model != None
    assert 42 != test_model
    assert None != test_model


@select_database("alnipt.tdb")
def test_export_import(load_database):
    "Equivalence of Model using re-imported database."
    dbf = load_database()
    test_model = Model(Database.from_string(dbf.to_string(fmt='tdb', if_incompatible='ignore'), fmt='tdb'), ['PT', 'NI', 'VA'], 'FCC_L12')
    ref_model = Model(dbf, ['NI', 'PT', 'VA'], 'FCC_L12')
    assert test_model == ref_model


@select_database("alnipt.tdb")
def test_model_pickle(load_database):
    "Model pickle roundtrip."
    dbf = load_database()
    test_model = Model(dbf, ['NI', 'PT', 'VA'], 'FCC_L12')
    new_model = pickle.loads(pickle.dumps(test_model))
    assert test_model == new_model


@select_database("alcrni.tdb")
def test_custom_model_contributions(load_database):
    "Building a custom model using contributions."
    dbf = load_database()
    class CustomModel(Model):
        contributions = [('zzz', 'test'), ('xxx', 'test2'), ('yyy', 'test3')]
        def test(self, dbe):
            return 0
        def test2(self, dbe):
            return 0
        def test3(self, dbe):
            return 0
    CustomModel(dbf, ['AL', 'CR'], 'L12_FCC')


@select_database("alfe.tdb")
def test_degree_of_ordering(load_database):
    "Degree of ordering should be calculated properly."
    dbf = load_database()
    my_phases = ['B2_BCC']
    comps = ['AL', 'FE', 'VA']
    conds = {v.T: 300, v.P: 101325, v.X('AL'): 0.25}
    eqx = equilibrium(dbf, comps, my_phases, conds, output='degree_of_ordering', verbose=True)
    print('Degree of ordering: {}'.format(eqx.degree_of_ordering.values.flatten()))
    assert np.isclose(eqx.degree_of_ordering.values.flatten(), np.array([0.6666]), atol=1e-4)

def test_detect_pure_vacancy_phases():
    "Detecting a pure vacancy phase"
    ZRO2_CUBIC_BCC_TDB = """
        ELEMENT /-   ELECTRON_GAS              0.0000E+00  0.0000E+00  0.0000E+00!
        ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00!
        ELEMENT AL   FCC_A1                    2.6982E+01  4.5773E+03  2.8321E+01!
        ELEMENT CU   FCC_A1                    6.3546E+01  5.0041E+03  3.3150E+01!
        ELEMENT O    1/2_MOLE_O2(G)            1.5999E+01  4.3410E+03  1.0252E+02!
        ELEMENT ZR   HCP_A3                    9.1224E+01  5.5663E+03  3.9181E+01!

        PHASE BCC_A2  %  2 1   3 !
            CONSTITUENT BCC_A2  :AL,CU,ZR : O,VA% :  !

        PHASE ZRO2_CUBIC  %  2 1   2 !
            CONSTITUENT ZRO2_CUBIC  :VA,ZR% : O%,VA :  !
    """
    dbf = Database(ZRO2_CUBIC_BCC_TDB)
    with pytest.raises(DofError):
        Model(dbf,['AL','CU','VA'],'ZRO2_CUBIC')


@select_database("parameter_filter_test.tdb")
def test_constituents_not_in_model(load_database):
    """Test that parameters with constituent arrays not matching the phase model are filtered out correctly"""
    dbf = load_database()
    modA = Model(dbf, ['A', 'B'], 'ALPHA')
    modB = Model(dbf, ['B', 'C'], 'BETA')
    assert v.SiteFraction('ALPHA', 0, 'B') not in modA.ast.free_symbols
    assert v.SiteFraction('BETA', 1, 'D') not in modB.ast.free_symbols
    assert v.SiteFraction('BETA', 2, 'C') not in modB.ast.free_symbols


def test_order_disorder_interstital_sublattice_validation():
    # Check that substitutional/interstitial sublattices that break our
    # assumptions raise errors
    DBF_OrderDisorder_broken = Database("""
    ELEMENT VA   VACUUM   0.0000E+00  0.0000E+00  0.0000E+00 !
    ELEMENT A    DISORD     0.0000E+00  0.0000E+00  0.0000E+00 !
    ELEMENT B    DISORD     0.0000E+00  0.0000E+00  0.0000E+00 !
    ELEMENT C    DISORD     0.0000E+00  0.0000E+00  0.0000E+00 !

    DEFINE_SYSTEM_DEFAULT ELEMENT 2 !
    DEFAULT_COMMAND DEF_SYS_ELEMENT VA !

    TYPE_DEFINITION % SEQ *!
    TYPE_DEFINITION ' GES A_P_D ORD_MORE_INSTL DIS_PART DISORD ,,,!
    TYPE_DEFINITION & GES A_P_D ORD_LESS_INSTL DIS_PART DISORD ,,,!
    TYPE_DEFINITION ) GES A_P_D ORD_SUBS_INSTL DIS_PART DISORD ,,,!

    PHASE DISORD  %  2 1   3 !
    CONSTITUENT DISORD  : A,B,VA : VA :  !

    $ Has one more interstitial sublattice than disordered:
    PHASE ORD_MORE_INSTL %'  4 0.5  0.5  3 1 !
    CONSTITUENT ORD_MORE_INSTL  : A,B,VA : A,B,VA : VA : A : !

    $ Has one less interstitial sublattice than disordered:
    PHASE ORD_LESS_INSTL %&  2 0.5  0.5 !
    CONSTITUENT ORD_LESS_INSTL  : A,B,VA : A,B,VA : !

    $ The interstitial sublattice has the same species as the substitutional
    $ and cannot be distinguished:
    PHASE ORD_SUBS_INSTL %)  3 0.5  0.5 3 !
    CONSTITUENT ORD_SUBS_INSTL  : A,B,VA : A,B,VA : A,B,VA :  !

    """)

    # Case 1: Ordered phase has one more interstitial sublattice than disordered
    with pytest.raises(ValueError):
        Model(DBF_OrderDisorder_broken, ["A", "B", "VA"], "ORD_MORE_INSTL")

    # Case 2: Ordered phase has one more interstitial sublattice than disordered
    with pytest.raises(ValueError):
        Model(DBF_OrderDisorder_broken, ["A", "B", "VA"], "ORD_LESS_INSTL")

    # Case 3: The ordered phase has interstitial sublattice has the same species
    # as the substitutional and cannot be distinguished
    with pytest.raises(ValueError):
        Model(DBF_OrderDisorder_broken, ["A", "B", "VA"], "ORD_SUBS_INSTL")

@pytest.mark.filterwarnings("ignore:The order-disorder model for \"ORD_FCC\" has a contribution from the physical property model*:UserWarning")
@select_database("FeNi_deep_branching.tdb")
def test_model_deep_branching(load_database):
    "Models with very deep piecewise branching are optimized at construction time"
    dbf = load_database()
    mod = Model(dbf, ['FE', 'NI', 'VA'], 'ORD_FCC')
    # All we really care about is that the energy Hessian will calculate without hanging
    # However, that is a relatively long test. This just checks that the deep branches were cleaned up.
    # Without optimization/unwrapping, this would be about 57
    assert len(mod.GM.atoms(Piecewise)) < 30

def test_model_extrapolate_temperature():
    "Models extrapolate temperature bounds outside upper/lower limits"
    TDB_extrapolate = """
    ELEMENT VA   VACUUM   0.0000E+00  0.0000E+00  0.0000E+00 !
    ELEMENT A    DISORD     0.0000E+00  0.0000E+00  0.0000E+00 !
    ELEMENT B    DISORD     0.0000E+00  0.0000E+00  0.0000E+00 !

    FUNCTION GSYM1  500 +100;  5000.00 Y 40000; 10000 N !

    DEFINE_SYSTEM_DEFAULT ELEMENT 2 !
    DEFAULT_COMMAND DEF_SYS_ELEMENT VA !

    TYPE_DEFINITION % SEQ *!
    TYPE_DEFINITION ' GES A_P_D ORDERED DIS_PART DISORD ,,,!

    PHASE DISORD  %  2 1   3 !
    PHASE ORDERED %'  3 0.5  0.5  3  !

    CONSTITUENT DISORD  : A,B : VA :  !
    CONSTITUENT ORDERED  : A,B: A,B : VA :  !

    PARAMETER G(DISORD,A:VA;0)  298.15  -10000+GSYM1#; 1000 Y -3000-GSYM1#; 6000 N !
    PARAMETER G(DISORD,B:VA;0)  298.15  -10000+GSYM1#; 1000 Y -3000-GSYM1#; 6000 N !

    """
    dbf = Database(TDB_extrapolate)
    # First, confirm that turning the extrapolation off reproduces the correct behavior
    class modtype(Model):
        extrapolate_temperature_bounds = False
    mod = modtype(dbf, ['A', 'B', 'VA'], 'ORDERED')
    # Remove ideal mixing effects so we can test extrapolation easily
    mod.models['idmix'] = 0
    dof = {v.Y('ORDERED', 0, 'A'): 0.5, v.Y('ORDERED', 0, 'B'): 0.5,
               v.Y('ORDERED', 1, 'A'): 0.5, v.Y('ORDERED', 1, 'B'): 0.5,
               v.Y('ORDERED', 2, 'VA'): 1.0}
    assert mod.GM.subs(dof).subs({v.T: 200}).n(real=True) == 0.
    assert mod.GM.subs(dof).subs({v.T: 50000}).n(real=True) == 0.

    # Next, test that the default behavior (extrapolation) works as intended
    mod = Model(dbf, ['A', 'B', 'VA'], 'ORDERED')
    # Remove ideal mixing effects so we can test extrapolation easily
    mod.models['idmix'] = 0
    dof = {v.Y('ORDERED', 0, 'A'): 0.5, v.Y('ORDERED', 0, 'B'): 0.5,
               v.Y('ORDERED', 1, 'A'): 0.5, v.Y('ORDERED', 1, 'B'): 0.5,
               v.Y('ORDERED', 2, 'VA'): 1.0}
    assert mod.GM.subs(dof).subs({v.T: 200}).n(real=True) == -9900.
    assert mod.GM.subs(dof).subs({v.T: 50000}).n(real=True) == -43000.

def test_error_raised_for_higher_order_reciprocal_parameter():
    """
    The composition dependent reciprocal parameter on more than two sublattices
    should not be supported and should raise an error if a database has these parameters
    """
    test_tdb = """
    ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00!
    ELEMENT C    GRAPHITE                  1.2011E+01  1.0540E+03  5.7423E+00!
    ELEMENT MO   BCC_A2                    9.5940E+01  4.5890E+03  2.8560E+01!
    ELEMENT NB   BCC_A2                    9.2906E+01  5.2200E+03  3.6270E+01!
    ELEMENT AL   BCC_A2                    9.2906E+01  5.2200E+03  3.6270E+01!

    PHASE PHASE_CORRECT % 3 1 1 1 !
    CONSTITUENT PHASE_CORRECT : MO,NB : NB,AL : C,VA : !

    PARAMETER G(PHASE_CORRECT,MO,NB:NB,AL:C,VA;0) 298.15 -300000; 6000 N !

    PHASE PHASE_SUBLATTICE % 3 1 1 1 !
    CONSTITUENT PHASE_SUBLATTICE : MO,NB : NB,AL : C,VA : !

    PARAMETER G(PHASE_SUBLATTICE,MO,NB:NB,AL:C,VA;1) 298.15 -300000; 6000 N !

    PHASE PHASE_HIGH_ORDER % 2 1 1 !
    CONSTITUENT PHASE_HIGH_ORDER : MO,NB : C,VA : !

    PARAMETER G(PHASE_HIGH_ORDER,MO,NB:C,VA;3) 298.15 -300000; 6000 N !
    """
    dbf = Database(test_tdb)
    # Check that a model with 0th order reciprocal parameter loads correctly
    mod = Model(dbf, ["AL", "MO", "NB", "C", "VA"], "PHASE_CORRECT")

    # Check that a model with a higher order reciprocal parameter on three sublattices throws an error
    with pytest.raises(ValueError):
        mod = Model(dbf, ["AL", "MO", "NB", "C", "VA"], "PHASE_SUBLATTICE")

    # Check that a model with a higher order reciprocal parameter > 2 throws an error
    with pytest.raises(ValueError):
        mod = Model(dbf, ["MO", "NB", "C", "VA"], "PHASE_HIGH_ORDER")


def test_redundant_charge_balance_constraint_is_not_added():
    """Charge balance is skipped when it is implied by the site fraction balances (gh-712)"""
    # A charge balance constraint that is a linear combination of the site fraction
    # balance constraints makes the phase matrix in the energy minimizer singular, so it
    # must not be added. That happens whenever every constituent of a sublattice carries
    # the same charge, since then the charge of each sublattice is a fixed multiple of
    # that sublattice's site fraction sum.
    CHARGED_TDB = """
     ELEMENT /-   ELECTRON_GAS              0.0000E+00  0.0000E+00  0.0000E+00!
     ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00!
     ELEMENT O    1/2_MOLE_O2(G)            1.5999E+01  4.3410E+03  1.0252E+02!
     ELEMENT TI   HCP_A3                    4.7880E+01  4.8240E+03  3.0720E+01!

     SPECIES O-2                         O1/-2!
     SPECIES VA-2                        VA1/-2!
     SPECIES TI+2                        TI1/+2!
     SPECIES TI+3                        TI1/+3!
     SPECIES TI+4                        TI1/+4!

     $ Stoichiometric ionic compound: charge balance is redundant
     PHASE TIO_ALPHA % 2 1 1 !
     CONSTITUENT TIO_ALPHA : TI+2 : O-2 : !

     $ Mixing on the anion sublattice, but both constituents have the same charge,
     $ so charge balance is still redundant
     PHASE RUTILE % 2 1 2 !
     CONSTITUENT RUTILE : TI+4 : O-2,VA-2 : !

     $ Mixed valence on the cation sublattice: charge balance is independent
     PHASE B1_OXIDE % 2 1 1 !
     CONSTITUENT B1_OXIDE : TI+2,TI+3 : O-2 : !

     $ Neutral vacancies break the constant-charge-per-sublattice pattern
     PHASE HALITE % 2 1 1 !
     CONSTITUENT HALITE : TI+2,VA : O-2 : !
    """
    dbf = Database(CHARGED_TDB)
    comps = ["TI", "O", "VA"]

    def constraint_jacobian(mod):
        constraints = mod.get_internal_constraints()
        return np.array([[float(cons.diff(dof)) for dof in mod.site_fractions] for cons in constraints])

    for phase_name, expect_charge_balance in [("TIO_ALPHA", False), ("RUTILE", False),
                                              ("B1_OXIDE", True), ("HALITE", True)]:
        mod = Model(dbf, comps, phase_name)
        constraints = mod.get_internal_constraints()
        num_sublattices = len(mod.constituents)
        assert len(constraints) == num_sublattices + int(expect_charge_balance), \
            f"{phase_name} should{'' if expect_charge_balance else ' not'} have a charge balance constraint"
        # The internal constraints must always be linearly independent, otherwise the
        # phase matrix in the energy minimizer is singular
        jac = constraint_jacobian(mod)
        assert np.linalg.matrix_rank(jac) == jac.shape[0], f"{phase_name} constraints are linearly dependent"


def test_inconsistent_charge_balance_constraint_is_kept():
    """A phase that can never be electroneutral keeps its (infeasible) charge balance constraint"""
    # The charge balance of (LA+3)2(O-2)6 is linearly dependent on the site fraction
    # balances, but unlike the redundant case it is also inconsistent with them: the net
    # charge is -6 for any site fractions. Dropping it would silently turn an impossible
    # phase into a feasible one, so it is kept and the infeasibility is detected when no
    # feasible site fractions can be sampled.
    INFEASIBLE_TDB = """
     ELEMENT /-   ELECTRON_GAS              0.0000E+00  0.0000E+00  0.0000E+00!
     ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00!
     ELEMENT O    1/2_MOLE_O2(G)            1.5999E+01  4.3410E+03  1.0252E+02!
     ELEMENT LA   DHCP                      1.3891E+02  6.6665E+03  5.6902E+01!

     SPECIES O-2                         O1/-2!
     SPECIES LA+3                        LA1/+3!

     PHASE IMPOSSIBLE % 2 2 6 !
     CONSTITUENT IMPOSSIBLE : LA+3 : O-2 : !
    """
    dbf = Database(INFEASIBLE_TDB)
    mod = Model(dbf, ["LA", "O", "VA"], "IMPOSSIBLE")
    constraints = mod.get_internal_constraints()
    assert len(constraints) == len(mod.constituents) + 1


def _phase_record_outputs(dbf, comps, phase_name, num_points=10):
    """Everything a PhaseRecord evaluates, at reproducible random points

    The functions are compiled from scratch (not taken from the build_functions cache), as in a new process.
    """
    build_functions.cache_clear()
    build_constraint_functions.cache_clear()
    mod = Model(dbf, comps, phase_name)
    prx = PhaseRecordFactory(dbf, comps, {v.N, v.P, v.T}, {phase_name: mod}).get(phase_name)
    num_vars = prx.num_statevars + prx.phase_dof
    rng = np.random.default_rng(1769)
    outputs = []
    for _ in range(num_points):
        dof = rng.uniform(size=num_vars)
        dof[:3] = [1.0, 101325.0, 300.0 + 2000.0 * dof[2]]  # N, P, T
        sublattice_indices = np.array([site_fraction.sublattice_index for site_fraction in mod.site_fractions])
        for sublattice_index in np.unique(sublattice_indices):
            columns = 3 + np.nonzero(sublattice_indices == sublattice_index)[0]
            dof[columns] /= dof[columns].sum()
        for func, shape in [(prx.obj, (1,)), (prx.formulaobj, (1,)), (prx.formulagrad, (num_vars,)),
                            (prx.formulahess, (num_vars, num_vars)), (prx.internal_cons_func, (prx.num_internal_cons,)),
                            (prx.internal_cons_jac, (prx.num_internal_cons, num_vars)),
                            (prx.internal_cons_hess, (prx.num_internal_cons, num_vars, num_vars))]:
            out = np.zeros(shape)
            func(out, dof)
            outputs.append(out)
        for comp_idx in range(len(prx.nonvacant_elements)):
            for func, shape in [(prx.mass_obj, (1,)), (prx.formulamole_obj, (1,)), (prx.formulamole_grad, (num_vars,)),
                                (prx.formulamole_hess, (num_vars, num_vars))]:
                out = np.zeros(shape)
                func(out, dof, comp_idx)
                outputs.append(out)
    return outputs


def _assert_same_phase_records(dbf, other_dbf, comps):
    for phase_name in sorted(dbf.phases.keys()):
        expected = _phase_record_outputs(dbf, comps, phase_name)
        actual = _phase_record_outputs(other_dbf, comps, phase_name)
        for expected_values, actual_values in zip(expected, actual):
            np.testing.assert_array_equal(actual_values, expected_values, err_msg=phase_name)


@pytest.mark.filterwarnings("ignore:The order-disorder model for .* has a contribution from the physical property model:UserWarning")
@pytest.mark.parametrize("load_database, comps", [
    ("alni_dupin_2001.tdb", ["AL", "NI", "VA"]),
    ("Al-Fe_sundman2009.tdb", ["AL", "FE", "VA"]),
    ("Ocadiz-Flores.dat", ["F", "K", "LI", "NA", "NI", "VA"]),  # MQMQA liquids
], indirect=["load_database"])
def test_parameter_order_does_not_change_phase_records(load_database, comps):
    """Databases that differ only in the order of their parameters give bit-identical phase records

    Compiled functions add terms in the order that SymEngine stores them, which follows the order the Model
    added them. Since build_functions caches by equality, which ignores that order, the result for one
    database could otherwise also depend on whether an equal energy from another database was compiled first.
    """
    dbf = load_database()
    reordered_dbf = copy.deepcopy(dbf)
    parameters = reordered_dbf._parameters.all()
    reordered_dbf._parameters.truncate()
    reordered_dbf._parameters.insert_multiple([dict(param) for param in reversed(parameters)])
    _assert_same_phase_records(dbf, reordered_dbf, comps)


@pytest.mark.filterwarnings("ignore:The order-disorder model for .* has a contribution from the physical property model:UserWarning")
@pytest.mark.parametrize("load_database, comps", [
    ("alni_dupin_2001.tdb", ["AL", "NI", "VA"]),
    ("Al-Fe_sundman2009.tdb", ["AL", "FE", "VA"]),
    ("crtiv_ghosh.tdb", ["CR", "TI", "V", "VA"]),
], indirect=["load_database"])
def test_tdb_roundtrip_does_not_change_phase_records(load_database, comps):
    """A database written to and read from a TDB gives bit-identical phase records

    The TDB writer orders parameters and the terms of expressions differently than the original file.
    """
    dbf = load_database()
    roundtrip_dbf = Database.from_string(dbf.to_string(fmt='tdb', if_incompatible='ignore'), fmt='tdb')
    assert roundtrip_dbf == dbf
    _assert_same_phase_records(dbf, roundtrip_dbf, comps)


def test_phase_records_do_not_depend_on_hash_seed():
    """Phase records are bit-identical between processes with different PYTHONHASHSEED values

    Covers parameters generated from sublattice symmetry options (Al-Fe BCC_4SL) and the multiple
    Einstein temperatures of LNTHETA parameters (graphite), which were added in set iteration order.
    """
    code = textwrap.dedent("""
        import hashlib, warnings
        from importlib.resources import files
        from pycalphad import Database
        import pycalphad.tests.databases
        from pycalphad.tests.test_model import _phase_record_outputs
        warnings.filterwarnings("ignore")
        digest = hashlib.sha256()
        for db_name, comps, phase_name in [("Al-Fe_sundman2009.tdb", ["AL", "FE", "VA"], "BCC_4SL"),
                                           ("WC-22He-3g.tdb", ["C", "VA", "W"], "GRAPHITE_A9")]:
            dbf = Database(files(pycalphad.tests.databases).joinpath(db_name))
            for values in _phase_record_outputs(dbf, comps, phase_name):
                digest.update(values.tobytes())
        print(digest.hexdigest())
    """)
    processes = [subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True,
                                  env={**os.environ, "PYTHONHASHSEED": str(seed)})
                 for seed in range(3)]
    digests = set()
    for process in processes:
        stdout, _ = process.communicate()
        assert process.returncode == 0
        digests.add(stdout.strip())
    assert len(digests) == 1
