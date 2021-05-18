"""
The calculate test module verifies that calculate() calculates
Model quantities correctly.
"""

import pytest
from pycalphad import Database, calculate, Model
import numpy as np
from numpy.testing import assert_allclose
from pycalphad import ConditionError

from .fixtures import ALCRNI_DBF, ALFE_DBF, CUMG_PARAMETERS_DBF

def test_surface():
    "Bare minimum: calculation produces a result."
    calculate(ALCRNI_DBF, ['AL', 'CR', 'NI'], 'L12_FCC',
                T=1273., mode='numpy')

def test_unknown_model_attribute():
    "Sampling an unknown model attribute raises exception."
    with pytest.raises(AttributeError):
        calculate(ALCRNI_DBF, ['AL', 'CR', 'NI'], 'L12_FCC', T=1400.0, output='_fail_')

def test_statevar_upcast():
    "Integer state variable values are cast to float."
    calculate(ALCRNI_DBF, ['AL', 'CR', 'NI'], 'L12_FCC',
                T=1273, mode='numpy')

def test_points_kwarg_multi_phase():
    "Multi-phase calculation works when internal dof differ (gh-41)."
    calculate(ALCRNI_DBF, ['AL', 'CR', 'NI'], ['L12_FCC', 'LIQUID'],
                T=1273, points={'L12_FCC': [0.20, 0.05, 0.75, 0.05, 0.20, 0.75]}, mode='numpy')

def test_issue116():
    "Calculate gives correct result when a state variable is left as default (gh-116)."
    result_one = calculate(ALCRNI_DBF, ['AL', 'CR', 'NI'], 'LIQUID', T=400)
    result_one_values = result_one.GM.values
    result_two = calculate(ALCRNI_DBF, ['AL', 'CR', 'NI'], 'LIQUID', T=400, P=101325)
    result_two_values = result_two.GM.values
    result_three = calculate(ALCRNI_DBF, ['AL', 'CR', 'NI'], 'LIQUID', T=400, P=101325, N=1)
    result_three_values = result_three.GM.values
    np.testing.assert_array_equal(np.squeeze(result_one_values), np.squeeze(result_two_values))
    np.testing.assert_array_equal(np.squeeze(result_one_values), np.squeeze(result_three_values))
    # N is added automatically
    assert len(result_one_values.shape) == 3  # N, T, points
    assert result_one_values.shape[0] == 1
    assert len(result_two_values.shape) == 4  # N, P, T, points
    assert result_two_values.shape[:3] == (1, 1, 1)
    assert len(result_three_values.shape) == 4  # N, P, T, points
    assert result_three_values.shape[:3] == (1, 1, 1)


def test_calculate_some_phases_filtered():
    """
    Phases are filtered out from calculate() when some cannot be built.
    """
    # should not raise; AL13FE4 should be filtered out
    calculate(ALFE_DBF, ['AL', 'VA'], ['FCC_A1', 'AL13FE4'], T=1200, P=101325)


def test_calculate_raises_with_no_active_phases_passed():
    """Passing inactive phases to calculate() raises a ConditionError."""
    # Phase cannot be built without FE
    with pytest.raises(ConditionError):
        calculate(ALFE_DBF, ['AL', 'VA'], ['AL13FE4'], T=1200, P=101325)


def test_calculate_with_parameters_vectorized():
    # Second set of parameter values are directly copied from the TDB
    parameters = {'VV0000': [-33134.699474175846, -32539.5], 'VV0001': [7734.114029426941, 8236.3],
                  'VV0002': [-13498.542175596054, -14675.0], 'VV0003': [-26555.048975092268, -24441.2],
                  'VV0004': [20777.637577083482, 20149.6], 'VV0005': [41915.70425630003, 46500.0],
                  'VV0006': [-34525.21964215504, -39591.3], 'VV0007': [95457.14639216446, 104160.0],
                  'VV0008': [21139.578967453144, 21000.0], 'VV0009': [19047.833726419598, 17772.0],
                  'VV0010': [20468.91829601273, 21240.0], 'VV0011': [19601.617855958328, 14321.1],
                  'VV0012': [-4546.9325861738, -4923.18], 'VV0013': [-1640.6354331231278, -1962.8],
                  'VV0014': [-35682.950005357634, -31626.6]}
    res = calculate(CUMG_PARAMETERS_DBF, ['CU', 'MG'], ['HCP_A3'], parameters=parameters, T=743.15, P=1e5)
    res_noparams = calculate(CUMG_PARAMETERS_DBF, ['CU', 'MG'], ['HCP_A3'], parameters=None, T=743.15, P=1e5)
    param_values = []
    for symbol in sorted(parameters.keys()):
        param_values.append(parameters[symbol])
    param_values = np.array(param_values).T
    assert all(res['param_symbols'] == sorted([str(x) for x in parameters.keys()]))
    assert_allclose(np.squeeze(res['param_values'].values), param_values)
    assert_allclose(res.GM.isel(samples=1).values, res_noparams.GM.values)


def test_incompatible_model_instance_raises():
    "Calculate raises when an incompatible Model instance built with a different phase is passed."
    comps = ['AL', 'CR', 'NI']
    phase_name = 'L12_FCC'
    mod = Model(ALCRNI_DBF, comps, 'LIQUID')  # Model instance does not match the phase
    with pytest.raises(ValueError):
        calculate(ALCRNI_DBF, comps, phase_name, T=1400.0, output='_fail_', model=mod)


def test_single_model_instance_raises():
    "Calculate raises when a single Model instance is passed with multiple phases."
    comps = ['AL', 'CR', 'NI']
    phase_name = 'L12_FCC'
    mod = Model(ALCRNI_DBF, comps, 'L12_FCC')  # Model instance does not match the phase
    with pytest.raises(ValueError):
        calculate(ALCRNI_DBF, comps, ['LIQUID', 'L12_FCC'], T=1400.0, output='_fail_', model=mod)
