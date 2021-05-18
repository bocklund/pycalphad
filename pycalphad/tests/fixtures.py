from pytest import fixture
from pycalphad import Database
from pycalphad.tests.datasets import *


@fixture(scope='session')
def ALCOCRNI_DBF():
    yield Database(ALCOCRNI_TDB)

@fixture(scope='session')
def ALCRNI_DBF():
    yield Database(ALCRNI_TDB)

@fixture(scope='session')
def ALFE_DBF():
    yield Database(ALFE_TDB)

@fixture(scope='session')
def ALNIFCC4SL_DBF():
    yield Database(ALNIFCC4SL_TDB)

@fixture(scope='session')
def ALNIPT_DBF():
    yield Database(ALNIPT_TDB)

@fixture(scope='session')
def AL_C_FE_B2_DBF():
    yield Database(AL_C_FE_B2_TDB)

@fixture(scope='session')
def AL_PARAMETER_DBF():
    yield Database(AL_PARAMETER_TDB)

@fixture(scope='session')
def CRFE_DBF():
    yield Database(CRFE_BCC_MAGNETIC_TDB)

@fixture(scope='session')
def CUMG_DBF():
    yield Database(CUMG_TDB)

@fixture(scope='session')
def CUMG_PARAMETERS_DBF():
    yield Database(CUMG_PARAMETERS_TDB)

@fixture(scope='session')
def CUO_DBF():
    yield Database(CUO_TDB)

@fixture(scope='session')
def C_FE_DBF():
    yield Database(C_FE_BROSHE_TDB)

@fixture(scope='session')
def FEMN_DBF():
    yield Database(FEMN_TDB)

@fixture(scope='session')
def FE_MN_S_DBF():
    yield Database(FE_MN_S_TDB)

@fixture(scope='session')
def ISSUE43_DBF():
    yield Database(ISSUE43_TDB)

@fixture(scope='session')
def NI_AL_DUPIN_2001_DBF():
    yield Database(NI_AL_DUPIN_2001_TDB)

@fixture(scope='session')
def PBSN_DBF():
    yield Database(PBSN_TDB)

@fixture(scope='session')
def ROSE_DBF():
    yield Database(ROSE_TDB)

@fixture(scope='session')
def TOUGH_CHEMPOT_DBF():
    yield Database(ALNI_TOUGH_CHEMPOT_TDB)

@fixture(scope='session')
def VA_INTERACTION_DBF():
    yield Database(VA_INTERACTION_TDB)
