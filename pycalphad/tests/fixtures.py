from pytest import fixture
from pycalphad import Database
from pycalphad.tests.datasets import *


@fixture(scope='session')
def ALCOCRNI_DBF():
    Database(ALCOCRNI_TDB)

@fixture(scope='session')
def ALCRNI_DBF():
    Database(ALCRNI_TDB)

@fixture(scope='session')
def ALFE_DBF():
    Database(ALFE_TDB)

@fixture(scope='session')
def ALNIFCC4SL_DBF():
    Database(ALNIFCC4SL_TDB)

@fixture(scope='session')
def ALNIPT_DBF():
    Database(ALNIPT_TDB)

@fixture(scope='session')
def AL_C_FE_B2_DBF():
    Database(AL_C_FE_B2_TDB)

@fixture(scope='session')
def AL_PARAMETER_DBF():
    Database(AL_PARAMETER_TDB)

@fixture(scope='session')
def CRFE_DBF():
    Database(CRFE_BCC_MAGNETIC_TDB)

@fixture(scope='session')
def CUMG_DBF():
    Database(CUMG_TDB)

@fixture(scope='session')
def CUMG_PARAMETERS_DBF():
    Database(CUMG_PARAMETERS_TDB)

@fixture(scope='session')
def CUO_DBF():
    Database(CUO_TDB)

@fixture(scope='session')
def C_FE_DBF():
    Database(C_FE_BROSHE_TDB)

@fixture(scope='session')
def FEMN_DBF():
    Database(FEMN_TDB)

@fixture(scope='session')
def FE_MN_S_DBF():
    Database(FE_MN_S_TDB)

@fixture(scope='session')
def ISSUE43_DBF():
    Database(ISSUE43_TDB)

@fixture(scope='session')
def NI_AL_DUPIN_2001_DBF():
    Database(NI_AL_DUPIN_2001_TDB)

@fixture(scope='session')
def PBSN_DBF():
    Database(PBSN_TDB)

@fixture(scope='session')
def ROSE_DBF():
    Database(ROSE_TDB)

@fixture(scope='session')
def TOUGH_CHEMPOT_DBF():
    Database(ALNI_TOUGH_CHEMPOT_TDB)

@fixture(scope='session')
def VA_INTERACTION_DBF():
    Database(VA_INTERACTION_TDB)
