from importlib.resources import files

import pytest

import pycalphad.tests.databases
from pycalphad import Database
from pycalphad.io.cs_dat import TokenParser, TokenParserError, parse_header


@pytest.mark.parametrize(
    ("extras", "blank_lines"),
    [("", ""), ("  7  8", ""), ("\t7   8  9   ", "\n  \t")],
)
def test_parse_header_discards_only_second_line_extra_tokens(extras, blank_lines):
    toks = TokenParser(
        "title\n"
        f"2 1 1 0{extras}\n"
        f"{blank_lines}AL NI\n"
        "26.9815 58.6934\n"
        "2 1 2\n"
        "2 3 4\n"
    )

    assert toks.parse(str) == "title"
    header = parse_header(toks)

    assert header.list_soln_species_count == [1]
    assert header.num_stoich_phases == 0
    assert header.pure_elements == ["AL", "NI"]
    assert header.pure_elements_mass == [26.9815, 58.6934]
    assert header.gibbs_coefficient_idxs == [1, 2]
    assert header.excess_coefficient_idxs == [3, 4]


def test_parse_header_extras_do_not_hide_later_malformed_line():
    toks = TokenParser("title\n1 1 1 0 7 8\nAL\nnot-a-mass\n")

    assert toks.parse(str) == "title"
    with pytest.raises(TokenParserError, match="not-a-mass"):
        parse_header(toks)


def _dat_text(name):
    return files(pycalphad.tests.databases).joinpath(name).read_text()


def _mqmqa_constituents(dbf, phase):
    return {s.name: dict(s.constituents) for sublattice in dbf.phases[phase].constituents for s in sublattice}


@pytest.mark.parametrize("database, phase, expected", [
    ("Shishin_Fe-Sb-O-S_slag.dat", "SLAG-LIQ", {"FE2++2.0": {"FE": 1.0}, "FE3++3.0": {"FE": 1.0}, "SB3++3.0": {"SB": 1.0}}),
    ("ClAlNa.dat", "MSSOLN", {"AL+3.0": {"AL": 1.0}, "AL2+6.0": {"AL": 2.0}, "VA-1.0": {"VA": 1.0}}),
    ("FeTiVO.dat", "SLAGBSOLN", {"FE[2+]+2.0": {"FE": 1.0}, "TI[4+]+4.0": {"TI": 1.0}, "V2O3+4.0": {"V": 2.0, "O": 3.0}}),
])
def test_mqmqa_constituent_compositions_come_from_pair_records(database, phase, expected):
    """MQMQA constituents take their elements from the pair records, not their labels."""
    constituents = _mqmqa_constituents(Database.from_string(_dat_text(database), fmt="dat"), phase)
    for name, composition in expected.items():
        assert constituents[name] == composition


def test_mqmqa_constituent_labelled_with_other_elements_than_its_pair_records():
    """A constituent labelled K whose pair record is LiF is the Li constituent."""
    text = _dat_text("Ocadiz-Flores.dat").replace(" Li                       Ni\n", " K                        Ni\n")
    with pytest.warns(UserWarning, match="Constituent K of phase LIQUID is named LI"):
        dbf = Database.from_string(text, fmt="dat")
    assert _mqmqa_constituents(dbf, "LIQUID")["LI+1.0"] == {"LI": 1.0}


def test_mqmqa_inconsistent_pair_records_raise():
    """Pair records that disagree on the constituents raise."""
    record = " NaCl                         0.00    0.00\n   4  2           1.0    0.0    1.0\n"
    text = _dat_text("ClAlNa.dat").replace(record, record.replace("1.0    0.0    1.0", "2.0    0.0    1.0"))
    with pytest.raises(ValueError, match="disagree on the CL content"):
        Database.from_string(text, fmt="dat")


def test_mqmqa_pair_records_out_of_pair_index_order_warn():
    """Pair records out of pair-index order warn."""
    text = _dat_text("Shishin_Fe-Sb-O-S_slag.dat")
    i, j, k = (text.index(f"\n {name}\n") for name in ("FeO", "Fe2O3", "FeS"))
    with pytest.warns(UserWarning, match="not charge-balanced"):
        Database.from_string(text[:i] + text[j:k] + text[i:j] + text[k:], fmt="dat")


def test_numbers_after_a_species_name_warn():
    """Nonzero numbers after a species name warn."""
    text = _dat_text("Shishin_Fe-Sb-O-S_slag.dat").replace("\n FeO\n", "\n FeO                       0.500000    0.418400E+07\n")
    with pytest.warns(UserWarning, match="Ignoring the numbers"):
        Database.from_string(text, fmt="dat")
