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


def _mqmqa_constituents(dbf, phase):
    return {s.name: dict(s.constituents) for sublattice in dbf.phases[phase].constituents for s in sublattice}


def _load(name):
    return Database(files(pycalphad.tests.databases).joinpath(name))


@pytest.mark.parametrize("database, phase, expected", [
    ("Shishin_Fe-Sb-O-S_slag.dat", "SLAG-LIQ", {"FE2++2.0": {"FE": 1.0}, "FE3++3.0": {"FE": 1.0}, "SB3++3.0": {"SB": 1.0}}),
    ("ClAlNa.dat", "MSSOLN", {"AL+3.0": {"AL": 1.0}, "AL2+6.0": {"AL": 2.0}, "VA-1.0": {"VA": 1.0}}),
    ("Be-F-Li.dat", "MSFL", {"BE+2.0": {"BE": 1.0}, "BE2+4.0": {"BE": 2.0}}),
    ("FeTiVO.dat", "SLAGBSOLN", {"FE[2+]+2.0": {"FE": 1.0}, "TI[4+]+4.0": {"TI": 1.0}, "V2O3+4.0": {"V": 2.0, "O": 3.0}}),
    ("Starykh_Cu-O-Si-Sn.dat", "LIQ(MATTE_METAL)", {"CU+1.0": {"CU": 1.0}, "CU2++1.0": {"CU": 1.0}}),
    ("Ocadiz-Flores.dat", "LIQUID", {"LI+1.0": {"LI": 1.0}}),  # labelled K, but its pair record is LiF
])
def test_mqmqa_constituent_compositions_come_from_pair_records(database, phase, expected):
    """Element content of SUBG/SUBQ constituents comes from the pair records; labels are only names."""
    constituents = _mqmqa_constituents(_load(database), phase)
    for name, composition in expected.items():
        assert constituents[name] == composition


@pytest.mark.parametrize("database, phase, groups", [
    ("ClAlNa.dat", "MSSOLN", {"NA+1.0": 1, "AL+3.0": 2, "AL2+6.0": 2, "CL-1.0": 1, "VA-1.0": 1}),
    ("Be-F-Li.dat", "MSFL", {"LI+1.0": 1, "BE+2.0": 2, "BE2+4.0": 2, "F-1.0": 1}),
])
def test_mqmqa_chemical_groups_follow_constituent_order(database, phase, groups):
    """Chemical groups are assigned in the order the constituents are listed, not sorted by name."""
    hints = _load(database).phases[phase].model_hints["mqmqa"]["chemical_groups"]
    assigned = {s.name: g for sublattice in ("cations", "anions") for s, g in hints[sublattice].items()}
    assert assigned == groups


def test_mqmqa_pair_records_out_of_pair_index_order_warn():
    """Record k fills pair-index column k even when it belongs elsewhere (as in Thermochimica); the parser warns."""
    lines = files(pycalphad.tests.databases).joinpath("KF-NIF2_switched.dat").read_text().split("\n")
    nif2 = [i for i, line in enumerate(lines) if line.strip() == "NiF2"][0]
    kf = [i for i, line in enumerate(lines) if line.strip() == "KF"][0]
    end = [i for i, line in enumerate(lines) if line.split() == ["2", "1"]][0]
    swapped = "\n".join(lines[:nif2] + lines[kf:end] + lines[nif2:kf] + lines[end:])
    with pytest.warns(UserWarning, match="Pair record (KF|NIF2) of phase LIQUID2"):
        dbf = Database.from_string(swapped, fmt="dat")
    assert _mqmqa_constituents(dbf, "LIQUID2")["K+2.0"] == {"K": 1.0}  # the KF record in the Ni2+ column
