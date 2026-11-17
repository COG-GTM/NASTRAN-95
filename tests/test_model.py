from conftest import ROOT
from nastran95py.bulk import nastran_real, read_deck
from nastran95py.model import build_model


def test_nastran_real_formats():
    assert nastran_real("10.4 +6") == 10.4e6
    assert nastran_real(".2523 -3") == 0.2523e-3
    assert nastran_real("1.04+7") == 1.04e7
    assert nastran_real("2.1") == 2.1


def test_continuations_merge():
    deck = read_deck((ROOT / "inp/d01013a.inp").read_text(errors="replace"))
    spc1 = [c for c in deck.cards if c[0] == "SPC1" and c[2] == 12]
    assert len(spc1) == 1 and len(spc1[0]) == 3 + 24


def test_delta_wing_model():
    m = build_model((ROOT / "inp/d01013a.inp").read_text(errors="replace"))
    kinds = {}
    for e in m.elements:
        kinds[e.type] = kinds.get(e.type, 0) + 1
    assert kinds == {"CONROD": 36, "CQDMEM1": 14, "CROD": 24, "CSHEAR": 36, "CTRMEM": 3}
    assert len(m.grids) == 48 and m.grids[11].ps == "456"
    assert m.mats[1].e == 10.4e6 and abs(m.mats[1].nu - 0.3) < 1e-12
    assert m.forces[1][0][0] == 16 and m.forces[1][0][3][2] == 500.0
    assert m.case["SPC"] == "1" and m.case["LOAD"] == "1"
