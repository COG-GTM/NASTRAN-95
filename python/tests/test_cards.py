import pytest

from nastran95py.reference.cards import card_signature, expand_ids, read_deck, to_value


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", None),
        ("  12  ", 12),
        ("-3", -3),
        ("1.04+7", 1.04e7),
        ("12.700-6", 12.7e-6),
        ("10.4  +6", 10.4e6),
        (".333", 0.333),
        ("1.0D-3", 1.0e-3),
        ("2.5E+2", 250.0),
        ("THRU", "THRU"),
        ("123456", 123456),
    ],
)
def test_to_value(raw, expected):
    assert to_value(raw) == expected


def test_expand_ids_thru_and_except():
    assert expand_ids([1, "THRU", 4, 9]) == [1, 2, 3, 4, 9]
    assert expand_ids([1, "THRU", 6, "EXCEPT", 3, 5]) == [1, 2, 4, 6]


DECK = """ID    TEST,DECK
APP   DISPLACEMENT
SOL   1,1
CEND
TITLE = SMALL
LOAD = 1
DISP = ALL
BEGIN BULK
GRID    1               0.0     0.0     0.0
CQDMEM2 1       10      1       2       3       4       30.0
PQDMEM2 10      20      .1      11      20      .2
MAT1    20      1.04+7          .3      .1      12.6-6  70.0            +M1
+M1     12.5+4  9.0+4   7.0+4
TABLEM1 30                                                              +T1
+T1     0.0     1.0     100.0   2.0     ENDT
FORCE,  1,      1,      , 5.0,1.0,  0.0,0.0
ENDDATA
"""


def test_read_deck_sections_and_continuations():
    deck = read_deck(DECK)
    assert deck.executive[-1] == "CEND"
    assert deck.case_control == ["TITLE = SMALL", "LOAD = 1", "DISP = ALL"]
    names = [c.name for c in deck.bulk]
    assert names == ["GRID", "CQDMEM2", "PQDMEM2", "MAT1", "TABLEM1", "FORCE"]
    assert sum(ln.startswith("+M1") for ln in deck.bulk_physical) == 1
    mat1 = deck.bulk[3]
    assert mat1.get(2) == 1.04e7
    assert mat1.get(3) is None
    assert mat1.get(9) == 1.25e5
    assert mat1.get(11) == 7.0e4
    assert deck.bulk[1].get(7) == 30.0
    assert deck.bulk[4].fields[8:] == [0.0, 1.0, 100.0, 2.0, "ENDT"]
    force = deck.bulk[5]
    assert (force.get(1), force.get(2), force.get(3), force.get(4)) == (1, 1, None, 5.0)


def test_card_signature_free_and_fixed_field_agree():
    fixed = "FORCE   1       1               5.0     1.0     0.0     0.0"
    assert card_signature("FORCE,  1,      1,      , 5.0,1.0,  0.0,0.0") == card_signature(fixed)
    assert card_signature(fixed) != card_signature(fixed.replace("5.0", "6.0"))


def test_orphan_continuation_rejected():
    with pytest.raises(ValueError, match="orphan"):
        read_deck("CEND\nBEGIN BULK\n+X      1\nGRID    1\nENDDATA\n")
