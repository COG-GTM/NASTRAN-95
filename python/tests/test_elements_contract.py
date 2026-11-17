import numpy as np
import pytest

from nastran95py.elements import LAYOUTS, ElementData, GridPoint
from nastran95py.io import real_to_bits

ROD_WORDS = [11, 1, 7, 3, 0.5, 0.25, 0.1, 0.0, 0, 0.0, 0.0, 0.0, 2, 10.0, 0.0, 0.0, 68.0]


def test_layout_lengths_match_fortran_ecpt():
    assert LAYOUTS["ROD"].length == 17
    assert LAYOUTS["BAR"].length == 42


def test_rod_parse():
    data = LAYOUTS["ROD"].parse(ROD_WORDS)
    assert data.eid == 11
    assert data["sil"] == (1, 7)
    assert data["mid"] == 3
    assert data["area"] == 0.5
    assert [g.cid for g in data.grids] == [0, 2]
    np.testing.assert_array_equal(data.grids[1].xyz, [10, 0, 0])
    assert data.temperature == 68.0


def test_rod_pack_round_trip_typed_and_raw():
    layout = LAYOUTS["ROD"]
    data = layout.parse(ROD_WORDS)
    assert layout.pack(data) == [float(w) if isinstance(w, float) else w for w in ROD_WORDS]
    raw = layout.pack(data, raw=True)
    assert raw[4] == real_to_bits(0.5)
    assert raw[0] == 11
    again = layout.parse(raw, raw=True)
    assert again.fields["area"] == 0.5
    assert again.temperature == 68.0


def test_bar_vector_fields():
    layout = LAYOUTS["BAR"]
    words = list(range(1, 43))
    data = layout.parse(words)
    np.testing.assert_array_equal(data["v"], [4, 5, 6])
    assert data["pin"] == (8, 9)
    assert data["mid"] == 16
    assert len(data["stress_points"]) == 8
    assert data.grids[0].cid == 34
    assert data.temperature == 42.0
    packed = layout.pack(data)
    assert packed[0] == 1 and packed[15] == 16
    assert packed == [w if isinstance(w, int) else float(w) for w in packed]
    assert [float(w) for w in packed] == [float(w) for w in words]


def test_validation():
    layout = LAYOUTS["ROD"]
    with pytest.raises(ValueError):
        layout.parse(ROD_WORDS[:-1])
    data = layout.parse(ROD_WORDS)
    with pytest.raises(ValueError):
        LAYOUTS["BAR"].pack(data)
    short = ElementData("ROD", data.fields, [GridPoint(0, [0, 0, 0])], 0.0)
    with pytest.raises(ValueError):
        layout.pack(short)
