"""MATT1 is ignored without a material temperature set (``PREMAT``: TEMPID = 0)."""

import numpy as np
import pytest

from nastran95py.bulk import load_temperatures, read_ecpts
from nastran95py.elements import (
    Qdmem2Batch,
    QdmemBatch,
    qdmem2_thermal_loads,
    qdmem_matrices,
    qdmm1d,
    qdmm2d,
    rodd,
    sheard,
    trimem,
    trmemd,
)

BULK = (
    "GRID    1               0.      0.      0.",
    "GRID    2               2.      0.      0.",
    "GRID    3               2.      1.      0.",
    "GRID    4               0.      1.      0.",
    "MAT1    7       1.+7            .3      .1      6.-6    70.",
    "PROD    1       7       .5      .1",
    "PSHEAR  2       7       .25",
    "PTRMEM  3       7       .25     .01",
    "PQDMEM  4       7       .25     .01",
    "PQDMEM1 5       7       .25     .01",
    "PQDMEM2 6       7       .25     .01",
    "CROD    1       1       1       2",
    "CSHEAR  2       2       1       2       3       4",
    "CTRMEM  3       3       1       2       3       30.",
    "CQDMEM  4       4       1       2       3       4       30.",
    "CQDMEM1 5       5       1       2       3       4       30.",
    "CQDMEM2 6       6       1       2       3       4       30.",
    "TEMPD   8       50.",
    "TEMP    9       1       150.    2       250.    3       350.",
    "TEMPD   9       100.",
)
MATT1 = (
    "MATT1   7       11      12",
    "TABLEM1 11                                                              +T1",
    "+T1     0.      2.+7    1000.   3.+7    ENDT",
    "TABLEM1 12                                                              +T2",
    "+T2     0.      8.+6    1000.   1.2+7   ENDT",
)


def _read(case: str, matt1: bool):
    cards = BULK + (MATT1 if matt1 else ())
    text = "ID T,T\nCEND\n" + case + "BEGIN BULK\n" + "\n".join(cards) + "\nENDDATA\n"
    model, ecpts = read_ecpts(text)
    return model, {rec.data.eid: rec for recs in ecpts.values() for rec in recs}


def _stiffness(eid: int, model, recs) -> np.ndarray:
    d, mats, cstm = recs[eid].data, model.materials, model.cstm
    if eid == 4:
        return qdmem_matrices(QdmemBatch.from_elements([d]), mats, cstm).k
    kernel = {1: rodd, 2: sheard, 3: trmemd, 5: qdmm1d, 6: qdmm2d}[eid]
    return kernel(d, mats, cstm).stiffness.values


NO_SET = ["", "SUBCASE 1\nTEMPERATURE(LOAD) = 9\n"]


@pytest.mark.parametrize("case", NO_SET)
def test_no_material_set_has_no_element_temperature(case):
    model, recs = _read(case, matt1=True)
    assert model.materials.mat1(7, 50.0).e == pytest.approx(2.05e7)
    assert {r.data.temperature for r in recs.values()} == {None}
    # TA1ETD writes 0.0 into the ECPT temperature word when ITEMP = 0.
    assert {r.words()[-1] for r in recs.values()} == {0.0}


@pytest.mark.parametrize("case", NO_SET)
@pytest.mark.parametrize("eid", [1, 2, 3, 4, 5, 6])
def test_matt1_ignored_without_material_set(case, eid):
    plain = _stiffness(eid, *_read(case, matt1=False))
    np.testing.assert_array_equal(_stiffness(eid, *_read(case, matt1=True)), plain)
    hot = _stiffness(eid, *_read("TEMPERATURE(MATERIAL) = 8\n", matt1=True))
    assert not np.allclose(hot, plain)


def test_load_only_set_applies_thermal_loads_with_plain_mat1():
    case = NO_SET[1]
    out = []
    for matt1 in (False, True):
        model, recs = _read(case, matt1)
        temps = load_temperatures(model, recs.values(), 9)
        assert temps[6] == pytest.approx((150 + 250 + 350 + 100) / 4)
        batch = Qdmem2Batch.from_elements([recs[6].data], model.materials)
        q2 = qdmem2_thermal_loads(batch, [temps[6]], model.cstm)[0].loads
        tri = trimem(recs[3].data, model.materials, temps[3], model.cstm).loads
        out.append((q2, tri))
    for plain, with_matt1 in zip(*out, strict=True):
        assert np.abs(plain).max() > 0.0
        np.testing.assert_array_equal(with_matt1, plain)
