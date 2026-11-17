"""Static load-set data ``EDTL`` reads for element thermal and deformation loads."""

from __future__ import annotations

from collections.abc import Iterable

from nastran95py.bulk.ecpt import EcptRecord, element_temperature
from nastran95py.bulk.model import BulkDataError, BulkModel
from nastran95py.reference.cards import Card


def deformations(cards: Iterable[Card], set_id: int) -> dict[int, float]:
    """``DEFORM`` set ``set_id``: element id -> enforced axial deformation ``D``."""
    out: dict[int, float] = {}
    for c in cards:
        if c.name != "DEFORM" or c.get(1) != set_id:
            continue
        for k in (2, 4, 6):
            eid = c.get(k)
            if eid is None:
                continue
            if eid in out:
                raise BulkDataError(f"DEFORM set {set_id}: element {eid} given twice")
            out[int(eid)] = float(c.get(k + 1, 0.0))
    return out


def load_temperatures(
    model: BulkModel, records: Iterable[EcptRecord], set_id: int
) -> dict[int, float]:
    """Element temperature ``TI(1)`` of each record in load temperature set ``set_id``.

    The mean of the connected grids' TEMP/TEMPD values, as the ECPT element
    temperature word is built.
    """
    return {r.data.eid: element_temperature(model, r.connection.grids, set_id) for r in records}


def load_grid_temperatures(
    model: BulkModel, records: Iterable[EcptRecord], set_id: int
) -> dict[int, tuple[float, ...]]:
    """Connected-grid TEMP/TEMPD values of each record in load temperature set ``set_id``.

    What ``SSGETD (ELID, TI, GRIDS)`` returns in ``TI(2..GRIDS+1)`` for elements such
    as CSHEAR (``TSHEAR``) that load from the individual grid temperatures.
    """
    return {
        r.data.eid: tuple(model.grid_temperature(set_id, g) for g in r.connection.grids)
        for r in records
    }
