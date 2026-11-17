"""Build ECPT records (the TA1 output) from a :class:`BulkModel`.

Each connection card becomes one :class:`EcptRecord` holding the decoded
``ElementData`` for its layout: header words from the connection and property
cards (grid ids replaced by scalar index numbers, as in the Fortran), one BGPDT
block per grid (displacement CID and basic coordinates, in connection order) and
the element temperature.

The element temperature follows ``TA1B``/``TA1ETD``: with no material
temperature set it is 0.0, otherwise the average of the connected grid
temperatures of that set (TEMP, falling back to the set's TEMPD default).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from nastran95py.bulk.model import PROPERTY_FOR, BulkDataError, BulkModel, Connection
from nastran95py.elements.contract import CARD_LAYOUTS, LAYOUTS, ElementData, GridPoint

_TEMPERATURE = re.compile(r"^\s*TEMP[A-Z]*\s*(?:\(([^)]*)\))?\s*=\s*(\d+)\s*$")


@dataclass(frozen=True)
class EcptRecord:
    connection: Connection
    data: ElementData

    @property
    def card(self) -> str:
        return self.connection.card

    @property
    def element(self) -> str:
        return self.data.element

    @property
    def grid_ids(self) -> tuple[int, ...]:
        return self.connection.grids

    def words(self, raw: bool = False) -> list[int | float]:
        """Flat ECPT words (``raw=True``: 32-bit patterns as the Fortran stores them)."""
        return LAYOUTS[self.element].pack(self.data, raw=raw)


def material_temperature_set(case_control: list[str]) -> int | None:
    """TEMPERATURE(MATERIAL|BOTH) set id from case control lines, or ``None``.

    TEMPERATURE(LOAD) selections are ignored. The material temperature set is
    problem-wide in the Fortran (``KSYSTM(10)``), so different sets are rejected.
    """
    sets = set()
    for raw in case_control:
        m = _TEMPERATURE.match(raw.split("$", 1)[0].upper())
        if m and (m.group(1) or "BOTH").strip().startswith(("MAT", "BOTH")):
            sets.add(int(m.group(2)))
    if len(sets) > 1:
        raise BulkDataError(f"case control selects material temperature sets {sorted(sets)}")
    return sets.pop() if sets else None


def element_temperature(model: BulkModel, grids: tuple[int, ...], temp_set: int | None) -> float:
    if not temp_set:
        return 0.0
    return sum(model.grid_temperature(temp_set, g) for g in grids) / len(grids)


def build_ecpt(model: BulkModel, conn: Connection, temp_set: int | None = None) -> EcptRecord:
    """ECPT record for one connection card."""
    name = CARD_LAYOUTS.get(conn.card)
    if name is None:
        raise BulkDataError(f"no ECPT layout for {conn.card}")
    layout = LAYOUTS[name]
    sil = model.sil
    missing = [g for g in conn.grids if g not in sil]
    if missing:
        raise BulkDataError(f"{conn.card} {conn.eid}: undefined grid points {missing}")
    fields: dict = {"eid": conn.eid, "sil": tuple(sil[g] for g in conn.grids)}
    if conn.card == "CONROD":
        assert conn.conrod is not None
        mid, a, j, c, nsm = conn.conrod
        fields.update(mid=mid, area=a, j=j, c=c, nsm=nsm)
    elif conn.card == "CROD":
        prop = model.rod_properties.get(conn.pid)  # type: ignore[arg-type]
        if prop is None:
            raise BulkDataError(f"CROD {conn.eid}: PROD {conn.pid} is not defined")
        fields.update(mid=prop.mid, area=prop.a, j=prop.j, c=prop.c, nsm=prop.nsm)
    else:
        pcard = PROPERTY_FOR[conn.card]
        plate = model.plate_properties.get((pcard, conn.pid))  # type: ignore[arg-type]
        if plate is None:
            raise BulkDataError(f"{conn.card} {conn.eid}: {pcard} {conn.pid} is not defined")
        fields.update(mid=plate.mid, t=plate.t, nsm=plate.nsm)
        if any(f.name == "theta" for f in layout.header):
            fields["theta"] = conn.theta
    grids = [GridPoint(*model.bgpdt(g)) for g in conn.grids]
    temperature = element_temperature(model, conn.grids, temp_set)
    data = ElementData(name, fields, grids, temperature)
    return EcptRecord(conn, data)


def build_ecpts(model: BulkModel, temp_set: int | None = None) -> dict[str, list[EcptRecord]]:
    """ECPT records for every supported connection, grouped by layout, sorted by EID."""
    out: dict[str, list[EcptRecord]] = {}
    seen: set[int] = set()
    for conn in model.connections:
        if conn.eid in seen:
            raise BulkDataError(f"duplicate element id {conn.eid}")
        seen.add(conn.eid)
        rec = build_ecpt(model, conn, temp_set)
        out.setdefault(rec.element, []).append(rec)
    return {k: sorted(v, key=lambda r: r.data.eid) for k, v in sorted(out.items())}


def read_ecpts(text: str) -> tuple[BulkModel, dict[str, list[EcptRecord]]]:
    """Read a whole deck: bulk model plus ECPT records at its material temperature set."""
    from nastran95py.reference.cards import read_deck

    deck = read_deck(text)
    model = BulkModel.from_deck(deck)
    temp_set = material_temperature_set(deck.case_control)
    return model, build_ecpts(model, temp_set)
