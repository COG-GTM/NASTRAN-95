"""Element data contract mirroring the Fortran EST/ECPT records.

Every element routine in the Fortran receives one flat ECPT array laid out as:

1. a header of element connection and property words (ECT + EPT data),
2. one ``(CID, X, Y, Z)`` block per connected grid point from the BGPDT
   (coordinates in basic),
3. a trailer, normally the average element temperature.

``EcptLayout`` describes that layout per element type so the Python element
routines all take the same input (``ElementData``) and can be fed directly from,
or packed back into, the Fortran word array used by the parity harness.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from nastran95py.io.words import bits_to_real, int_to_bits, real_to_bits

FloatArray = NDArray[np.float64]

GRID_BLOCK_WORDS = 4


class FieldKind(Enum):
    INT = "I"
    REAL = "R"


@dataclass(frozen=True)
class Field:
    name: str
    kind: FieldKind
    count: int = 1


@dataclass
class GridPoint:
    """BGPDT entry for one connected grid: displacement CID and basic coordinates."""

    cid: int
    xyz: FloatArray

    def __post_init__(self) -> None:
        self.xyz = np.asarray(self.xyz, dtype=float).reshape(3)


@dataclass
class ElementData:
    """Decoded ECPT record shared by every element routine."""

    element: str
    fields: dict[str, int | float | tuple[int, ...] | FloatArray]
    grids: list[GridPoint]
    temperature: float | None = None
    """``MAT`` lookup temperature; ``None`` without a material temperature set."""
    extra: dict[str, int | float | tuple[int, ...] | FloatArray] = field(default_factory=dict)

    @property
    def eid(self) -> int:
        return int(self.fields["eid"])  # type: ignore[arg-type]

    def __getitem__(self, name: str) -> int | float | tuple[int, ...] | FloatArray:
        if name in self.fields:
            return self.fields[name]
        return self.extra[name]


INT, REAL = FieldKind.INT, FieldKind.REAL


@dataclass(frozen=True)
class EcptLayout:
    """Word layout of one element type's ECPT record."""

    element: str
    header: tuple[Field, ...]
    ngrids: int
    trailer: tuple[Field, ...] = (Field("eltemp", REAL),)
    source: str = ""

    @property
    def length(self) -> int:
        return (
            sum(f.count for f in self.header)
            + GRID_BLOCK_WORDS * self.ngrids
            + sum(f.count for f in self.trailer)
        )

    def parse(
        self, words: Sequence[int | float], raw: bool = False, material_temperature: bool = True
    ) -> ElementData:
        """Decode a flat ECPT array.

        With ``raw=True`` the words are 32-bit patterns (signed int32) as stored
        by the Fortran, and REAL fields are reinterpreted from their bits.

        ``material_temperature`` says whether a material temperature set is
        selected (``TEMPID`` nonzero). The ECPT word alone cannot tell: TA1ETD
        writes 0.0 without a set, and ``PREMAT`` then ignores MATT1 whatever the
        word holds. With ``False`` the decoded ``temperature`` is ``None``, so
        ``parse(pack(d), material_temperature=d.temperature is not None)``
        round-trips.
        """
        if len(words) != self.length:
            raise ValueError(f"{self.element} ECPT needs {self.length} words, got {len(words)}")
        pos = 0

        def take(f: Field) -> int | float | tuple[int, ...] | FloatArray:
            nonlocal pos
            chunk = words[pos : pos + f.count]
            pos += f.count
            if f.kind is FieldKind.INT:
                ints = tuple(int(w) for w in chunk)
                return ints[0] if f.count == 1 else ints
            reals = [bits_to_real(int(w)) if raw else float(w) for w in chunk]
            return reals[0] if f.count == 1 else np.array(reals)

        fields = {f.name: take(f) for f in self.header}
        grids = []
        for _ in range(self.ngrids):
            cid = int(words[pos])
            xyz = [bits_to_real(int(w)) if raw else float(w) for w in words[pos + 1 : pos + 4]]
            grids.append(GridPoint(cid, np.array(xyz)))
            pos += GRID_BLOCK_WORDS
        extra = {f.name: take(f) for f in self.trailer}
        eltemp = float(extra.pop("eltemp", 0.0))  # type: ignore[arg-type]
        temperature = eltemp if material_temperature else None
        return ElementData(self.element, fields, grids, temperature, extra)

    def pack(self, data: ElementData, raw: bool = False) -> list[int | float]:
        """Encode ``data`` into a flat ECPT array (inverse of :meth:`parse`)."""
        if data.element != self.element:
            raise ValueError(f"cannot pack {data.element} with the {self.element} layout")
        if len(data.grids) != self.ngrids:
            raise ValueError(f"{self.element} needs {self.ngrids} grids, got {len(data.grids)}")

        def real(x: float) -> int | float:
            return real_to_bits(float(x)) if raw else float(x)

        def put(f: Field, value: object) -> list[int | float]:
            values = [value] if f.count == 1 else list(value)  # type: ignore[call-overload]
            if len(values) != f.count:
                raise ValueError(f"{self.element}.{f.name} needs {f.count} values")
            if f.kind is FieldKind.INT:
                return [int_to_bits(int(v)) for v in values]
            return [real(v) for v in values]

        out: list[int | float] = []
        for f in self.header:
            out += put(f, data.fields[f.name])
        for g in data.grids:
            out += [int_to_bits(g.cid), *(real(c) for c in g.xyz)]
        # TA1ETD writes TI = 0.0 when there is no temperature set (ITEMP = 0).
        eltemp = 0.0 if data.temperature is None else data.temperature
        trailer_values = {"eltemp": eltemp, **data.extra}
        for f in self.trailer:
            out += put(f, trailer_values[f.name])
        return out


ROD = EcptLayout(
    "ROD",
    (
        Field("eid", INT),
        Field("sil", INT, 2),
        Field("mid", INT),
        Field("area", REAL),
        Field("j", REAL),
        Field("c", REAL),
        Field("nsm", REAL),
    ),
    ngrids=2,
    source="mis/krod.f",
)

TUBE = EcptLayout(
    "TUBE",
    (
        Field("eid", INT),
        Field("sil", INT, 2),
        Field("mid", INT),
        Field("od", REAL),
        Field("t", REAL),
        Field("nsm", REAL),
    ),
    ngrids=2,
    source="mis/rod.f",
)

BAR = EcptLayout(
    "BAR",
    (
        Field("eid", INT),
        Field("sil", INT, 2),
        Field("v", REAL, 3),
        Field("cssv", INT),
        Field("pin", INT, 2),
        Field("offset_a", REAL, 3),
        Field("offset_b", REAL, 3),
        Field("mid", INT),
        Field("area", REAL),
        Field("i1", REAL),
        Field("i2", REAL),
        Field("j", REAL),
        Field("nsm", REAL),
        Field("fe", REAL),
        Field("stress_points", REAL, 8),
        Field("k", REAL, 2),
        Field("i12", REAL),
    ),
    ngrids=2,
    source="mis/kbar.f",
)


def _membrane(element: str, ngrids: int, source: str, theta: bool = True) -> EcptLayout:
    """Shear panel / membrane layout: EID, SILs, [THETA], MID, T, NSM, grids, ELTEMP."""
    return EcptLayout(
        element,
        (
            Field("eid", INT),
            Field("sil", INT, ngrids),
            *((Field("theta", REAL),) if theta else ()),
            Field("mid", INT),
            Field("t", REAL),
            Field("nsm", REAL),
        ),
        ngrids=ngrids,
        source=source,
    )


SHEAR = _membrane("SHEAR", 4, "mis/kpanel.f", theta=False)
TRMEM = _membrane("TRMEM", 3, "mis/ktrmem.f")
QDMEM = _membrane("QDMEM", 4, "mis/kqdmem.f")
QDMEM1 = _membrane("QDMEM1", 4, "mis/qdmm1d.f")
QDMEM2 = _membrane("QDMEM2", 4, "mis/qdmm2d.f")

LAYOUTS: dict[str, EcptLayout] = {
    layout.element: layout for layout in (ROD, TUBE, BAR, SHEAR, TRMEM, QDMEM, QDMEM1, QDMEM2)
}

CARD_LAYOUTS: dict[str, str] = {
    "CROD": "ROD",
    "CONROD": "ROD",
    "CBAR": "BAR",
    "CSHEAR": "SHEAR",
    "CTRMEM": "TRMEM",
    "CQDMEM": "QDMEM",
    "CQDMEM1": "QDMEM1",
    "CQDMEM2": "QDMEM2",
}
"""Connection card -> ECPT layout (element type name as in the Fortran ``ELEM`` table)."""
