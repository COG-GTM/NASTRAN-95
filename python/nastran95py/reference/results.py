"""Parse result tables out of a NASTRAN-95 printed listing (``demoout/*.out``).

Uses the section index from :mod:`nastran95py.reference.listing` and turns each
displacement, SPC force, load vector, element force and element stress section
into a :class:`Table` with one row per grid point or element. SORT2 sections
(one page per point/element, rows by subcase) are regrouped into one table per
subcase so both sorts have the same shape.

Values are kept exactly as printed (``float(token)``); blank fields (e.g. an
unprinted safety margin) are ``None``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from nastran95py.reference.listing import index_listing, spaced_title

PARSED_KINDS = ("displacement", "spcforce", "oload", "force", "stress")

_GRID_COLUMNS = ("t1", "t2", "t3", "r1", "r2", "r3")
_MEMBRANE = ("normal_x", "normal_y", "shear_xy", "angle", "major", "minor", "max_shear")
_SHEAR_FORCE = (
    "f1_from4",
    "f1_from2",
    "f2_from1",
    "f2_from3",
    "f3_from2",
    "f3_from4",
    "f4_from3",
    "f4_from1",
    "kick1",
    "shear12",
    "kick2",
    "shear23",
    "kick3",
    "shear34",
    "kick4",
    "shear41",
)

#: Value columns (after the id) of each (kind, element type) table, in print order.
COLUMNS: dict[tuple[str, str | None], tuple[str, ...]] = {
    ("displacement", None): _GRID_COLUMNS,
    ("spcforce", None): _GRID_COLUMNS,
    ("oload", None): _GRID_COLUMNS,
    ("stress", "CTRMEM"): _MEMBRANE,
    ("stress", "CQDMEM"): _MEMBRANE,
    ("stress", "CQDMEM1"): _MEMBRANE,
    ("stress", "CQDMEM2"): _MEMBRANE,
    ("stress", "CSHEAR"): ("max_shear", "avg_shear", "margin"),
    ("stress", "CROD"): ("axial", "axial_margin", "torsional", "torsional_margin"),
    ("stress", "CONROD"): ("axial", "axial_margin", "torsional", "torsional_margin"),
    ("force", "CROD"): ("axial", "torque"),
    ("force", "CONROD"): ("axial", "torque"),
    ("force", "CSHEAR"): _SHEAR_FORCE,
}

_INT = re.compile(r"\d+")
_SORT2_KEY = re.compile(r"^\s*(POINT|ELEMENT)-ID\s*=\s*(\d+)")
_PAGE = re.compile(r"^1.*/\s*PAGE\s+\d+\s*$")
_TOKEN = re.compile(r"\S+")
_LABEL = re.compile(r"\S+(?: \S+)*")


class ListingParseError(ValueError):
    pass


@dataclass
class Table:
    kind: str
    element_type: str | None
    subcase: int | None
    occurrence: int
    sort: str
    title: str
    columns: tuple[str, ...]
    scan: dict | None = None
    lines: list[int] = field(default_factory=list)
    ids: list[int] = field(default_factory=list)
    rows: list[list[float | None]] = field(default_factory=list)
    point_types: list[str] | None = None

    @property
    def key(self) -> str:
        """Stable identifier, e.g. ``stress/CQDMEM/sc1/occ1`` or ``.../scan-MAJOR-2``."""
        parts = [
            self.kind,
            self.element_type or "grid",
            f"sc{self.subcase}",
            f"occ{self.occurrence}",
        ]
        if self.scan:
            parts.append(f"scan-{self.scan['field']}-{self.scan['set']}")
        return "/".join(parts)


def _is_row(line: str) -> bool:
    toks = line[1:].split()
    return bool(toks) and _INT.fullmatch(toks[0]) is not None


def _labels(header: str) -> list[tuple[str, float]]:
    """Header labels (split on runs of 2+ blanks) with their centre column."""
    return [(m[0], (m.start() + m.end() - 1) / 2) for m in _LABEL.finditer(header)]


def _assign(line: str, labels: list[tuple[str, float]]) -> list[str | None]:
    """Place the tokens of a data row under the header labels.

    A full row is taken in order; a row with blank fields (unprinted margins,
    an empty right half of a two-up table) is matched by nearest label centre.
    """
    toks = [(m[0], (m.start() + m.end() - 1) / 2) for m in _TOKEN.finditer(line, 1)]
    if len(toks) == len(labels):
        return [t for t, _ in toks]
    if len(toks) > len(labels):
        raise ListingParseError(f"row has more fields than header: {line!r}")
    out: list[str | None] = [None] * len(labels)
    last = -1
    for tok, centre in toks:
        k = min(range(len(labels)), key=lambda j: abs(labels[j][1] - centre))
        if k <= last:
            raise ListingParseError(f"ambiguous field placement in {line!r}")
        out[k] = tok
        last = k
    return out


def _num(tok: str | None) -> float | None:
    return None if tok is None else float(tok.replace("D", "E"))


def _sort2_id(lines: list[str], start: int) -> int | None:
    """``POINT-ID``/``ELEMENT-ID`` printed above the title on a SORT2 page."""
    for i in range(start - 1, max(start - 8, -1), -1):
        if m := _SORT2_KEY.match(lines[i]):
            return int(m[2])
        if _PAGE.match(lines[i]):
            break
    return None


def _section_records(lines: list[str], sec: dict, ncols: int):
    """Yield ``(row_id, header_id, point_type, values)`` per record in a section.

    ``row_id`` is the first integer of the record (grid/element id for SORT1,
    subcase for SORT2) and ``header_id`` the SORT2 point/element id.
    """
    kind, etype = sec["kind"], sec["element_type"]
    first, last = sec["lines"][0] - 1, sec["lines"][1]
    sort2 = sec["sort"] == "SORT2"
    header_id = _sort2_id(lines, first) if sort2 else None
    grid = etype is None
    width = ncols + (2 if grid else 1)
    labels: list[tuple[str, float]] | None = None
    prev = ""
    i = first
    while i < last:
        line = lines[i]
        if sort2 and (m := _SORT2_KEY.match(line)):
            header_id = int(m[2])
        if not _is_row(line):
            if line.strip() and spaced_title(line) is None:
                prev = line
            labels = None
            i += 1
            continue
        if (kind, etype) == ("force", "CSHEAR"):
            toks = line[1:].split() + (lines[i + 1].split() if i + 1 < last else [])
            if len(toks) != 1 + ncols:
                raise ListingParseError(f"CSHEAR force record at line {i + 1}: {toks}")
            yield int(toks[0]), header_id, None, [_num(t) for t in toks[1:]]
            i += 2
            continue
        if labels is None:
            labels = _labels(prev)
            if len(labels) not in (width, 2 * width):
                raise ListingParseError(
                    f"{sec['title']}: header at line {i} has {len(labels)} labels, "
                    f"expected {width} or {2 * width}: {prev!r}"
                )
        fields = _assign(line, labels)
        for half in range(len(labels) // width):
            chunk = fields[half * width : (half + 1) * width]
            if chunk[0] is None:
                if any(f is not None for f in chunk):
                    raise ListingParseError(f"values without an id at line {i + 1}")
                continue
            ptype = chunk[1] if grid else None
            vals = chunk[2:] if grid else chunk[1:]
            yield int(chunk[0]), header_id, ptype, [_num(v) for v in vals]
        i += 1


def parse_tables(text: str, default_subcase: int | None = None, index: dict | None = None):
    """Parse every result section of a listing into :class:`Table` objects.

    ``default_subcase`` labels SORT1 sections whose page carries no
    ``SUBCASE n`` (single-subcase decks). ``index`` reuses an
    :func:`~nastran95py.reference.listing.index_listing` result.
    """
    lines = text.splitlines()
    index = index or index_listing(text)
    tables: list[Table] = []
    for sec in index["sections"]:
        if sec["kind"] not in PARSED_KINDS:
            continue
        cols_key = (sec["kind"], sec["element_type"])
        if cols_key not in COLUMNS:
            raise ListingParseError(f"no column layout for {cols_key} ({sec['title']})")
        cols = COLUMNS[cols_key]
        by_subcase: dict[int | None, Table] = {}

        def table_for(subcase, sec=sec, cols=cols, by_subcase=by_subcase):
            if subcase not in by_subcase:
                by_subcase[subcase] = Table(
                    kind=sec["kind"],
                    element_type=sec["element_type"],
                    subcase=subcase,
                    occurrence=sec["occurrence"],
                    sort=sec["sort"],
                    title=sec["title"],
                    columns=cols,
                    scan=sec["scan"],
                    lines=list(sec["lines"]),
                    point_types=[] if sec["element_type"] is None else None,
                )
            return by_subcase[subcase]

        for row_id, header_id, ptype, vals in _section_records(lines, sec, len(cols)):
            if sec["sort"] == "SORT2":
                if header_id is None:
                    raise ListingParseError(f"SORT2 row without POINT/ELEMENT-ID in {sec}")
                tab, ident = table_for(row_id), header_id
            else:
                sc = sec["subcase"] if sec["subcase"] is not None else default_subcase
                tab, ident = table_for(sc), row_id
            tab.ids.append(ident)
            tab.rows.append(vals)
            if tab.point_types is not None:
                tab.point_types.append(ptype)
        tables.extend(by_subcase[k] for k in sorted(by_subcase, key=lambda s: (s is None, s)))
    return tables
