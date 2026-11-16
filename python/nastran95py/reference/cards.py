"""Minimal reader for NASTRAN-95 input decks, scoped to the RF1 deck inventory.

Splits a deck into executive, case control and bulk data and parses bulk cards
into named fields (fixed, large and free field, continuations). It does not
resolve properties or build EST/ECPT records: the full bulk-data reader is
UNT186-7, which will extend or replace this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_INT = re.compile(r"[+-]?\d+")
_NASTRAN_EXP = re.compile(r"([+-]?(?:\d+\.?\d*|\.\d+))([+-]\d+)")


def to_value(raw: str) -> int | float | str | None:
    """Convert one bulk data field to int, float, str or None (blank).

    Handles NASTRAN reals without an ``E`` (``1.04+7``, ``12.700-6``),
    ``D`` exponents and blanks embedded in a field (``10.4  +6``).
    """
    s = raw.strip()
    if not s:
        return None
    t = s.replace(" ", "")
    if _INT.fullmatch(t):
        return int(t)
    u = t.upper().replace("D", "E")
    try:
        return float(u)
    except ValueError:
        pass
    m = _NASTRAN_EXP.fullmatch(u)
    if m:
        return float(f"{m[1]}e{m[2]}")
    return s


@dataclass
class Card:
    name: str
    fields: list = field(default_factory=list)
    lineno: int = 0
    physical: list[str] = field(default_factory=list)

    def get(self, i: int, default=None):
        """Field ``i`` (1-based after the name, continuation fields follow)."""
        if 1 <= i <= len(self.fields) and self.fields[i - 1] is not None:
            return self.fields[i - 1]
        return default


@dataclass
class Deck:
    executive: list[str]
    case_control: list[str]
    bulk: list[Card]
    bulk_physical: list[str]


def _split_fixed(line: str) -> tuple[str, list[str], str]:
    line = line.rstrip("\n").ljust(80)
    name = line[0:8]
    head = name.strip()
    if head.endswith("*") or head.startswith("*"):
        fields = [line[8 + 16 * i : 24 + 16 * i] for i in range(4)]
    else:
        fields = [line[8 + 8 * i : 16 + 8 * i] for i in range(8)]
    return name, fields, line[72:80]


def _split_free(line: str) -> tuple[str, list[str], str]:
    parts = [p for p in line.rstrip("\n").split(",")]
    parts += [""] * (10 - len(parts))
    return parts[0], parts[1:9], parts[9] if len(parts) > 9 else ""


def card_signature(line: str) -> tuple:
    """Normalized content of one physical bulk data card (fixed, large or free field)."""
    name, raw_fields, cont = _split_free(line) if "," in line[:72] else _split_fixed(line)
    values = [to_value(f) for f in raw_fields]
    while values and values[-1] is None:
        values.pop()
    return (name.strip().upper(), tuple(values), cont.strip())


def parse_bulk(lines: list[tuple[int, str]]) -> tuple[list[Card], list[str]]:
    """Assemble bulk data lines (with continuations) into logical cards."""
    cards: list[Card] = []
    physical_names: list[str] = []
    open_conts: dict[str, Card] = {}
    for lineno, line in lines:
        if not line.strip() or line.lstrip().startswith("$"):
            continue
        name, raw_fields, cont = _split_free(line) if "," in line[:72] else _split_fixed(line)
        mnemonic = name.strip()
        physical_names.append(line.rstrip("\n"))
        values = [to_value(f) for f in raw_fields]
        is_cont = mnemonic == "" or mnemonic[0] in "+*"
        if is_cont:
            key = mnemonic[1:].strip()
            parent = open_conts.pop(key, None) if key else None
            if parent is None and cards and not key:
                parent = cards[-1]
            if parent is None:
                raise ValueError(f"line {lineno}: orphan continuation {mnemonic!r}")
            parent.fields.extend(values)
            parent.physical.append(line.rstrip("\n"))
            card = parent
        else:
            card = Card(
                name=mnemonic.rstrip("*").upper(),
                fields=values,
                lineno=lineno,
                physical=[line.rstrip("\n")],
            )
            cards.append(card)
        cont_key = cont.strip()
        if cont_key:
            open_conts[cont_key[1:].strip()] = card
    for c in cards:
        while c.fields and c.fields[-1] is None:
            c.fields.pop()
    return cards, physical_names


def read_deck(text: str) -> Deck:
    """Split a deck into executive control, case control and bulk data."""
    executive: list[str] = []
    case: list[str] = []
    bulk: list[tuple[int, str]] = []
    section = "exec"
    for lineno, line in enumerate(text.splitlines(), start=1):
        head = line.strip().upper()
        if section == "exec":
            executive.append(line.rstrip())
            if head.startswith("CEND"):
                section = "case"
        elif section == "case":
            if head.startswith("BEGIN BULK"):
                section = "bulk"
            else:
                case.append(line.rstrip())
        elif section == "bulk":
            if head.startswith("ENDDATA"):
                section = "done"
            else:
                bulk.append((lineno, line))
    cards, names = parse_bulk(bulk)
    return Deck(executive=executive, case_control=case, bulk=cards, bulk_physical=names)


def expand_ids(tokens: list) -> list[int]:
    """Expand ``a THRU b`` (and ``EXCEPT``) sequences into a list of ids."""
    out: list[int] = []
    excluded: set[int] = set()
    vals = [t for t in tokens if t is not None]
    i = 0
    mode_except = False
    while i < len(vals):
        v = vals[i]
        if isinstance(v, str) and v.upper() == "EXCEPT":
            mode_except = True
            i += 1
            continue
        if i + 2 < len(vals) and isinstance(vals[i + 1], str) and vals[i + 1].upper() == "THRU":
            rng = range(int(v), int(vals[i + 2]) + 1)
            (excluded.update if mode_except else out.extend)(rng)
            i += 3
            continue
        if isinstance(v, int):
            (excluded.add if mode_except else out.append)(v)
        i += 1
    return [x for x in out if x not in excluded]
