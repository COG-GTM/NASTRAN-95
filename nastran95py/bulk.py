"""Reader for NASTRAN input decks: executive control, case control and small-field bulk data."""
import re
from dataclasses import dataclass, field


def _clean(text):
    return [ln.rstrip("\r\n\x1a").expandtabs(8) for ln in text.splitlines()]


def nastran_real(s):
    """Parse a NASTRAN real field: '10.4 +6', '.2523 -3', '1.04+7', '2.1', '7.E-3'."""
    t = s.replace(" ", "").upper().replace("D", "E")
    if not t:
        return 0.0
    t = re.sub(r"(?<=[0-9.])([+-]\d+)$", r"E\1", t)
    return float(t)


def value(s):
    """Field -> int, float, str or None (blank)."""
    t = s.strip()
    if not t:
        return None
    if re.fullmatch(r"[+-]?\d+", t):
        return int(t)
    if re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)\s*([EeDd]?\s*[+-]?\s*\d+)?", t):
        return nastran_real(t)
    return t


@dataclass
class Deck:
    executive: list = field(default_factory=list)
    case: list = field(default_factory=list)
    cards: list = field(default_factory=list)  # [name, f1, f2, ...] with continuations merged


def _fields(line):
    line = line.ljust(80)
    if "," in line.split("$")[0] and not line.startswith(" "):
        parts = line.split("$")[0].rstrip().split(",")
        return parts[0].strip(), [value(p) for p in parts[1:]], None
    name = line[0:8].strip()
    data = [value(line[8 + 8 * i: 16 + 8 * i]) for i in range(8)]
    cont = line[72:80].strip() or None
    return name, data, cont


def read_deck(text):
    lines = _clean(text)
    deck = Deck()
    i = 0
    while i < len(lines) and lines[i].strip().upper() != "CEND":
        deck.executive.append(lines[i])
        i += 1
    i += 1
    while i < len(lines) and not lines[i].strip().upper().startswith("BEGIN BULK"):
        if lines[i].strip() and not lines[i].lstrip().startswith("$"):
            deck.case.append(lines[i].strip())
        i += 1
    i += 1
    pending = {}
    order = []
    for line in lines[i:]:
        if not line.strip() or line.startswith("$"):
            continue
        if line.strip().upper().startswith("ENDDATA"):
            break
        name, data, cont = _fields(line)
        if name.startswith("+") or name.startswith("*"):
            card = pending.pop(name, None)
            if card is None:
                raise ValueError(f"orphan continuation {name!r}")
        else:
            card = [name.upper()]
            order.append(card)
        card.extend(data)
        if cont:
            pending[cont] = card
    for card in order:
        while len(card) > 1 and card[-1] is None:
            card.pop()
    deck.cards = order
    return deck


def case_control(case_lines):
    """Subcase-1 case control as {KEY: value-string}, e.g. {'SPC': '1', 'LOAD': '1'}: the
    defaults above the first SUBCASE plus that subcase's own entries."""
    out = {}
    subcases = 0
    for ln in case_lines:
        if ln.strip().upper().startswith("SUBCASE"):
            subcases += 1
            if subcases > 1:
                break
            continue
        if "=" in ln:
            k, v = ln.split("=", 1)
            out[k.strip().upper()] = v.strip()
    return out
