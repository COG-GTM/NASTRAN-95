"""Executive and case control deck summaries for RF1 decks."""

from __future__ import annotations

import re

from nastran95py.reference.cards import expand_ids, to_value

# NASTRAN matches case control keywords on their leading characters.
_OUTPUT_KEYWORDS = [
    ("DISP", "DISPLACEMENT"),
    ("SPCF", "SPCFORCE"),
    ("ELST", "STRESS"),
    ("STRE", "STRESS"),
    ("ELFO", "FORCE"),
    ("FORC", "FORCE"),
    ("OLOA", "OLOAD"),
    ("GPFO", "GPFORCE"),
    ("ESE", "ESE"),
]
_SELECTORS = [("LOAD", "LOAD"), ("SPC", "SPC"), ("MPC", "MPC"), ("TEMP", "TEMPERATURE")]
_TEXT = [("TITL", "TITLE"), ("SUBT", "SUBTITLE"), ("LABE", "LABEL")]
_ASSIGN = re.compile(r"^\s*([A-Z0-9]+)\s*(\(([^)]*)\))?\s*=\s*(.*)$")


def parse_executive(lines: list[str]) -> dict:
    out: dict = {
        "id": None,
        "app": None,
        "sol": None,
        "time": None,
        "alter": False,
        "diag": [],
        "nastran": [],
    }
    for raw in lines:
        line = raw.split("$", 1)[0].rstrip()
        parts = line.split(None, 1)
        if not parts:
            continue
        key, rest = parts[0].upper(), (parts[1].strip() if len(parts) > 1 else "")
        if key == "ID":
            out["id"] = rest
        elif key == "APP":
            out["app"] = rest.upper()
        elif key == "SOL":
            out["sol"] = rest.replace(" ", "")
        elif key == "TIME":
            out["time"] = int(rest) if rest.isdigit() else rest
        elif key == "ALTER":
            out["alter"] = True
        elif key == "DIAG":
            out["diag"].append(rest)
        elif key == "NASTRAN":
            out["nastran"].append(rest)
    sol = (out["sol"] or "").split(",")
    out["rigid_format"] = int(sol[0]) if sol and sol[0].isdigit() else None
    return out


def _logical_lines(lines: list[str]) -> list[str]:
    merged: list[str] = []
    buf = ""
    for raw in lines:
        line = raw.split("$", 1)[0].rstrip() if not raw.lstrip().startswith("$") else ""
        if not line.strip():
            continue
        buf = f"{buf} {line.strip()}" if buf else line
        if buf.rstrip().endswith(","):
            continue
        merged.append(buf)
        buf = ""
    if buf:
        merged.append(buf)
    return merged


def _set_tokens(text: str) -> list:
    tokens = [t for t in re.split(r"[,\s]+", text.strip()) if t]
    return [to_value(t) if not t.isalpha() else t.upper() for t in tokens]


def _match(word: str, table: list[tuple[str, str]]) -> str | None:
    for prefix, canonical in table:
        if word.startswith(prefix):
            return canonical
    return None


def parse_case_control(lines: list[str]) -> dict:
    """Return sets, output requests and per-subcase selections.

    Entries above the first ``SUBCASE`` are defaults inherited by every
    subcase; the plot/XY packets (``OUTPUT(PLOT)``, ``OUTPUT(XYOUT)``) are skipped.
    """
    sets: dict[int, list[int] | str] = {}
    default: dict = {"selectors": {}, "outputs": {}, "text": {}}
    subcases: list[dict] = []
    current = default
    for line in _logical_lines(lines):
        head = line.strip().upper()
        if head.startswith("OUTPUT"):
            if "PLOT" in head or "XY" in head:
                break
            continue
        m_sub = re.match(r"^SUBCASE\s+(\d+)", head)
        if m_sub:
            current = {"id": int(m_sub[1]), "selectors": {}, "outputs": {}, "text": {}}
            subcases.append(current)
            continue
        m_set = re.match(r"^SET\s+(\d+)\s*=\s*(.*)$", head)
        if m_set:
            body = m_set[2].strip()
            sets[int(m_set[1])] = "ALL" if body == "ALL" else expand_ids(_set_tokens(body))
            continue
        m = _ASSIGN.match(head)
        if not m:
            continue
        word, opts, value = m[1], m[3], m[4].strip()
        if word.startswith("SCAN"):
            options = [o.strip() for o in (opts or "").split(",") if o.strip()]
            current.setdefault("scan", []).append({"options": options, "value": value})
            continue
        text_key = _match(word, _TEXT)
        if text_key:
            current["text"][text_key] = line.split("=", 1)[1].strip()
            continue
        out_key = _match(word, _OUTPUT_KEYWORDS)
        if out_key:
            target = value if value in ("ALL", "NONE") else int(value)
            options = [o.strip() for o in opts.split(",")] if opts else []
            current["outputs"][out_key] = {"set": target, "options": options}
            continue
        sel_key = _match(word, _SELECTORS)
        if sel_key:
            entry: dict = {"set": int(value)}
            if sel_key == "TEMPERATURE":
                entry["type"] = (opts or "BOTH").strip()
            current["selectors"][sel_key] = entry
    if not subcases:
        subcases = [{"id": 1, "selectors": {}, "outputs": {}, "text": {}}]
    resolved = []
    for sc in subcases:
        resolved.append(
            {
                "id": sc["id"],
                "label": sc["text"].get("LABEL", default["text"].get("LABEL")),
                "selectors": {**default["selectors"], **sc["selectors"]},
                "outputs": {**default["outputs"], **sc["outputs"]},
                "scan": default.get("scan", []) + sc.get("scan", []),
            }
        )
    return {
        "title": default["text"].get("TITLE"),
        "subtitle": default["text"].get("SUBTITLE"),
        "sets": {str(k): v for k, v in sorted(sets.items())},
        "subcases": resolved,
    }
