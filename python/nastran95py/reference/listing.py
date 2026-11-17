"""Index of a NASTRAN-95 printed listing (``demoout/*.out``).

Locates the output sections (displacements, SPC forces, element stresses and
forces, ...) by page and line so later phases can parse golden values, and
extracts the sorted bulk data echo and ``MATPRN`` matrix prints.
"""

from __future__ import annotations

import re

_PAGE = re.compile(r"^1.*/\s*PAGE\s+(\d+)\s*$")
_SUBCASE = re.compile(r"SUBCASE\s+(\d+)\s*$")
_SORT2_KEY = re.compile(r"^\s*(POINT|ELEMENT)-ID\s*=\s*(\d+)")
_ECHO = re.compile(r"^\s+\d+-\s{8}(.*)$")
_ELEM = re.compile(r"\((C[A-Z0-9]+)\)$")
_MATRIX = re.compile(
    r"MATRIX\s+(\S+)\s+\(GINO NAME\s+\d+\s*\)\s+IS A\s+(.+?)\s+(\d+)\s+COLUMN X"
    r"\s+(\d+)\s+ROW\s+(\S+)"
)
_COLUMN = re.compile(r"^0COLUMN\s+(\d+)\s+ROWS\s+(\d+)\s+THRU\s+(\d+)")
_INT = re.compile(r"\d+")
_SCAN = re.compile(r"SCANNED BY FIELD:\s+(\S+)\s+SET:\s+(\d+)")


def spaced_title(line: str) -> str | None:
    """Collapse a letter-spaced heading (``S T R E S S E S   I N ...``)."""
    s = line[1:].strip() if line[:1] in "01 " else line.strip()
    if len(s) < 9 or not s[0].isalpha():
        return None
    probe = s[:12]
    if any(probe[i] != " " for i in range(1, len(probe), 2)):
        return None
    words = re.split(r"\s{2,}", s)
    return " ".join(w.replace(" ", "") for w in words)


def classify(title: str) -> str:
    if title == "DISPLACEMENT VECTOR":
        return "displacement"
    if title == "FORCES OF SINGLE-POINT CONSTRAINT":
        return "spcforce"
    if title == "LOAD VECTOR":
        return "oload"
    if title.startswith("STRESSES"):
        return "stress"
    if title.startswith(("FORCES IN", "FORCES ACTING ON")):
        return "force"
    if title == "ELEMENT STRAIN ENERGIES":
        return "ese"
    if title == "GRID POINT FORCE BALANCE":
        return "gpforce"
    if title == "SORTED BULK DATA ECHO":
        return "bulk_echo"
    return "other"


def _pages(lines: list[str]) -> list[tuple[int, int, int]]:
    """(page number, first line index, end line index) for each page."""
    starts = [(i, int(m[1])) for i, ln in enumerate(lines) if (m := _PAGE.match(ln))]
    out = []
    for k, (i, num) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        out.append((num, i, end))
    return out


def _row_ids(rows: list[str]) -> list[int]:
    ids: list[int] = []
    for row in rows:
        ids.extend(int(t) for t in row.split() if _INT.fullmatch(t))
    return ids


def index_listing(text: str) -> dict:
    lines = text.splitlines()
    end_of_job = next((i for i, ln in enumerate(lines) if "END OF JOB" in ln), len(lines))
    sections: list[dict] = []
    occurrences: dict[tuple, int] = {}
    last_key = None
    for page, start, end in _pages(lines):
        if start >= end_of_job:
            break
        end = min(end, end_of_job)
        label = lines[start + 2] if start + 2 < end else ""
        m_sc = _SUBCASE.search(label)
        subcase = int(m_sc[1]) if m_sc else None
        sort2_id = None
        scan = None
        for i in range(start + 1, min(start + 6, end)):
            if m := _SCAN.search(lines[i]):
                scan = {"field": m[1], "set": int(m[2])}
        title_at = [(i, t) for i in range(start + 3, end) if (t := spaced_title(lines[i]))]
        for i in range(start + 3, end):
            if m := _SORT2_KEY.match(lines[i]):
                sort2_id = int(m[2])
                break
        if not title_at:
            last_key = None
            continue
        for k, (i, title) in enumerate(title_at):
            block_end = title_at[k + 1][0] if k + 1 < len(title_at) else end
            rows = [
                ln
                for ln in lines[i + 1 : block_end]
                if ln[1:].split() and _INT.fullmatch(ln[1:].split()[0])
            ]
            if classify(title) == "bulk_echo":
                rows = []
            scan_key = (scan["field"], scan["set"]) if scan else None
            key = (title, subcase, sort2_id is not None, scan_key)
            if key == last_key and sections:
                sec = sections[-1]
                sec["pages"][1] = page
                sec["lines"][1] = block_end
            else:
                occ_key = (title, subcase, scan_key)
                occ = occurrences.get(occ_key, 0) + 1
                occurrences[occ_key] = occ
                m_el = _ELEM.search(title)
                sec = {
                    "title": title,
                    "kind": classify(title),
                    "element_type": m_el[1] if m_el else None,
                    "subcase": subcase,
                    "sort": "SORT2" if sort2_id is not None else "SORT1",
                    "occurrence": occ,
                    "scan": scan,
                    "pages": [page, page],
                    "lines": [i + 1, block_end],
                    "rows": 0,
                    "ids": [],
                    "subcases_in_rows": [],
                }
                sections.append(sec)
            sec["rows"] += len(rows)
            if sort2_id is not None:
                if sort2_id not in sec["ids"]:
                    sec["ids"].append(sort2_id)
                # SORT2 rows lead with the subcase; CSHEAR packs two subcases per row.
                for sc in _row_ids(rows):
                    if sc not in sec["subcases_in_rows"]:
                        sec["subcases_in_rows"].append(sc)
            elif sec["kind"] in ("displacement", "spcforce", "oload", "stress", "force"):
                sec["ids"].extend(_row_ids(rows))
            last_key = key
    for sec in sections:
        if sec["sort"] == "SORT1":
            sec.pop("subcases_in_rows")
    return {
        "pages": max((p for p, _, _ in _pages(lines)), default=0),
        "lines": len(lines),
        "sections": sections,
        "bulk_echo": bulk_echo(lines),
        "epsilon": [
            {"subcase": int(m[1]), "value": float(m[2])}
            for ln in lines
            if (m := re.search(r"FOR SUBCASE NUMBER\s+(\d+), EPSILON SUB E =\s*(\S+)", ln))
        ],
        "fatal_messages": sum(1 for ln in lines if "FATAL MESSAGE" in ln),
        "warning_messages": sum(1 for ln in lines if "WARNING MESSAGE" in ln),
        "matrices": matrix_prints(lines),
    }


def bulk_echo(lines: list[str]) -> list[str]:
    """Card images (columns 1-80) of the SORTED BULK DATA ECHO, one per physical card."""
    cards: list[str] = []
    in_echo = False
    for ln in lines:
        if (t := spaced_title(ln)) is not None:
            in_echo = t == "SORTED BULK DATA ECHO"
            continue
        if in_echo:
            if ln.strip() == "ENDDATA":
                in_echo = False
                continue
            if m := _ECHO.match(ln):
                cards.append(m[1][:80])
    return cards


def matrix_prints(lines: list[str]) -> list[dict]:
    """Parse ``MATPRN`` output into ``{name, rows, cols, form, line, entries}``.

    ``entries`` holds the non-zero terms as ``[row, col, value]`` (1-based).
    """
    out: list[dict] = []
    i = 0
    while i < len(lines):
        m = _MATRIX.search(lines[i])
        if not m:
            i += 1
            continue
        mat = {
            "name": m[1],
            "type": m[2],
            "cols": int(m[3]),
            "rows": int(m[4]),
            "form": m[5].rstrip("."),
            "line": i + 1,
            "entries": [],
        }
        i += 1
        while i < len(lines) and not lines[i].startswith("0THE NUMBER"):
            mc = _COLUMN.match(lines[i])
            if not mc:
                i += 1
                continue
            col, row = int(mc[1]), int(mc[2])
            i += 1
            while i < len(lines) and not lines[i].startswith("0"):
                for tok in lines[i].split():
                    v = float(tok.replace("D", "E"))
                    if v != 0.0:
                        mat["entries"].append([row, col, v])
                    row += 1
                i += 1
        out.append(mat)
    return out
