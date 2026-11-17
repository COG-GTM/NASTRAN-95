"""Golden reference values parsed from the COSMIC listings (``demoout/*.out``).

One JSON file per inventory deck under ``tests/reference/golden/`` holds every
displacement, SPC force, load vector, element force and element stress table
(see :mod:`nastran95py.reference.results`) plus any ``MATPRN`` matrix prints.

Comparison tolerance
--------------------
The listings print 5-7 significant figures (``E14.6``/``E12.5``; principal
angles to 4 decimals) from a single-precision run, so a port is compared as::

    |actual - golden| <= atol + rtol * |golden|
    rtol = 1e-4
    atol = 1e-4 * max |golden| over that column of that table

i.e. a relative 1e-4 test with an absolute floor scaled to the largest
magnitude of each quantity. The floor absorbs round-off noise printed for
values that are zero in exact arithmetic (e.g. ``3.456080E-10`` SPC forces
next to ``4.3E+03``). A column that is entirely zero (rotations, rod torque)
falls back to the largest magnitude in the whole table. Safety margins are
printed to 2 significant figures (``2.9E+00``) so their ``rtol`` is 5e-2.

Regenerate with ``python -m nastran95py.reference.golden`` (``--check`` fails
if the files are stale).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nastran95py.reference.inventory import PACKAGE_ROOT, load_manifest, repo_root
from nastran95py.reference.listing import index_listing
from nastran95py.reference.results import Table, parse_tables

GOLDEN_DIR = PACKAGE_ROOT / "tests" / "reference" / "golden"
SCHEMA = 1
RTOL = 1e-4
ATOL_SCALE = 1e-4
MARGIN_RTOL = 5e-2
MARGIN_COLUMNS = ("margin", "axial_margin", "torsional_margin")
TOLERANCE = {
    "rtol": RTOL,
    "atol_scale": ATOL_SCALE,
    "atol": "atol_scale * max|golden| of the column (whole table if the column is all zero)",
    "rtol_overrides": dict.fromkeys(MARGIN_COLUMNS, MARGIN_RTOL),
}


def golden_path(deck_id: str) -> Path:
    return GOLDEN_DIR / f"{deck_id}.json"


def _table_json(t: Table) -> dict:
    out = {
        "key": t.key,
        "kind": t.kind,
        "element_type": t.element_type,
        "subcase": t.subcase,
        "occurrence": t.occurrence,
        "sort": t.sort,
        "scan": t.scan,
        "title": t.title,
        "listing_lines": t.lines,
        "columns": ["id", *t.columns],
        "rows": [[i, *r] for i, r in zip(t.ids, t.rows, strict=True)],
    }
    if t.point_types is not None:
        out["point_types"] = t.point_types
    return out


def build_golden(deck: dict, root: Path | None = None) -> dict:
    root = root or repo_root()
    text = (root / deck["listing_path"]).read_text(errors="replace")
    subcases = [s["id"] for s in deck["case_control"]["subcases"]]
    index = index_listing(text)
    tables = parse_tables(text, subcases[0] if len(subcases) == 1 else None, index)
    return {
        "schema": SCHEMA,
        "deck": deck["id"],
        "listing": deck["listing_path"],
        "listing_sha256": deck["sha256"]["listing"],
        "source": "parsed from the COSMIC listing by nastran95py.reference.golden",
        "tolerance": TOLERANCE,
        "subcases": subcases,
        "tables": [_table_json(t) for t in tables],
        "matrices": index["matrices"],
    }


_INNER_LIST = re.compile(r"\[\s*([^\[\]{}]*?)\s*\]")


def dumps(golden: dict) -> str:
    """JSON with one table row (innermost list) per line for readable diffs."""
    text = json.dumps(golden, indent=1)
    return (
        _INNER_LIST.sub(lambda m: "[" + ", ".join(s.strip() for s in m[1].split(",")) + "]", text)
        + "\n"
    )


def load_golden(deck_id: str, directory: Path = GOLDEN_DIR) -> dict:
    return json.loads((Path(directory) / f"{deck_id}.json").read_text())


@dataclass
class GoldenTable:
    key: str
    columns: tuple[str, ...]
    ids: np.ndarray
    values: np.ndarray  # float, NaN where the listing left the field blank
    meta: dict

    def column(self, name: str) -> np.ndarray:
        return self.values[:, self.columns.index(name)]

    def by_id(self) -> dict[int, np.ndarray]:
        return {int(i): row for i, row in zip(self.ids, self.values, strict=True)}


def tables(golden: dict) -> dict[str, GoldenTable]:
    """Golden tables keyed by ``Table.key`` as NumPy arrays."""
    out = {}
    for t in golden["tables"]:
        rows = t["rows"]
        vals = np.array(
            [[np.nan if v is None else v for v in r[1:]] for r in rows], dtype=float
        ).reshape(len(rows), len(t["columns"]) - 1)
        meta = {k: v for k, v in t.items() if k not in ("rows", "columns")}
        out[t["key"]] = GoldenTable(
            t["key"], tuple(t["columns"][1:]), np.array([r[0] for r in rows]), vals, meta
        )
    return out


def tolerances(table: GoldenTable) -> tuple[np.ndarray, np.ndarray]:
    """Per-column ``(rtol, atol)`` for comparing against ``table`` (see module doc)."""
    mags = np.abs(np.nan_to_num(table.values))
    col_max = mags.max(axis=0) if mags.size else np.zeros(len(table.columns))
    floor = np.where(col_max > 0, col_max, mags.max() if mags.size else 0.0)
    rtol = np.array([MARGIN_RTOL if c in MARGIN_COLUMNS else RTOL for c in table.columns])
    return rtol, ATOL_SCALE * floor


@dataclass
class Comparison:
    key: str
    n: int
    max_abs: float
    max_rel: float
    failures: int

    @property
    def passed(self) -> bool:
        return self.failures == 0


def compare(table: GoldenTable, actual) -> Comparison:
    """Compare ``actual`` (same shape as ``table.values``) to the golden table.

    Blank golden fields (NaN) are skipped. ``max_rel`` only counts entries
    above the absolute floor, where a relative error is meaningful.
    """
    actual = np.asarray(actual, dtype=float)
    if actual.shape != table.values.shape:
        raise ValueError(f"{table.key}: shape {actual.shape} != {table.values.shape}")
    rtol, atol = tolerances(table)
    mask = ~np.isnan(table.values)
    g = np.where(mask, table.values, 0.0)
    a = np.where(mask, actual, 0.0)
    err = np.abs(a - g)
    ok = (err <= atol + rtol * np.abs(g)) | ~mask
    big = mask & (np.abs(g) > atol)
    rel = np.divide(err, np.abs(g), out=np.zeros_like(err), where=big)
    return Comparison(
        table.key,
        int(mask.sum()),
        float(err.max(initial=0.0)),
        float(rel.max(initial=0.0)),
        int((~ok).sum()),
    )


MEMBRANE_TYPES = ("CTRMEM", "CQDMEM", "CQDMEM1", "CQDMEM2")
_FLOAT = re.compile(r"^[-+]?\d*\.\d*(?:[ED][-+]?\d+)?$")


def membrane_consistency(table: GoldenTable) -> Comparison:
    """Recompute the principal stresses, angle and max shear of a membrane table.

    Checks the column mapping and that the printed precision fits the
    tolerance: Mohr's circle from the printed NORMAL-X/NORMAL-Y/SHEAR-XY must
    reproduce the printed MAJOR/MINOR/MAX SHEAR/ANGLE within :func:`compare`.
    """
    sx, sy, txy = (table.column(c) for c in ("normal_x", "normal_y", "shear_xy"))
    centre, radius = (sx + sy) / 2, np.hypot((sx - sy) / 2, txy)
    angle = np.degrees(0.5 * np.arctan2(2 * txy, sx - sy))
    expected = table.values.copy()
    for name, v in (("angle", angle), ("major", centre + radius), ("minor", centre - radius)):
        expected[:, table.columns.index(name)] = v
    expected[:, table.columns.index("max_shear")] = radius
    return compare(table, expected)


def listing_roundtrip(golden: dict, root: Path | None = None) -> Comparison:
    """Golden values vs an independent re-tokenisation of the listing lines.

    Every decimal token in each table's listing line range is collected
    without any column logic and compared, sorted, with the golden values of
    the tables built from that range. Exact parsing gives zero error.
    """
    root = root or repo_root()
    lines = (root / golden["listing"]).read_text(errors="replace").splitlines()
    by_range: dict[tuple, list[float]] = {}
    for t in golden["tables"]:
        vals = by_range.setdefault(tuple(t["listing_lines"]), [])
        vals.extend(v for r in t["rows"] for v in r[1:] if v is not None)
    n, max_abs, max_rel, failures = 0, 0.0, 0.0, 0
    for (first, last), vals in by_range.items():
        toks = [
            float(tok.replace("D", "E"))
            for ln in lines[first - 1 : last]
            for tok in ln[1:].split()
            if _FLOAT.match(tok)
        ]
        if len(toks) != len(vals):
            failures += abs(len(toks) - len(vals)) or 1
            continue
        g, a = np.sort(vals), np.sort(toks)
        err = np.abs(a - g)
        n += len(g)
        max_abs = max(max_abs, float(err.max(initial=0.0)))
        nz = g != 0
        max_rel = max(max_rel, float((err[nz] / np.abs(g[nz])).max(initial=0.0)))
        failures += int((err > 0).sum())
    return Comparison(golden["deck"], n, max_abs, max_rel, failures)


def report(directory: Path = GOLDEN_DIR) -> str:
    """Parity table: golden vs listing tokens and Mohr's-circle consistency."""
    head = f"{'deck':<9}{'check':<44}{'values':>8}{'max abs err':>14}{'max rel err':>14}  ok"
    out = [head]
    for deck in load_manifest()["decks"]:
        golden = load_golden(deck["id"], directory)
        checks = [("listing round-trip", listing_roundtrip(golden))]
        for key, t in tables(golden).items():
            if t.meta["kind"] == "stress" and t.meta["element_type"] in MEMBRANE_TYPES:
                checks.append((f"mohr {key}", membrane_consistency(t)))
        for name, c in checks:
            out.append(
                f"{deck['id']:<9}{name:<44}{c.n:>8}{c.max_abs:>14.3e}{c.max_rel:>14.3e}"
                f"  {'yes' if c.passed else 'NO'}"
            )
    return "\n".join(out)


def build_all(root: Path | None = None) -> dict[str, str]:
    """Rendered golden JSON for every deck in the inventory manifest."""
    return {d["id"]: dumps(build_golden(d, root)) for d in load_manifest()["decks"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write golden files from demoout listings")
    parser.add_argument("--check", action="store_true", help="fail if golden files are stale")
    parser.add_argument("--report", action="store_true", help="print the parity table")
    args = parser.parse_args(argv)
    if args.report:
        print(report())
        return 0
    files = {golden_path(k): body for k, body in build_all().items()}
    stale = [p for p, body in files.items() if not p.exists() or p.read_text() != body]
    if args.check:
        for p in stale:
            print(f"stale: {p}", file=sys.stderr)
        return 1 if stale else 0
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for p, body in files.items():
        p.write_text(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
