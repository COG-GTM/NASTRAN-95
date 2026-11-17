"""Inventory of the Rigid Format 1 demo decks and their reference listings.

``python -m nastran95py.reference.inventory`` regenerates
``python/tests/reference/manifest.json`` and ``DECKS.md``; ``--check`` fails
when either is stale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

from nastran95py.reference.cards import Card, card_signature, expand_ids, read_deck
from nastran95py.reference.casecontrol import parse_case_control, parse_executive
from nastran95py.reference.listing import index_listing

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PACKAGE_ROOT / "tests" / "reference" / "manifest.json"
DECKS_MD_PATH = PACKAGE_ROOT / "tests" / "reference" / "DECKS.md"

TARGET_ELEMENTS = ("CROD", "CONROD", "CSHEAR", "CTRMEM", "CQDMEM", "CQDMEM1", "CQDMEM2")

RF1_DECKS: dict[str, str] = {
    "d01011a": "Delta wing: CROD, CONROD, CSHEAR, CTRMEM, CQDMEM under a trailing-edge load",
    "d01012a": "Delta wing: CQDMEM1 and CQDMEM2 halves, two load subcases, SORT2 output",
    "d01013a": "Delta wing with CQDMEM1 skins; rod and shear forces plus grid point force balance",
    "d01014a": "Delta wing with CQDMEM2 skins; element strain energies",
    "d01032a": "Free plate, CQDMEM1, linear thermal load, temperature-dependent MAT1",
    "d01033a": "Free plate, CQDMEM2, linear thermal load, temperature-dependent MAT1",
    "d01034a": "As d01033a with SCAN output selection on CQDMEM2 stresses",
    "d01161a": "Fully stressed design of a reinforced hole (CQDMEM, CTRMEM, CROD); FSD iterations",
    "t01241a": "Three-rod bar (CROD); DMAP ALTER prints KGGX and exits before the solve",
}

NOTES: dict[str, list[str]] = {
    "d01161a": [
        "POPT/PLIMIT drive a fully stressed design loop: each subcase prints one block per "
        "iteration (`occurrence` in the listing index); the first occurrence is the analysis "
        "of the as-input properties.",
        "TEMPERATURE(MATERIALS) = 3000 selects TEMPD 3000 for MATT1/TABLEM1 lookups only.",
    ],
    "t01241a": [
        "The DMAP ALTER prints the global stiffness KGGX (MATPRN) and then `JUMP FINIS`, so the "
        "listing has no displacements; KGGX is the CROD stiffness reference.",
        "The ALTER deliberately contains erroneous PARAML/PARAMR/PARAMD calls (marked "
        "INTENTIONAL ERROR in the deck).",
    ],
    "d01012a": ["Output is SORT2: sections are keyed by POINT-ID / ELEMENT-ID per page."],
    "d01034a": ["SCAN cards restrict which CQDMEM2 stress rows are printed."],
    "d01032a": [
        "TEMPERATURE = 1 (BOTH): TEMP set 1 is both the thermal load and the material temperature."
    ],
}
NOTES["d01033a"] = NOTES["d01032a"]
NOTES["d01034a"] = NOTES["d01034a"] + NOTES["d01032a"]

FORTRAN_STATUS = (
    "bin/nastrn.exe is a 32-bit SPARC Solaris binary (not runnable on x86-64 Linux) and the"
    " sources in mis/ need a Fortran 77 rebuild; the demoout/*.out listings are the reference."
)

PROPERTY_CARDS = ("PROD", "PSHEAR", "PTRMEM", "PQDMEM", "PQDMEM1", "PQDMEM2")
MATERIAL_CARDS = ("MAT1", "MAT2", "MATT1", "TABLEM1")
MAT1_FIELDS = ("MID", "E", "G", "NU", "RHO", "A", "TREF", "GE", "ST", "SC", "SS", "MCSID")
MAT2_FIELDS = (
    "MID",
    "G11",
    "G12",
    "G13",
    "G22",
    "G23",
    "G33",
    "RHO",
    "A1",
    "A2",
    "A12",
    "TREF",
    "GE",
    "ST",
    "SC",
    "SS",
    "MCSID",
)
MATT1_FIELDS = ("MID", "E", "G", "NU", "RHO", "A", None, "GE", "ST", "SC", "SS")


def repo_root() -> Path:
    env = os.environ.get("NASTRAN95_ROOT")
    if env:
        return Path(env)
    for parent in Path(__file__).resolve().parents:
        if (parent / "inp").is_dir() and (parent / "demoout").is_dir():
            return parent
    raise FileNotFoundError("NASTRAN-95 checkout with inp/ and demoout/ not found")


def _named(card: Card, names) -> dict:
    return {n: card.get(i + 1) for i, n in enumerate(names) if n and card.get(i + 1) is not None}


def _elements(cards: list[Card]) -> dict:
    elems: dict[str, list[dict]] = {}
    for c in cards:
        if c.name == "CROD":
            for k in range(0, len(c.fields), 4):
                eid = c.get(k + 1)
                if eid is not None:
                    elems.setdefault("CROD", []).append(
                        {
                            "eid": eid,
                            "pid": c.get(k + 2, eid),
                            "grids": [c.get(k + 3), c.get(k + 4)],
                        }
                    )
        elif c.name == "CONROD":
            elems.setdefault("CONROD", []).append(
                {
                    "eid": c.get(1),
                    "grids": [c.get(2), c.get(3)],
                    "mid": c.get(4),
                    "A": c.get(5),
                    "J": c.get(6),
                    "C": c.get(7),
                    "NSM": c.get(8),
                }
            )
        elif c.name in ("CSHEAR", "CTRMEM", "CQDMEM", "CQDMEM1", "CQDMEM2"):
            n = 3 if c.name == "CTRMEM" else 4
            elems.setdefault(c.name, []).append(
                {
                    "eid": c.get(1),
                    "pid": c.get(2, c.get(1)),
                    "grids": [c.get(3 + i) for i in range(n)],
                    **(
                        {"theta": c.get(3 + n)}
                        if c.name != "CSHEAR" and c.get(3 + n) is not None
                        else {}
                    ),
                }
            )
    out = {}
    for name in TARGET_ELEMENTS:
        if name in elems:
            rows = sorted(elems[name], key=lambda e: e["eid"])
            out[name] = {
                "count": len(rows),
                "ids": [e["eid"] for e in rows],
                "property_ids": sorted({e["pid"] for e in rows if "pid" in e}),
                "connectivity": rows,
            }
    return out


def _properties(cards: list[Card]) -> dict:
    out: dict[str, list[dict]] = {}
    for c in cards:
        if c.name == "PROD":
            out.setdefault("PROD", []).append(_named(c, ("PID", "MID", "A", "J", "C", "NSM")))
        elif c.name in PROPERTY_CARDS:
            for k in range(0, len(c.fields), 4):
                if c.get(k + 1) is not None:
                    out.setdefault(c.name, []).append(
                        {
                            n: c.get(k + 1 + i)
                            for i, n in enumerate(("PID", "MID", "T", "NSM"))
                            if c.get(k + 1 + i) is not None
                        }
                    )
    return {k: sorted(v, key=lambda p: p["PID"]) for k, v in sorted(out.items())}


def _materials(cards: list[Card]) -> dict:
    out: dict[str, list[dict]] = {}
    for c in cards:
        if c.name == "MAT1":
            out.setdefault("MAT1", []).append(_named(c, MAT1_FIELDS))
        elif c.name == "MAT2":
            out.setdefault("MAT2", []).append(_named(c, MAT2_FIELDS))
        elif c.name == "MATT1":
            out.setdefault("MATT1", []).append(_named(c, MATT1_FIELDS))
        elif c.name == "TABLEM1":
            pts = [v for v in c.fields[8:] if v is not None]
            if "ENDT" in pts:
                pts = pts[: pts.index("ENDT")]
            out.setdefault("TABLEM1", []).append(
                {
                    "ID": c.get(1),
                    "points": [[pts[i], pts[i + 1]] for i in range(0, len(pts) - 1, 2)],
                }
            )
    return dict(sorted(out.items()))


def _loads(cards: list[Card]) -> dict:
    out: dict = {}
    for c in cards:
        if c.name in ("FORCE", "MOMENT"):
            out.setdefault(c.name, []).append(
                {
                    "SID": c.get(1),
                    "G": c.get(2),
                    "CID": c.get(3, 0),
                    "F": c.get(4),
                    "N": [c.get(5, 0.0), c.get(6, 0.0), c.get(7, 0.0)],
                }
            )
        elif c.name == "TEMP":
            for k in range(1, 7, 2):
                if c.get(k + 1) is not None:
                    out.setdefault("TEMP", {}).setdefault(str(c.get(1)), {})[str(c.get(k + 1))] = (
                        c.get(k + 2)
                    )
        elif c.name == "TEMPD":
            for k in range(0, len(c.fields), 2):
                if c.get(k + 1) is not None:
                    out.setdefault("TEMPD", {})[str(c.get(k + 1))] = c.get(k + 2)
        elif c.name == "LOAD":
            out.setdefault("LOAD", []).append({"SID": c.get(1), "fields": c.fields[1:]})
    if "TEMP" in out:
        out["TEMP"] = {
            sid: {"count": len(t), "min": min(t.values()), "max": max(t.values()), "grids": t}
            for sid, t in out["TEMP"].items()
        }
    return out


def _constraints(cards: list[Card]) -> dict:
    spc1: dict[str, list[dict]] = {}
    grdset = None
    for c in cards:
        if c.name == "SPC1":
            spc1.setdefault(str(c.get(1)), []).append(
                {"C": str(c.get(2)), "grids": expand_ids(c.fields[2:])}
            )
        elif c.name == "GRDSET":
            grdset = {"CP": c.get(2), "CD": c.get(6), "PS": c.get(7) and str(c.get(7))}
    return {"GRDSET": grdset, "SPC1": spc1}


def _grids(cards: list[Card]) -> dict:
    grids = [c for c in cards if c.name == "GRID"]
    cp = sorted({c.get(2, 0) for c in grids})
    cd = sorted({c.get(6, 0) for c in grids})
    return {
        "count": len(grids),
        "ids": sorted(c.get(1) for c in grids),
        "cp": cp,
        "cd": cd,
        "coordinate_systems": sorted({c.name for c in cards if c.name.startswith("CORD")}),
    }


def _requested_ids(case: dict, subcase: dict, kind: str, universe: list[int]) -> list[int]:
    target = subcase["outputs"][kind]["set"]
    if target == "ALL":
        return sorted(universe)
    members = case["sets"].get(str(target), [])
    if members == "ALL":
        return sorted(universe)
    return sorted(set(members) & set(universe))


def _listing_coverage(deck: dict) -> list[dict]:
    """Compare ids requested by DISPLACEMENT/STRESS with ids printed in the listing."""
    checks = []
    case, listing = deck["case_control"], deck["listing"]
    all_eids = [i for e in deck["elements"].values() for i in e["ids"]]
    for sc in case["subcases"]:
        for kind, out_kind, universe in (
            ("DISPLACEMENT", "displacement", deck["grids"]["ids"]),
            ("STRESS", "stress", all_eids),
        ):
            if kind not in sc["outputs"]:
                continue
            secs = [
                s
                for s in listing["sections"]
                if s["kind"] == out_kind
                and s["occurrence"] == 1
                and s["scan"] is None
                and (s["subcase"] in (None, sc["id"]))
                and (s["sort"] == "SORT1" or sc["id"] in s.get("subcases_in_rows", []))
                and (s["element_type"] in (None, *TARGET_ELEMENTS))
            ]
            listed = sorted({i for s in secs for i in s["ids"]})
            requested = _requested_ids(case, sc, kind, universe)
            checks.append(
                {
                    "subcase": sc["id"],
                    "request": kind,
                    "requested": len(requested),
                    "listed": len(listed),
                    "missing_from_listing": sorted(set(requested) - set(listed)),
                    "unrequested_in_listing": sorted(set(listed) - set(requested)),
                    "scan_filtered": bool(sc["scan"]),
                }
            )
    return checks


def rod_kggx_parity(deck: dict, cards: list[Card]) -> dict | None:
    """Closed-form CROD axial stiffness AE/L vs the KGGX terms printed by MATPRN."""
    mats = [m for m in deck["listing"]["matrices"] if m["name"] == "KGGX"]
    if not mats or set(deck["elements"]) - {"CROD", "CONROD"}:
        return None
    xyz = {
        c.get(1): (c.get(3, 0.0), c.get(4, 0.0), c.get(5, 0.0)) for c in cards if c.name == "GRID"
    }
    prod = {p["PID"]: p for p in deck["properties"].get("PROD", [])}
    mat1 = {m["MID"]: m for m in deck["materials"].get("MAT1", [])}
    order = sorted(xyz)
    dof = {g: 6 * k for k, g in enumerate(order)}
    expected: dict[tuple[int, int], float] = {}
    for e in deck["elements"]["CROD"]["connectivity"]:
        p = prod[e["pid"]]
        g1, g2 = e["grids"]
        d = [b - a for a, b in zip(xyz[g1], xyz[g2], strict=True)]
        length = math.sqrt(sum(x * x for x in d))
        k = mat1[p["MID"]]["E"] * p["A"] / length
        u = [x / length for x in d]
        for a, ga in ((0, g1), (1, g2)):
            for b, gb in ((0, g1), (1, g2)):
                sign = 1.0 if a == b else -1.0
                for i in range(3):
                    for j in range(3):
                        key = (dof[ga] + i + 1, dof[gb] + j + 1)
                        expected[key] = expected.get(key, 0.0) + sign * k * u[i] * u[j]
    printed: dict[tuple[int, int], float] = {}
    for r, c, v in mats[0]["entries"]:
        printed[(r, c)] = v
        printed[(c, r)] = v
    keys = {k for k, v in expected.items() if v != 0.0} | set(printed)
    abs_err = max(abs(expected.get(k, 0.0) - printed.get(k, 0.0)) for k in keys)
    rel_err = max(
        abs(expected.get(k, 0.0) - printed.get(k, 0.0)) / max(abs(printed.get(k, 0.0)), 1e-300)
        for k in keys
    )
    return {
        "matrix": "KGGX",
        "terms_compared": len(keys),
        "max_abs_error": abs_err,
        "max_rel_error": rel_err,
    }


def build_deck(deck_id: str, root: Path) -> dict:
    inp, out = root / "inp" / f"{deck_id}.inp", root / "demoout" / f"{deck_id}.out"
    text = inp.read_text(errors="replace")
    deck = read_deck(text)
    listing_text = out.read_text(errors="replace")
    listing = index_listing(listing_text)
    cards = deck.bulk
    entry = {
        "id": deck_id,
        "description": RF1_DECKS[deck_id],
        "inp": f"inp/{deck_id}.inp",
        "listing_path": f"demoout/{deck_id}.out",
        "sha256": {
            "inp": hashlib.sha256(inp.read_bytes()).hexdigest(),
            "listing": hashlib.sha256(out.read_bytes()).hexdigest(),
        },
        "executive": parse_executive(deck.executive),
        "case_control": parse_case_control(deck.case_control),
        "grids": _grids(cards),
        "elements": _elements(cards),
        "properties": _properties(cards),
        "materials": _materials(cards),
        "loads": _loads(cards),
        "constraints": _constraints(cards),
        "params": {str(c.get(1)): c.get(2) for c in cards if c.name == "PARAM"},
        "card_counts": dict(sorted(Counter(c.name for c in cards).items())),
        "listing": {k: v for k, v in listing.items() if k != "bulk_echo"},
        "notes": NOTES.get(deck_id, []),
    }
    echo = Counter(card_signature(c) for c in listing["bulk_echo"])
    physical = Counter(card_signature(c) for c in deck.bulk_physical)
    entry["checks"] = {
        "bulk_echo_matches_deck": echo == physical,
        "bulk_echo_cards": sum(echo.values()),
        "deck_physical_cards": sum(physical.values()),
        "listing_coverage": _listing_coverage(entry),
    }
    parity = rod_kggx_parity(entry, cards)
    if parity:
        entry["checks"]["kggx_parity"] = parity
    return entry


def build_manifest(root: Path | None = None) -> dict:
    root = root or repo_root()
    return {
        "schema": 1,
        "rigid_format": 1,
        "reference_source": "demoout/*.out (COSMIC 1995 SUN SOLARIS listings)",
        "fortran_run": False,
        "fortran_run_reason": FORTRAN_STATUS,
        "target_elements": list(TARGET_ELEMENTS),
        "decks": [build_deck(d, root) for d in RF1_DECKS],
    }


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    return json.loads(Path(path).read_text())


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def scan_label(section: dict) -> str:
    scan = section.get("scan")
    return f" (SCAN {scan['field']}, set {scan['set']})" if scan else ""


def render_markdown(manifest: dict) -> str:
    out = [
        "# Rigid Format 1 reference decks",
        "",
        "Generated by `python -m nastran95py.reference.inventory` from `inp/` and `demoout/`;"
        " do not edit by hand. The machine-readable form is [`manifest.json`](manifest.json).",
        "",
        "Reference values come from the checked-in COSMIC listings (`demoout/*.out`). The"
        " Fortran executable was not rerun: " + FORTRAN_STATUS[0].lower() + FORTRAN_STATUS[1:],
        "",
        "Output request names are canonical: `STRESS` covers `ELSTRESS`/`STRESSES`, `FORCE`"
        " covers `ELFORCE`, `SPCFORCE` covers `SPCF`. Listing sections give page and line"
        " ranges into the `.out` file; `occurrence` > 1 marks repeated blocks (design"
        " iterations in d01161a).",
        "",
        "| Deck | SOL | Subcases | " + " | ".join(TARGET_ELEMENTS) + " | Loads | Output requests |",
        "|---|---|---|" + "---|" * len(TARGET_ELEMENTS) + "---|---|",
    ]
    for d in manifest["decks"]:
        counts = [str(d["elements"].get(e, {}).get("count", "")) for e in TARGET_ELEMENTS]
        loads = sorted(k for k in d["loads"])
        outs = sorted({k for sc in d["case_control"]["subcases"] for k in sc["outputs"]})
        out.append(
            f"| [{d['id']}](#{d['id']}) | {d['executive']['sol']} | "
            f"{len(d['case_control']['subcases'])} | "
            + " | ".join(counts)
            + f" | {', '.join(loads) or '-'} | {', '.join(outs) or '-'} |"
        )
    for d in manifest["decks"]:
        cc = d["case_control"]
        out += [
            "",
            f"## {d['id']}",
            "",
            f"{d['description']}.",
            "",
            f"- Title: {cc['title']}",
            f"- Subtitle: {cc['subtitle']}",
            f"- Deck: `{d['inp']}`; listing: `{d['listing_path']}` ({d['listing']['pages']} pages)",
            f"- Executive: APP {d['executive']['app']}, SOL {d['executive']['sol']}"
            + (", DMAP ALTER" if d["executive"]["alter"] else ""),
            f"- Grids: {d['grids']['count']} (CP {d['grids']['cp']}, CD {d['grids']['cd']});"
            f" GRDSET PS {(d['constraints']['GRDSET'] or {}).get('PS')}",
            "",
            "**Elements**",
            "",
            "| Type | Count | Property ids |",
            "|---|---|---|",
        ]
        for name, e in d["elements"].items():
            out.append(f"| {name} | {e['count']} | {e['property_ids'] or '- (CONROD)'} |")
        out += ["", "**Properties and materials**", ""]
        for name, rows in {**d["properties"], **d["materials"]}.items():
            for r in rows:
                body = ", ".join(f"{k}={_fmt(v)}" for k, v in r.items() if k != "points")
                if "points" in r:
                    body += f", {len(r['points'])} points"
                out.append(f"- {name}: {body}")
        out += ["", "**Loads and constraints**", ""]
        for name in ("FORCE", "MOMENT"):
            by_sid = Counter(f["SID"] for f in d["loads"].get(name, []))
            for sid, n in sorted(by_sid.items()):
                out.append(f"- {name} set {sid}: {n} card(s)")
        for sid, t in d["loads"].get("TEMP", {}).items():
            out.append(
                f"- TEMP set {sid}: {t['count']} grids, {_fmt(t['min'])} to {_fmt(t['max'])}"
            )
        for sid, t in d["loads"].get("TEMPD", {}).items():
            out.append(f"- TEMPD set {sid}: {_fmt(t)}")
        for sid, rows in d["constraints"]["SPC1"].items():
            out.append(
                "- SPC1 set "
                + sid
                + ": "
                + "; ".join(f"C={r['C']} on {len(r['grids'])} grids" for r in rows)
            )
        out += [
            "",
            "**Subcases and output requests**",
            "",
            "| Subcase | Label | Selections | Outputs |",
            "|---|---|---|---|",
        ]
        for sc in cc["subcases"]:
            sel = ", ".join(
                f"{k}={v['set']}" + (f"({v['type']})" if "type" in v else "")
                for k, v in sc["selectors"].items()
            )
            outs = ", ".join(
                f"{k}={v['set']}" + (f" ({','.join(v['options'])})" if v["options"] else "")
                for k, v in sc["outputs"].items()
            )
            for scan in sc["scan"]:
                outs += f", SCAN({','.join(scan['options'])})={scan['value']}"
            out.append(f"| {sc['id']} | {sc['label'] or ''} | {sel} | {outs} |")
        secs = [s for s in d["listing"]["sections"] if s["kind"] not in ("other", "bulk_echo")]
        if secs:
            out += [
                "",
                "**Listing sections**",
                "",
                "| Kind | Element | Subcase | Sort | Occ. | Pages | Lines | Rows |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for s in secs:
                out.append(
                    f"| {s['kind']}{scan_label(s)} | {s['element_type'] or ''} |"
                    f" {s['subcase'] or ''} |"
                    f" {s['sort']} | {s['occurrence']} | {s['pages'][0]}-{s['pages'][1]} |"
                    f" {s['lines'][0]}-{s['lines'][1]} | {s['rows']} |"
                )
        ch = d["checks"]
        out += [
            "",
            "**Checks**",
            "",
            f"- Sorted bulk data echo matches the deck: {ch['bulk_echo_matches_deck']}"
            f" ({ch['bulk_echo_cards']} echoed / {ch['deck_physical_cards']} input cards)",
        ]
        for c in ch["listing_coverage"]:
            out.append(
                f"- Subcase {c['subcase']} {c['request']}: {c['listed']} listed /"
                f" {c['requested']} requested" + (" (SCAN-filtered)" if c["scan_filtered"] else "")
            )
        if "kggx_parity" in ch:
            p = ch["kggx_parity"]
            out.append(
                f"- KGGX vs closed-form CROD AE/L: {p['terms_compared']} terms, max abs error"
                f" {p['max_abs_error']:.3g}, max rel error {p['max_rel_error']:.3g}"
            )
        if d["notes"]:
            out += ["", "**Notes**", ""] + [f"- {n}" for n in d["notes"]]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if outputs are stale")
    args = parser.parse_args(argv)
    manifest = build_manifest()
    files = {
        MANIFEST_PATH: json.dumps(manifest, indent=1) + "\n",
        DECKS_MD_PATH: render_markdown(manifest),
    }
    stale = [p for p, body in files.items() if not p.exists() or p.read_text() != body]
    if args.check:
        for p in stale:
            print(f"stale: {p}", file=sys.stderr)
        return 1 if stale else 0
    for p, body in files.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
