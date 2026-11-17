"""Drive the element-level Fortran harness (``build/harness``) and parse its output.

The harness links the unmodified RF1 element routines from ``mis/`` (EMG, SMA1
and SDR2 paths for CROD/CONROD, CSHEAR, CTRMEM, CQDMEM, CQDMEM1 and CQDMEM2).
Each case passes an EST/ECPT record word by word; the harness prints every
matrix the routine hands to ``EMGOUT``/``SMA1B`` and the SDR2 phase-1/phase-2
words, with enough digits to round-trip exactly.

``python -m tests.fortran_harness.runner`` rebuilds the harness, runs all
hand-built cases and rewrites ``golden/harness_golden.json``.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import cases as C

HERE = Path(__file__).parent
BINARY = HERE / "build" / "harness"
GOLDEN = HERE / "golden" / "harness_golden.json"


def build() -> Path:
    env = dict(os.environ)
    subprocess.run(["make", "-s", "-C", str(HERE)], check=True, env=env)
    return BINARY


@dataclass
class CaseResult:
    case: C.Case
    matrices: list[dict] = field(default_factory=list)
    parts: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    words: dict[str, np.ndarray] = field(default_factory=dict)
    nogo: bool = False

    def stiffness(self) -> np.ndarray:
        """Full symmetric element K: from EMGOUT (EMG) or assembled SMA1B partitions."""
        if self.case.kind == C.EMG:
            m = next(m for m in self.matrices if m["file"] == 1)
            return _square(m)
        sils = sorted({j for _, j in self.parts})
        n = 6 * len(sils)
        k = np.zeros((n, n))
        for (piv, j), ke in self.parts.items():
            r, c = sils.index(piv), sils.index(j)
            k[6 * r : 6 * r + 6, 6 * c : 6 * c + 6] = ke.reshape(6, 6)
        return k

    def mass(self) -> np.ndarray:
        return _square(next(m for m in self.matrices if m["file"] == 2))


def _square(m: dict) -> np.ndarray:
    """EMGOUT data as a full matrix (DICT(2): 1 = square column-wise, 2 = diagonal)."""
    n = m["ndof"]
    vals = np.asarray(m["values"])
    if m["dict2"] == 2:  # diagonal form
        return np.diag(vals)
    return vals.reshape(n, n).T


def real_word_as_int(x: float) -> int:
    """Recover an integer word the harness printed as its REAL bit pattern."""
    return struct.unpack("<i", struct.pack("<f", x))[0]


def stdin_for(cases: list[C.Case]) -> str:
    lines = [str(len(C.MAT1))]
    lines += [" ".join([str(mid)] + [repr(float(v)) for v in vals]) for mid, vals in C.MAT1]
    lines.append(str(len(C.MAT2)))
    lines += [" ".join([str(mid)] + [repr(float(v)) for v in vals]) for mid, vals in C.MAT2]
    lines.append("0")
    lines.append(str(len(cases)))
    for case in cases:
        lines.append(f"{case.kind} {case.routine} {len(case.words)}")
        lines.append(" ".join("1" if isinstance(w, int) else "0" for w in case.words))
        lines.append(" ".join(str(w) if isinstance(w, int) else repr(float(w)) for w in case.words))
        lines.append(" ".join(str(x) for x in case.extra))
    return "\n".join(lines) + "\n"


def parse(text: str, cases: list[C.Case]) -> list[CaseResult]:
    results: list[CaseResult] = []
    tokens = iter(text.splitlines())
    current: CaseResult | None = None
    pending: tuple[str, list[int]] | None = None
    buf: list[float] = []

    def flush():
        nonlocal pending, buf
        if pending is None or current is None:
            return
        tag, hdr = pending
        vals = np.array(buf, dtype=float)
        if tag == "MATRIX":
            current.matrices.append(
                {
                    "file": hdr[0],
                    "dict2": hdr[1],
                    "ndof": hdr[2],
                    "icode": hdr[3],
                    "lbuf": hdr[4],
                    "prec": hdr[5],
                    "values": vals,
                }
            )
        elif tag == "PART":
            current.parts[(hdr[0], hdr[1])] = vals
        else:
            current.words[tag] = vals
        pending, buf = None, []

    for line in tokens:
        head = line.split()
        if not head:
            continue
        tag = head[0]
        if tag == "CASE":
            flush()
            current = CaseResult(cases[int(head[1]) - 1])
            results.append(current)
        elif tag in {"MATRIX", "PART", "PHASE1", "STRESS", "FORCE"}:
            flush()
            pending = (tag, [int(x) for x in head[1:]])
        elif tag == "END":
            flush()
        elif tag == "NOGO":
            current.nogo = True
        else:
            try:
                buf.extend(float(x) for x in head)
            except ValueError:
                continue  # diagnostic lines printed by MESAGE/element routines
    flush()
    return results


def run(cases: list[C.Case] | None = None, binary: Path = BINARY) -> list[CaseResult]:
    cases = C.ALL_CASES if cases is None else cases
    out = subprocess.run(
        [str(binary)], input=stdin_for(cases), capture_output=True, text=True, check=True
    )
    results = parse(out.stdout, cases)
    if len(results) != len(cases):
        raise RuntimeError(f"harness produced {len(results)} of {len(cases)} cases:\n{out.stdout}")
    return results


def to_golden(results: list[CaseResult]) -> dict:
    data = {}
    for r in results:
        entry: dict = {"element": r.case.element, "routine": r.case.routine, "kind": r.case.kind}
        if r.matrices:
            entry["matrices"] = [
                {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m.items()}
                for m in r.matrices
            ]
        if r.parts:
            entry["parts"] = {f"{p}:{j}": v.tolist() for (p, j), v in r.parts.items()}
        if r.words:
            entry["words"] = {k: v.tolist() for k, v in r.words.items()}
        data[r.case.name] = entry
    return data


def main() -> int:
    build()
    golden = to_golden(run())
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(golden, indent=1) + "\n")
    print(f"wrote {len(golden)} cases to {GOLDEN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
