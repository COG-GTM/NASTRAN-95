# nastran95py

Python port of the NASTRAN-95 element routines (RF1 element library). Each
kernel mirrors a Fortran routine in `../mis/` and is validated numerically
against the compiled Fortran.

**Scope.** This is a library of element-level routines that take the same
inputs as the Fortran (MPT material records, CSTM coordinate records, EST/ECPT
element records). It is *not* a replacement for the GINO/DMAP executive: the
bulk-data reader (`nastran95py.bulk`) only builds ECPT records for the RF1 decks;
there is no DMAP sequencing or matrix assembly here.

| Subpackage | Fortran counterpart |
|---|---|
| `nastran95py.materials` | `MAT` in `mis/premat.f` (MAT1 isotropic, MAT2 anisotropic; `inflag` 1-4) |
| `nastran95py.coords` | `TRANSS`/`TRANSD` (`mis/pretrs.f`, `mis/pretrd.f`), CSTM 14-word records |
| `nastran95py.elements` | EST/ECPT record contract (`ElementData`, `EcptLayout`; ROD, BAR layouts) |
| `nastran95py.io` | 32-bit INTEGER/REAL word reinterpretation, Fortran `D` exponents |

## Reference baseline

`nastran95py.reference` inventories the Rigid Format 1 demo decks in `../inp/`
and indexes their COSMIC listings in `../demoout/` (the Fortran reference, since
`../bin/nastrn.exe` is a SPARC build). Its deck reader covers only what the
inventory needs. Generated outputs:

- `tests/reference/manifest.json`: machine-readable manifest (elements,
  properties, materials, loads, subcases/output requests, listing section page and
  line ranges, printed matrices, consistency checks). Load it with
  `nastran95py.reference.inventory.load_manifest()`.
- `tests/reference/DECKS.md`: the same inventory as a readable catalogue.

```bash
python -m nastran95py.reference.inventory          # regenerate
python -m nastran95py.reference.inventory --check  # fail if stale (CI)
```

Golden values (displacements, SPC forces, element forces and stresses) parsed
from the listings live in `tests/reference/golden/` with their comparison
tolerance; see its [README](tests/reference/golden/README.md).

## Setup

```bash
cd python
python -m pip install -e ".[dev]"   # Python 3.11+, NumPy, pytest, ruff
```

## Tests and parity

```bash
ruff check . && ruff format --check .
python -m pytest                      # unit tests + Fortran parity
make -C tests/fortran_kernels         # optional: build the kernel harness by hand
```

Parity tests (`tests/test_fortran_parity.py`, marker `fortran`) build
`tests/fortran_kernels/` with gfortran: the unmodified `premat.f`, `pretrs.f`,
`pretrd.f`, `gmmats.f` and `gmmatd.f` linked against small GINO stubs and a
driver. pytest feeds the same inputs to Python and Fortran and prints a
"Fortran parity" table with the max absolute and relative error per kernel.
Without gfortran these tests skip; set `NASTRAN95PY_REQUIRE_FORTRAN=1` (as CI
does) to make that a failure. Run only parity with `python -m pytest -m fortran`.

The element-level harness lives in `tests/fortran_harness/`; CI runs
`make -C python/tests/fortran_harness` whenever that directory exists.
It links the unmodified RF1 element routines (EMG `rods/rodd`, `shears/sheard`,
`trmems/trmemd`, `qdmm1s/d`, `qdmm2s/d`; SMA1 `krod`, `ktrmem`, `kqdmem`,
`ktriqd`; SDR2 `srod1/2`, `spanl1/2`, `strme1`, `sqdme1`, `sqdm11/12/21/22`,
`stqme2`; the CTRMEM thermal load `trimem`) and dumps K, M, loads and
stress/force words for the hand-built cases in
`tests/fortran_harness/cases.py` (`tests/test_fortran_harness.py`);
`python -m tests.fortran_harness.runner` regenerates `golden/harness_golden.json`
(`python -m tests.fortran_harness.rod_cases` regenerates the coordinate-system
RODD cases in `golden/rod_golden.json`, used by `tests/test_rod.py`;
`python -m tests.fortran_harness.trmem_cases` regenerates the CTRMEM
`TRMEMD`/`TRIMEM` cases in `golden/trmem_golden.json`, used by `tests/test_trmem.py`).

`build/rodload` (same directory) links `mis/rod.f`, the EDTL thermal and
`DEFORM` load routine for ROD/CONROD/TUBE, with `basglb`/`mpyl` and the
SSGETD/FEDT/CSTM stubs in `loadstubs.f`; `python -m tests.fortran_harness.rod_load_cases`
regenerates `golden/rod_load_golden.json`, used by `tests/test_rod_loads.py`.

`build/qdmm1load` links `mis/qdmm1.f`, the CQDMEM1 EDTL thermal load, with
`basglb`/`premat`/`gmmats`; `python -m tests.fortran_harness.qdmem1_cases`
regenerates `golden/qdmem1_golden.json` (warped, anisotropic, coordinate-system
and diagonal-intersection-branch `QDMM1D`/`QDMM1S`/`SQDM11`/`SQDM12` cases plus
`QDMM1` loads), used by `tests/test_qdmem1.py` together with the CQDMEM1 stress
tables of d01012a, d01013a and d01032a (the last via a static solve, since its
printed displacements never cover a whole element). Note that `QDMM1D`/`QDMM1S`
take DEGRA from word 5 of `/CONDAS/` (4 pi^2), so their MAT2 material angle is
`THETA * 4 pi^2` radians; the port reproduces this (`qdmm1d_material_angle`).

`build/tshearld` links `mis/tshear.f`, the EDTL thermal load routine for CSHEAR
(pseudo-rods on the four panel edges), against the same `basglb`/`mpyl` and
`loadstubs.f` stubs (SSGETD also returns the grid temperatures `TI(2..5)`);
`python -m tests.fortran_harness.shear_load_cases` regenerates
`golden/shear_load_golden.json`, used by `tests/test_shear_loads.py`.

`python -m tests.fortran_harness.shear_cases` regenerates the SHEARD cases
(every flexibility branch, warped/skewed/CSTM panels, and the MESAGE 2026/2027
exits) in `golden/shear_golden.json`, used by `tests/test_shear.py`.

`python -m tests.fortran_harness.rod_stress_cases` reruns the `srod1`/`srod2`
rod stress recovery cases (coordinate systems, MAT1 limits, load temperature,
`DEFORM`) and regenerates `golden/rod_stress_golden.json`, used by
`tests/test_rod_stress.py` with the SROD cases of `harness_golden.json` and the
rod stress/force tables of the listings (d01161a, d01013a).

`python -m tests.fortran_harness.trmem_stress_cases` reruns the `strme1`/`stqme2`
CTRMEM stress recovery cases (coordinate systems, material angles, load
temperature, zero stress) and regenerates `golden/trmem_stress_golden.json`, used
by `tests/test_trmem_stress.py` with the STRME cases of `harness_golden.json` and
the "STRESSES IN TRIANGULAR MEMBRANES" tables of the listings (d01011a, d01012a,
d01013a, d01014a, d01161a).

`python -m tests.fortran_harness.shear_stress_cases` regenerates the SDR2
`SPANL1`/`SPANL2` cases (the same panels with general displacements, a material
with a shear allowable for the margin of safety) in
`golden/shear_stress_golden.json`, used by `tests/test_shear_stress.py`.

`python -m tests.fortran_harness.qdmem_cases` reruns the plain CQDMEM cases
(warped/skewed quadrilaterals, coordinate systems, material angles, NSM, load
temperature, zero stress) and regenerates `golden/qdmem_golden.json`: `kqdmem`
stiffness and `sqdme1`/`stqme2` stresses on `build/harness`, plus the
`qdmem` thermal load (four `trimem (1)` calls) and the `masstq (1)` lumped
mass on `build/qdmemx` (`qdmemx.f`, with the `basglb`/`loadstubs.f` CSTM stub).
`tests/test_qdmem.py` checks `nastran95py.elements.qdmem` against them, the
CQDMEM cases of `harness_golden.json`, and the "STRESSES IN QUADRILATERAL
MEMBRANES" tables of the listings (d01011a, d01161a).
