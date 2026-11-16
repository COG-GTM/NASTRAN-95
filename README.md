# NASTRAN-95

NASTRAN has been released under the  
[NASA Open Source Agreement version 1.3](https://github.com/nasa/NASTRAN-95/raw/master/NASA%20Open%20Source%20Agreement-NASTRAN%2095.doc).


NASTRAN is the NASA Structural Analysis System, a finite element analysis program (FEA) completed in the early 1970's. It was the first of its kind and opened the door to computer-aided engineering. Subsections of a design can be modeled and then larger groupings of these elements can again be modeled. NASTRAN can handle elastic stability analysis, complex eigenvalues for vibration and dynamic stability analysis, dynamic response for transient and steady state loads, and random excitation, and static response to concentrated and distributed loads, thermal expansion, and enforced deformations.

NOTE: There is no technical support available for this software.

---

## Building and running on Linux (gfortran)

This fork adds a gfortran build. The NASA sources are compiled as delivered apart from
seven small portability fixes (listed in the commit history: ETIME, LINK/RENAME
intrinsics, one Hollerith DATA statement and one A1 format read).

```bash
make -j"$(nproc)"                         # bin/nastran.exe, rigid formats staged in build/rf
scripts/run.sh d01013a d01101a            # runs inp/<deck>.inp, writes out/<deck>.out
scripts/compare_demoout.py d01013a d01101a  # compares out/ against NASA's demoout/
```

`scripts/run.sh` sets the environment the original `bin/nastran` launcher set (RFDIR,
DBMEM, OCMEM, FTnn scratch files) and runs each deck in its own scratch directory.
`scripts/compare_demoout.py` masks dates and CPU timings and compares numbers to one
unit in the last printed digit or 1e-4 relative, since the 1995 reference listings
were produced on a Sun SPARC.

All 132 demo decks in `inp/` run to completion. 61 match `demoout/` line for line after
masking; most of the others differ in round-off-sized numbers or in restart decks that
need the checkpoint of an earlier run. D01-01-1A (CQDMEM elements) gives a different
answer from the reference and is not used below.

## Python port: Rigid Format 1 statics for membrane and rod elements

`nastran95py/` ports the static analysis path (Rigid Format 1) for the element group of
the delta-wing demonstration problems D01-01-3A and D01-01-4A to Python 3 and NumPy:

| Element | NASTRAN-95 source ported |
|---|---|
| CROD, CONROD | `mis/rodd.f`, `mis/srod1.f`, `mis/srod2.f` |
| CSHEAR | `mis/sheard.f`, `mis/spanl1.f`, `mis/spanl2.f` |
| CTRMEM | `mis/trmemd.f`, `mis/strme1.f` |
| CQDMEM1 | `mis/qdmm1d.f`, `mis/sqdm11.f`, `mis/sqdm12.f` |
| CQDMEM2 | `mis/qdmm2d.f`, `mis/sqdm21.f`, `mis/sqdm22.f` |

`nastran95py/static.py` does what the executive's static DMAP sequence does for this
group (EMG/EMA assembly, GP4 constraints, SSG1 loads, SSG2/SSG3 solve, SDR1/SDR2
recovery). Python was chosen over modern Fortran or C++ because the point of the port
is a readable, testable reference for one analysis path; NumPy covers the dense linear
algebra a 48-grid model needs.

Parity is checked against the gfortran listings in `tests/golden/` (regenerate with
`scripts/make_golden.sh`). A value matches when it is within 1e-5 relative, one unit in
the last printed digit, or 1e-6 of the table's largest magnitude (round-off zeros).

```bash
python3 -m pytest tests
python3 -m nastran95py.parity_run tests/decks/d01013a_all.inp tests/golden/d01013a_all.out
```

`tests/decks/*_all.inp` are the NASA decks with every output request set to ALL, so
every grid and element can be compared.
