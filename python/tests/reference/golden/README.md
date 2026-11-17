# Golden reference values

One JSON file per deck in [`../manifest.json`](../manifest.json), parsed from the
COSMIC listing `demoout/<deck>.out` by `nastran95py.reference.golden`. Do not
edit by hand; regenerate with:

```bash
python -m nastran95py.reference.golden            # write
python -m nastran95py.reference.golden --check    # fail if stale
python -m nastran95py.reference.golden --report   # parity table
```

## Layout

- `tables`: one entry per result table, keyed `<kind>/<element|grid>/sc<subcase>/occ<n>`
  (plus `/scan-<field>-<set>` for SCAN-filtered prints). `kind` is `displacement`,
  `spcforce`, `oload`, `force` or `stress`. `columns` starts with `id`; each row is
  `[id, values...]` exactly as printed, with `null` for blank fields (unprinted
  safety margins). SORT2 listings are regrouped into one table per subcase
  (`sort` records the original). `occurrence` counts repeated prints, e.g. the
  fully stressed design iterations of d01161a. Grid tables carry `point_types`.
- `matrices`: `MATPRN` prints (t01241a KGGX) as 1-based `[row, col, value]`.
- `tolerance`: the comparison rule below.

Load as NumPy with `nastran95py.reference.golden.tables(load_golden(deck))` and
compare a port's output with `compare(table, actual)`.

## Tolerance

The listings print 5-7 significant figures from a single-precision run. The
default check is `|actual - golden| <= atol + rtol * |golden|` with

- `rtol = 1e-4`;
- `atol = 1e-4 * max|golden|` over that column of that table (the whole table's
  max when the column is all zero, e.g. rotations or rod torque). This floor
  absorbs round-off printed for values that are zero in exact arithmetic
  (`3.456080E-10` SPC forces next to `5.06E+03`) and the 4-decimal principal angles;
- `rtol = 5e-2` for safety margins, which are printed to 2 significant figures.

Recomputing MAJOR/MINOR/MAX SHEAR/ANGLE from the printed NORMAL-X/NORMAL-Y/SHEAR-XY
(Mohr's circle) passes this rule for every membrane table, while a plain relative
1e-4 test does not (worst 2.7e-4, small angles), which is why the floor is needed.
