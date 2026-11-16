from nastran95py.reference.listing import index_listing, spaced_title

PAGE = "1     DEMO" + " " * 35 + "/    95 SUN SOLARIS NASTRAN    / MAY 17, 95 / PAGE    {n}"

LISTING = "\n".join(
    [
        PAGE.format(n=1),
        "      SUB",
        "0     LABEL                                                 SUBCASE 1",
        "                                             D I S P L A C E M E N T   V E C T O R",
        "      POINT ID.   TYPE          T1             T2",
        "             1      G      0.0            0.0",
        "             2      G      1.0E-3         2.0E-3",
        PAGE.format(n=2),
        "      SUB",
        "0     LABEL                                                 SUBCASE 1",
        " " * 21
        + "S T R E S S E S   I N   T R I A N G U L A R   M E M B R A N E S      ( C T R M E M )",
        "  ELEMENT  NORMAL-X",
        "       10        8.375298E+02",
        "0      MATRIX KGGX     (GINO NAME 101 ) IS A D.P.REAL      2 COLUMN X      2 ROW"
        " SYMMETRC MATRIX.",
        "0COLUMN      1      ROWS      1 THRU      2     -----",
        "",
        "   2.0D+00 -2.0D+00",
        "0COLUMN      2      ROWS      2 THRU      2     -----",
        "",
        "   2.0D+00",
        "0THE NUMBER OF NON-ZERO TERMS IN THE DENSEST COLUMN =       2",
        "                                        * * * END OF JOB * * *",
        PAGE.format(n=3),
        "      SUB",
        "0     LABEL",
        "                                             D I S P L A C E M E N T   V E C T O R",
        "             9      G      0.0",
    ]
)


def test_spaced_title():
    assert spaced_title("0     D I S P L A C E M E N T   V E C T O R") == "DISPLACEMENT VECTOR"
    assert spaced_title("      POINT ID.   TYPE") is None


def test_index_listing_sections_and_matrix():
    idx = index_listing(LISTING)
    disp, stress = idx["sections"]
    assert disp["kind"] == "displacement" and disp["subcase"] == 1
    assert disp["ids"] == [1, 2] and disp["pages"] == [1, 1]
    assert stress["element_type"] == "CTRMEM" and stress["ids"] == [10]
    (kggx,) = idx["matrices"]
    assert (kggx["name"], kggx["rows"], kggx["cols"], kggx["form"]) == (
        "KGGX",
        2,
        2,
        "SYMMETRC",
    )
    assert kggx["entries"] == [[1, 1, 2.0], [2, 1, -2.0], [2, 2, 2.0]]
