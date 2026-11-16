from nastran95py.reference.casecontrol import parse_case_control, parse_executive


def test_executive():
    ex = parse_executive(
        ["ID    A,B", "APP   DISPLACEMENT", "SOL   1,0", "TIME  5", "ALTER 58 $ PRINT", "CEND"]
    )
    assert ex["app"] == "DISPLACEMENT"
    assert ex["sol"] == "1,0"
    assert ex["rigid_format"] == 1
    assert ex["time"] == 5
    assert ex["alter"] is True


def test_case_control_defaults_sets_and_subcases():
    cc = parse_case_control(
        [
            "TITLE = WING",
            "LABEL = TOP",
            "   SET 1 = 11 THRU 13, 20,",
            "           30 THRU 31",
            "DISPLACEMENTS (SORT2) = 1",
            "  SPCF = ALL",
            "  ELSTRESS = ALL",
            "  SPC = 1",
            "TEMPERATURE(MATERIALS) = 3000",
            "SUBCASE 10",
            "  LOAD = 5",
            "SUBCASE 12",
            "  LABEL = SECOND",
            "  LOAD = 6",
            "  ELFORCE = 1",
            "  SCAN (STRESS, CROD, AXIAL) = 5, SET 1",
            "OUTPUT(PLOT)",
            "  SET 9 = ALL",
        ]
    )
    assert cc["title"] == "WING"
    assert cc["sets"] == {"1": [11, 12, 13, 20, 30, 31]}
    first, second = cc["subcases"]
    assert first["id"] == 10 and first["label"] == "TOP"
    assert first["selectors"]["LOAD"] == {"set": 5}
    assert first["selectors"]["TEMPERATURE"] == {"set": 3000, "type": "MATERIALS"}
    assert first["outputs"]["DISPLACEMENT"] == {"set": 1, "options": ["SORT2"]}
    assert first["outputs"]["SPCFORCE"]["set"] == "ALL"
    assert "FORCE" not in first["outputs"]
    assert second["label"] == "SECOND"
    assert second["outputs"]["FORCE"] == {"set": 1, "options": []}
    assert second["outputs"]["STRESS"]["set"] == "ALL"
    assert first["scan"] == []
    assert second["scan"] == [{"options": ["STRESS", "CROD", "AXIAL"], "value": "5, SET 1"}]


def test_case_control_without_subcase_is_one_subcase():
    cc = parse_case_control(["LOAD = 3", "DISP = ALL"])
    assert [s["id"] for s in cc["subcases"]] == [1]
    assert cc["subcases"][0]["selectors"]["LOAD"] == {"set": 3}
