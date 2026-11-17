"""Every inventoried RF1 deck reads into ECPT records.

Parity: the ECPT words built from ``inp/<deck>.inp`` are compared with those built
from the COSMIC listing's sorted bulk data echo (``demoout/<deck>.out``).
"""

import numpy as np
import pytest

from nastran95py.bulk import BulkModel, build_ecpts, read_ecpts
from nastran95py.elements import CARD_LAYOUTS, LAYOUTS
from nastran95py.reference.cards import parse_bulk, read_deck
from nastran95py.reference.inventory import RF1_DECKS, build_deck, repo_root
from nastran95py.reference.listing import index_listing

from .conftest import record_parity

TEMPERATURE_SETS = {"d01032a": 1, "d01033a": 1, "d01034a": 1, "d01161a": 3000}


def _inp(deck_id: str) -> str:
    return (repo_root() / "inp" / f"{deck_id}.inp").read_text(errors="replace")


@pytest.fixture(scope="module", params=sorted(RF1_DECKS))
def deck(request):
    deck_id = request.param
    model, ecpts = read_ecpts(_inp(deck_id))
    return deck_id, model, ecpts


def test_deck_reads_into_ecpt_records(deck):
    deck_id, model, ecpts = deck
    entry = build_deck(deck_id, repo_root())
    expected = {CARD_LAYOUTS[c]: 0 for c in entry["elements"]}
    for card, info in entry["elements"].items():
        expected[CARD_LAYOUTS[card]] += info["count"]
    assert {k: len(v) for k, v in ecpts.items()} == expected
    assert len(model.grids) == entry["grids"]["count"]
    sil = model.sil
    for name, records in ecpts.items():
        layout = LAYOUTS[name]
        for rec in records:
            words = rec.words()
            assert len(words) == layout.length
            assert rec.data.fields["sil"] == tuple(sil[g] for g in rec.grid_ids)
            assert all(isinstance(w, int) for w in rec.words(raw=True))
            back = layout.parse(rec.words(raw=True), raw=True)
            assert back.fields["eid"] == rec.data.eid
            np.testing.assert_array_equal(
                [g.xyz for g in back.grids], np.float32([g.xyz for g in rec.data.grids])
            )


def test_element_temperatures(deck):
    deck_id, model, ecpts = deck
    temps = {r.data.temperature for recs in ecpts.values() for r in recs}
    tset = TEMPERATURE_SETS.get(deck_id)
    if tset is None:
        assert temps == {0.0}
        return
    for recs in ecpts.values():
        for r in recs:
            expected = np.mean([model.grid_temperature(tset, g) for g in r.grid_ids])
            assert r.data.temperature == pytest.approx(expected)
    if deck_id == "d01161a":
        assert temps == {80.0}


def test_ecpt_matches_listing_bulk_echo(deck):
    deck_id, _, ecpts = deck
    listing = index_listing(
        (repo_root() / "demoout" / f"{deck_id}.out").read_text(errors="replace")
    )
    cards, _ = parse_bulk(list(enumerate(listing["bulk_echo"], 1)))
    echo = build_ecpts(BulkModel.from_cards(cards), TEMPERATURE_SETS.get(deck_id))
    assert echo.keys() == ecpts.keys()
    a, b = [], []
    for name in ecpts:
        assert [r.data.eid for r in echo[name]] == [r.data.eid for r in ecpts[name]]
        for x, y in zip(ecpts[name], echo[name], strict=True):
            a.extend(x.words())
            b.extend(y.words())
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    err = np.abs(a - b)
    rel = err / np.maximum(np.abs(b), np.finfo(float).tiny)
    record_parity(f"ECPT inp vs listing echo {deck_id}", a.size, err.max(), rel.max())
    np.testing.assert_array_equal(a, b)


def test_decks_use_only_supported_or_known_cards():
    known_unused = {"PARAM", "CNGRNT", "MATT1", "TABLEM1", "PLIMIT", "POPT"}
    for deck_id in RF1_DECKS:
        model = BulkModel.from_deck(read_deck(_inp(deck_id)))
        assert set(model.ignored) <= known_unused, deck_id
