"""Eingefrorene Messreihen: Schema, Vollständigkeit und Neu-Rechnen eines Ausschnitts als Band.

Ganze Zahlen (Zählungen) werden exakt verglichen, Deltas mit absoluter Toleranz 1e-5 (die Datei hält sechs Nachkommastellen; nicht
bytegleich über Plattformen: die Summenreihenfolge von numpy kann sich in der letzten Stelle unterscheiden). Die Suchläufe sind
folgentreu, deshalb genügt ein Lauf mit 200 Auswertungen, um die Stufen 100 und 200 eines eingefrorenen Laufs nachzurechnen; die
Mengen brauchen den ganzen Pool und werden an einem vollen Lauf nachgerechnet. Der Block mit 10 000 Auswertungen wird nur auf sein
Schema geprüft (ein Lauf dauert zu lange für die Suite)."""

import pytest

import generate_precomputed as G
import hva_constants as C
import hva_evaluation as ev
import hva_experiments as E
import hva_scenario as S
import hva_variation as V

PRE = ev.load_precomputed()
APPROX = dict(rel=1e-9, abs=1e-5)


# --- Schema und Vollständigkeit -------------------------------------------------------------------------------------------------------
def _check_run(r, n_eval, stufen, testsets, k_list=("1", "3", "5")):
    assert [c["n"] for c in r["cps"]] == list(stufen)
    assert r["train"] == r["cps"][-1]["train"] and r["test"] == r["cps"][-1]["test"] == r["ref"]["same60"]
    assert sum(r["bilanz"]) == 200 and r["stat"]["n"] == n_eval and isinstance(r["rule"], str) and r["rule"]
    assert set(r["stat"]) == {"n", "distinct", "eq_bf", "eq_ff", "eq_any", "better", "dup"} and r["pool"] + r["stat"]["dup"] == n_eval
    trains = [c["train"] for c in r["cps"]]
    assert all(b <= a + 1e-9 for a, b in zip(trains, trains[1:]))              # mehr Budget verschlechtert den Trainingsbestwert nie
    assert set(r["ref"]) == set(testsets) and set(r["sets"]) == set(C.SET_TYPES_FROZEN) and set(r["mem"]) == {"greedy", "topk"}
    for typ in C.SET_TYPES_FROZEN:
        assert set(r["sets"][typ]) == set(k_list)
        for k in k_list:
            z = r["sets"][typ][k]
            assert set(z["T"]) == set(testsets) and all(len(v) == 3 for v in z["T"].values())
            for ts in testsets:
                o, m, f = z["T"][ts]
                assert o <= m + 1e-9 and o <= f + 1e-9                          # die Orakelwahl ist nie schlechter als eine Wahl mit einem Lauf
            assert z["size"] == min(int(k), r["pool"] + (1 if typ == "bfplus" else 0)) or r["pool"] < int(k)
    # geschachtelte Mengen: größere Menge, nie schlechtere Orakelwahl (greedy und topk); Training der gierigen Menge nie über der Einzelregel
    for typ in ("greedy", "topk"):
        for ts in testsets:
            vals = [r["sets"][typ][k]["T"][ts][0] for k in k_list]
            assert all(b <= a + 1e-9 for a, b in zip(vals, vals[1:]))
    assert r["sets"]["greedy"]["3"]["train"] <= r["sets"]["topk"]["3"]["train"] + 1e-9
    assert r["sets"]["bfplus"]["1"]["T"]["same60"] == [0.0, 0.0, 0.0]


def test_schema_jede_zelle_ist_vorhanden():
    assert set(PRE) == {"meta", "spielraum", "m1", "portfolio", "m3", "portfolio_gross", "m4", "m6", "m7"}
    meta = PRE["meta"]
    assert meta["stufen"] == list(C.CHECKPOINTS) and (meta["laeufe"], meta["laeufe_gross"], meta["k_test"], meta["test_seed"]) == (20, 10, 200, 0)
    assert meta["wahlarten"] == list(C.CHOICES) and meta["mengengroessen"] == [1, 3, 5] and meta["zufallsmengen_je_lauf"] == 5 and meta["kinder"] == 300
    for fam in C.FAMILIES:
        assert set(PRE["m1"][fam]) == set(C.ARMS) and set(PRE["spielraum"][fam]) == set(C.TESTSETS)
        for arm in C.ARMS:
            runs = PRE["m1"][fam][arm]
            assert [r["lauf"] for r in runs] == list(range(20))
            for r in runs:
                _check_run(r, 1000, C.CHECKPOINTS, C.TESTSETS)
                assert (r["knoten"] is None) == (arm.endswith("-S"))
        assert [r["lauf"] for r in PRE["portfolio"][fam]] == list(range(20))
        assert set(PRE["m4"][fam]) == set(C.PARENTS)
        for p in C.PARENTS:
            recs = PRE["m4"][fam][p]
            assert [r["lauf"] for r in recs] == list(range(20))
            for r in recs:
                assert r["n"] == 300 == sum(r["c"]) == sum(r["art"][k][0] for k in V.KINDS) and 0 <= r["bt"] <= r["c"][0]
                assert all(r["art"][k][0] == sum(r["art"][k][1:]) for k in V.KINDS)
        assert (PRE["m6"][fam]["n"], PRE["m6"][fam]["n_items"]) == (200, 10)
        assert (PRE["spielraum"][fam]["n120"]["n"], PRE["spielraum"][fam]["n120"]["n_items"], PRE["spielraum"][fam]["n30"]["n"]) == (100, 120, 200)
    assert set(PRE["m3"]) == set(C.BIG_FAMILIES) == set(PRE["portfolio_gross"])
    for fam in C.BIG_FAMILIES:
        assert set(PRE["m3"][fam]) == set(C.BIG_ARMS)
        for runs in PRE["m3"][fam].values():
            assert [r["lauf"] for r in runs] == list(range(10))
            for r in runs:
                _check_run(r, 1000, C.CHECKPOINTS, ("same60",))
    assert set(PRE["m7"]) == {G.M7_FAMILY} and set(PRE["m7"][G.M7_FAMILY]) == {G.M7_ARM}
    recs = PRE["m7"][G.M7_FAMILY][G.M7_ARM]
    assert len(recs) == G.M7_RUNS == 2
    for r in recs:
        _check_run(r, 10000, G.M7_CHECKPOINTS, ("same60",))


def test_keine_nullspalte_ohne_erklaerung():
    """Eine Spalte mit lauter Nullen hieße: die Variante läuft nie. Jeder Arm hat Läufe mit Test-Delta ungleich null und verschiedene
    Endregeln; ZW-S unterscheidet sich von EA-S und GP0-T von GP-T; topk unterscheidet sich von greedy; jede Wahlart von den anderen."""
    for fam in C.FAMILIES:
        for arm in C.ARMS:
            runs = PRE["m1"][fam][arm]
            assert any(abs(r["test"]) > 1e-9 for r in runs), (fam, arm)
            assert all(r["stat"]["eq_bf"] < r["stat"]["n"] for r in runs), (fam, arm)
            assert len({r["rule"] for r in runs}) > 1, (fam, arm)
            g = [r["sets"]["greedy"]["3"]["T"]["same60"] for r in runs]
            t = [r["sets"]["topk"]["3"]["T"]["same60"] for r in runs]
            assert g != t and any(x[0] != x[1] for x in g) and any(x[1] != x[2] for x in g) and any(x[0] != x[2] for x in g), (fam, arm)
            assert any(r["sets"]["zufall"]["3"]["T"]["same60"][0] != r["sets"]["bfplus"]["3"]["T"]["same60"][0] for r in runs)
        for a, b in (("ZW-S", "EA-S"), ("GP0-T", "GP-T"), ("Z-S", "ZW-S"), ("Z-T", "GP0-T")):
            assert [r["train"] for r in PRE["m1"][fam][a]] != [r["train"] for r in PRE["m1"][fam][b]], (fam, a, b)
        for p in C.PARENTS:
            assert sum(r["c"][0] + r["c"][3] for r in PRE["m4"][fam][p]) > 0
            assert all(sum(r["art"][k][0] for r in PRE["m4"][fam][p]) > 0 for k in V.KINDS)


# --- Neu-Rechnen als Band --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("fam,arm,lauf", [("gl_20_70", "GP0-T", 3), ("clu3", "ZW-S", 7), ("stark", "Z-T", 11)])
def test_m1_stufen_neu_gerechnet(fam, arm, lauf):
    frozen = PRE["m1"][fam][arm][lauf]
    s = E.search_run(fam, arm, lauf, budget=200)
    rec = E.run_record(fam, arm, lauf, {"same60": S.test_set(fam)}, budget=200, set_types=("greedy",), k_list=(1,), search=s)
    assert [c["n"] for c in rec["cps"]] == [100, 200]
    for new, old in zip(rec["cps"], frozen["cps"][:2]):
        assert new["train"] == pytest.approx(old["train"], **APPROX) and new["test"] == pytest.approx(old["test"], **APPROX)


def test_m1_voller_lauf_mit_mengen_neu_gerechnet():
    fam, arm, lauf = "gl_20_70", "EA-S", 5
    frozen = PRE["m1"][fam][arm][lauf]
    tests = E.test_sets(fam, names=("same60", "n30", "fremd1"))
    rec = E.run_record(fam, arm, lauf, tests)
    assert rec["stat"] == frozen["stat"] and rec["pool"] == frozen["pool"] and rec["rule"] == frozen["rule"] and rec["bilanz"] == frozen["bilanz"]
    assert rec["mem"] == frozen["mem"]
    assert rec["train"] == pytest.approx(frozen["train"], **APPROX) and rec["test"] == pytest.approx(frozen["test"], **APPROX)
    for typ in C.SET_TYPES_FROZEN:
        for k in ("1", "3", "5"):
            assert rec["sets"][typ][k]["train"] == pytest.approx(frozen["sets"][typ][k]["train"], **APPROX)
            assert rec["sets"][typ][k]["eq_bf"] == pytest.approx(frozen["sets"][typ][k]["eq_bf"], **APPROX)
            for ts in tests:
                assert rec["sets"][typ][k]["T"][ts] == pytest.approx(frozen["sets"][typ][k]["T"][ts], **APPROX), (typ, k, ts)
    for ts in tests:
        assert rec["ref"][ts] == pytest.approx(frozen["ref"][ts], **APPROX)


def test_m3_stufe_100_mit_100_trainingsinstanzen_neu_gerechnet():
    fam, arm, lauf = "stark", "EA-S", 2
    frozen = PRE["m3"][fam][arm][lauf]
    s = E.search_run(fam, arm, lauf, K=C.K_TRAIN_BIG, budget=100)
    rec = E.run_record(fam, arm, lauf, {"same60": S.test_set(fam)}, K=C.K_TRAIN_BIG, budget=100, set_types=("greedy",), k_list=(1,), search=s)
    assert rec["cps"][0]["train"] == pytest.approx(frozen["cps"][0]["train"], **APPROX) and rec["cps"][0]["test"] == pytest.approx(frozen["cps"][0]["test"], **APPROX)


@pytest.mark.parametrize("fam,parent,lauf", [("gl_20_70", "bf", 0), ("clu3", "frisch", 3), ("stark", "s1_20", 19)])
def test_m4_kinder_neu_gerechnet(fam, parent, lauf):
    new = E.variation_record(fam, parent, lauf, C.FROZEN_CHILDREN, S.test_set(fam))
    assert new == PRE["m4"][fam][parent][lauf]                    # nur Zählungen: exakt


def test_spielraum_portfolio_und_optimum_neu_gerechnet():
    fam = "clu3"
    tests = E.test_sets(fam, names=("same60", "n30"))
    for ts, sizes in tests.items():
        sp = E.spielraum(sizes)
        old = PRE["spielraum"][fam][ts]
        for key, val in sp.items():
            assert val == pytest.approx(old[key], **APPROX) if isinstance(val, float) else val == old[key], (ts, key)
    port = E.portfolio_record(fam, 4, tests)
    old = PRE["portfolio"][fam][4]
    assert port["train"] == pytest.approx(old["train"], **APPROX) and port["size"] == old["size"] == 3
    for ts in tests:
        assert port["T"][ts] == pytest.approx(old["T"][ts], **APPROX)
    opt = E.spielraum_opt(S.test_set("gl_20_70", n=G.N_SMALL, K=60))
    assert opt["n"] == 60 and 0 <= opt["bf_optimal"] <= 60 and opt["lb_unter_opt"] >= 0 and opt["bf_ueber_opt"] >= 0
    full = PRE["m6"]["gl_20_70"]
    assert full["bf_optimal"] <= 200 and full["lb_eq_opt"] <= 200 and full["bf_ueber_opt"] > 0 and full["lb_unter_opt"] > 0


def test_arme_der_kopierquelle_stimmen_mit_deren_eingefrorenen_laeufen_ueberein_falls_vorhanden():
    """Z-S, EA-S, Z-T und GP-T benutzen dieselben Zufalls-Seeds wie die Kopierquelle: deren eingefrorene Läufe müssen dieselben sein."""
    import json
    from pathlib import Path

    quelle = Path(__file__).resolve().parent.parent.parent / "platzierungsfunktion-demo" / "plf_precomputed.json"
    if not quelle.exists():
        pytest.skip("platzierungsfunktion-demo liegt hier nicht vor")
    other = json.loads(quelle.read_text(encoding="utf-8"))
    n = 0
    for fam in C.FAMILIES:
        for arm in ("Z-S", "EA-S", "Z-T", "GP-T"):
            for a, b in zip(PRE["m1"][fam][arm], other["m1"][fam][arm]):
                assert a["stat"] == b["stat"] and a["rule"] == b["rule"] and a["bilanz"] == b["bilanz"]
                assert a["train"] == pytest.approx(b["train"], **APPROX) and a["test"] == pytest.approx(b["test"], **APPROX)
                n += 1
    assert n == 3 * 4 * 20
