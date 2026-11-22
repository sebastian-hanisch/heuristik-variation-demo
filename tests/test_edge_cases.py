"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): n = 1, n = 2, Gleichstände, inklusive Grenzen,
leere und entartete Eingaben, Nenner null."""

import numpy as np
import pytest

import hva_algorithm as A
import hva_arms as AR
import hva_evaluation as ev
import hva_experiments as E
import hva_rules as R
import hva_scenario as S
import hva_sets as M
import hva_variation as V
from hva_rng import SplitMix64


def test_ein_und_zwei_gegenstaende_und_ein_gegenstand_der_groesse_c():
    one = np.array([[37]], dtype=np.int64)
    two = np.array([[60, 40], [60, 41]], dtype=np.int64)
    full = np.array([[100, 100, 1]], dtype=np.int64)
    for fn in (A.score_ff, A.score_bf, A.score_wf, R.make_s_fn(V.s1(0.1))):
        assert A.simulate(one, fn)[0].tolist() == [1]
        assert A.simulate(two, fn)[0].tolist() == [1, 2]            # 60 + 40 passt genau (Grenze inklusive), 60 + 41 nicht
        assert A.simulate(full, fn)[0].tolist() == [3]
    assert A.lower_bound(two).tolist() == [1, 2] and A.opt_dp([60, 40]) == 1 and A.opt_dp([60, 41]) == 2 and A.opt_dp([]) == 0


def test_menge_der_groesse_eins_ist_die_einzelregel():
    table = np.array([[4, 5, 6], [5, 4, 6], [6, 6, 3]], dtype=np.int64)
    fits = table.mean(axis=1)
    g = M.greedy_set(table, fits, 1)
    assert g == M.topk_set(fits, 1) == [0]                          # Summen 15, 15, 15: Gleichstand, kleinster Index
    mt = table[g]
    assert M.oracle_choice(mt).tolist() == M.fixed_choice(mt, mt).tolist() == [4, 5, 6]
    assert M.runs_per_instance("orakel", 1) == 1


def test_pool_kleiner_als_k_und_alle_regeln_gleich():
    table = np.array([[4, 4, 4], [4, 4, 4]], dtype=np.int64)
    fits = table.mean(axis=1)
    assert M.greedy_set(table, fits, 5) == [0, 1] and M.topk_set(fits, 5) == [0, 1]
    assert sorted(M.random_set(2, 5, SplitMix64(1))) == [0, 1] and M.random_set(1, 3, SplitMix64(1)) == [0]
    assert M.set_cost(table, [0, 1]) == M.set_cost(table, [0]) == 12        # alle Regeln gleich: Gewinn 0, kein NaN
    assert M.labels(table).tolist() == [0, 0, 0] and M.fixed_member(table) == 0


def test_nur_eine_trainingsinstanz_und_gleichstand_beim_naechsten_nachbarn():
    train = np.array([[10, 20, 30, 40]], dtype=np.int64)
    test = np.array([[40, 30, 20, 10], [1, 1, 1, 1]], dtype=np.int64)
    nn = M.nearest_neighbor(M.first_features(train, 3), M.first_features(test, 3))
    assert nn.tolist() == [0, 0]                                    # Streuung der Merkmale 0: kein Fehler, Index 0
    same = np.array([[10, 20, 30], [10, 20, 30], [50, 60, 70]], dtype=np.int64)
    assert M.nearest_neighbor(M.first_features(same, 3), M.first_features(np.array([[10, 20, 30]]), 3)).tolist() == [0]      # zwei gleiche Nachbarn: der erste
    assert M.first_features(np.array([[5, 7]]), 10).tolist() == [[6.0, 1.0]]            # weniger als zehn Gegenstände: alle vorhandenen


def test_budget_100_pool_kleiner_als_budget_und_stufen():
    s = E.search_run("gl_20_70", "GP-T", 0, K=4, budget=100)
    assert s["stat"]["n"] == 100 and len(s["pool_inds"]) + s["stat"]["dup"] == 100 and [c[0] for c in s["cps"]] == [100]
    s2 = E.search_run("gl_20_70", "EA-S", 0, K=4, budget=250)
    assert [c[0] for c in s2["cps"]] == [100, 200, 250]


def test_kein_besseres_kind_gibt_keinen_nenner_null():
    kind, text = ev.verdict_variation(False, 100, 0, 60, 40, 0)
    assert kind == "warning" and "0 (0,0 %)" in text and "Kein Kind ist besser" in text
    assert ev.verdict_variation(False, 0, 0, 0, 0, 0) == ("info", "Keine Kinder gerechnet.")
    assert ev.pct(0, 0) == "–" and ev.pct(1, 4) == "25,0"
    r = E.variation_experiment("clu3", "bf", 0, 5, S.test_set("clu3", K=10), K=4)
    assert sum(r["counts"].values()) == 5 and r["better_test"] <= r["counts"]["besser"]


def test_urteil_grenzen_sind_inklusive_und_nullen_zaehlen_nicht():
    assert ev.urteil([-0.2] * 15 + [0.5] * 5)[0] == "besser"                    # Median genau -0,2 und genau 15 Läufe
    assert ev.urteil([-0.2] * 14 + [0.0] + [0.5] * 5)[0] == "gleich"           # nur 14 Läufe unter null
    assert ev.urteil([-0.19] * 20)[0] == "gleich"                               # Median knapp über der Schwelle
    assert ev.urteil([0.2] * 15 + [-0.5] * 5)[0] == "schlechter"
    lab, med, neg, pos = ev.urteil([0.0] * 20)
    assert (lab, med, neg, pos) == ("gleich", 0.0, 0, 0)
    assert ev.median([3, 1, 2]) == 2 and ev.median([4, 1, 3, 2]) == 2.5 and ev.median([7]) == 7


def test_fremde_familie_ist_nie_die_trainingsfamilie():
    import hva_constants as C

    for fam in C.FAMILIES:
        assert fam not in C.other_families(fam) and len(C.other_families(fam)) == 2
        assert C.FAMILY_LABELS[C.other_families(fam)[0]] in C.testset_label("fremd1", fam)
    tests = E.test_sets("clu3", names=("same60", "fremd1", "fremd2"))
    assert (tests["fremd1"] == S.test_set("gl_20_70")).all() and (tests["fremd2"] == S.test_set("stark")).all()
    assert not (tests["same60"] == tests["fremd1"]).all()


def test_seeds_haengen_nicht_an_der_zahl_der_trainingsinstanzen():
    assert AR.arm_seed("EA-S", "stark", 4) == AR.arm_seed("EA-S", "stark", 4)
    a = S.training_set("gl_20_70", 0, K=20)
    b = S.training_set("gl_20_70", 0, K=100)
    assert a.shape == (20, 60) and b.shape == (100, 60) and not (a == b[:20]).all()      # K steht im Zufalls-Seed der Trainingsmenge


def test_ungueltige_eingaben_werden_abgelehnt():
    with pytest.raises(ValueError):
        E.parent_ind("unbekannt", "gl_20_70", 0)
    with pytest.raises(ValueError):
        E.members_of({"zufall": []}, "unbekannt", 3)
    with pytest.raises(KeyError):
        AR.run_arm("11-W", S.training_set("gl_20_70", 0, K=2), 1, (10,))
