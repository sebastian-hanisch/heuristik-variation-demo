"""Das Handbeispiel als Tests: jede Zahl steht hier als Handrechnung, geprüft wird der Code der Demo (Simulator, Mengen, Wahlarten).

Teil A: eine Instanz, drei Regeln. Teil B: eine Variation (S1 mit a = 0 und a = 0,10). Teil C: vier Regeln, vier Trainings- und zwei
Testinstanzen. Teil D: Zählung. Die Prüfungen aus der Planung (dort mit eigenen Rechenwegen) laufen hier gegen `hva_*`; die direkten
Regeln und die Bruchrechnung kommen aus `tests/referenz.py`."""

import itertools
import random
from fractions import Fraction as Fr

import numpy as np

import hva_algorithm as A
import hva_experiments as E
import hva_rules as R
import hva_sets as M
import hva_variation as V
import referenz as ref

A_INST = [40, 75, 70, 15, 15, 25, 30, 25]
RULES = ("BF", "FF", "WF", "S1")


def _bins(items, rule):
    fn = R.make_s_fn(V.s1(0.10)) if rule == "S1" else A.KNOWN[rule]
    b, asg = A.simulate(np.array([items], dtype=np.int64), fn, want_assign=True)
    return int(b[0]), [int(x) for x in asg[0]]


# --- Teil A ----------------------------------------------------------------------------------------------------------------------------
def test_teil_a_drei_regeln_auf_einer_instanz():
    # First Fit : 40->B1(60) 75->B2(25) 70->B3(30) 15->B1(45) 15->B1(30) 25->B1(5) 30->B3(0) 25->B2(0)                 = 3 Behälter
    # Best Fit  : 40->B1 75->B2 70->B3 15->B2 (Lücke 10 < 15 < 45) 15->B3 25->B1 30->B1 25->B4                          = 4
    # Worst Fit : 40->B1 75->B2 70->B3 15->B1 15->B1 25->B1 (B1 und B3 je 30, B1 zuerst) 30->B3 25->B2                  = 3
    assert _bins(A_INST, "FF") == (3, [0, 1, 2, 0, 0, 0, 2, 1])
    assert _bins(A_INST, "BF") == (4, [0, 1, 2, 1, 2, 0, 0, 3])
    assert _bins(A_INST, "WF") == (3, [0, 1, 2, 0, 0, 0, 2, 1])
    assert sum(A_INST) == 295 and int(A.lower_bound(A_INST)[0]) == 3 == -(-295 // 100)
    assert A.opt_dp(A_INST) == 3 and ref.opt_enum(A_INST) == 3
    assert [ref.direct_pack(A_INST, r)[0] for r in ("FF", "BF", "WF")] == [3, 4, 3]


# --- Teil B ----------------------------------------------------------------------------------------------------------------------------
def test_teil_b_s1_mit_a_null_ist_best_fit_auf_300_instanzen():
    rnd = random.Random(8)
    same = 0
    for _ in range(300):
        it = [rnd.randrange(10, 80, 5) for _ in range(rnd.randrange(5, 11))]
        arr = np.array([it], dtype=np.int64)
        a = A.simulate(arr, R.make_s_fn(V.BEST_FIT_IND), want_assign=True)[1][0].tolist()
        b = ref.direct_pack(it, "BF")[1]
        same += a == b
    assert same == 300


def test_teil_b_eine_variation_aendert_die_packung_ueber_einen_gleichstand(hand_test):
    te1 = hand_test[0].tolist()
    # Best Fit 55->B1(45) 65->B2(35) 70->B3(30) 20->B3(10) 30->B2(5) 65->B4(35) 30->B4(5) 40->B1(5)            = 4
    # S1(0,10): gleich bis zum 5. Gegenstand 30: Lücken B1 15 und B2 5, Abstand zu 10 je 5: Gleichstand, B1; dann 65->B4,
    #           30->B2 (Abstand 5, B2 vor B4), 40 passt nirgends -> B5                                           = 5
    assert _bins(te1, "BF") == (4, [0, 1, 2, 2, 1, 3, 3, 0])
    assert _bins(te1, "S1") == (5, [0, 1, 2, 2, 0, 3, 1, 4])
    assert ref.pack_score_exact(te1, lambda s, r, g, p, e: -abs(g - Fr(1, 10))) == (5, [0, 1, 2, 2, 0, 3, 1, 4])
    assert ref.direct_pr(te1, 10) == (5, [0, 1, 2, 2, 0, 3, 1, 4])


# --- Teil C ----------------------------------------------------------------------------------------------------------------------------
def test_teil_c_tabelle_der_behaelterzahlen(hand_train, hand_test, hand_matrix, hand_matrix_test):
    h = E.hand_example()
    assert h["train_bins"].tolist() == hand_matrix.tolist() == [[4, 3, 4, 4], [4, 3, 5, 4], [5, 4, 5, 3], [4, 3, 4, 4]]
    assert h["test_bins"].tolist() == hand_matrix_test.tolist() == [[4, 4], [5, 4], [5, 3], [5, 4]]
    assert h["train_sums"] == [4 + 3 + 4 + 4, 4 + 3 + 5 + 4, 5 + 4 + 5 + 3, 4 + 3 + 4 + 4] == [15, 16, 17, 15]
    assert h["test_sums"] == [8, 9, 8, 9]
    assert h["lb_train"] == h["opt_train"] == [4, 3, 4, 3] and h["lb_test"] == h["opt_test"] == [4, 3]
    for i, items in enumerate(hand_train.tolist() + hand_test.tolist()):
        for j, rule in enumerate(("BF", "FF", "WF")):
            want = (hand_matrix[j].tolist() + hand_matrix_test[j].tolist())[i]
            assert ref.direct_pack(items, rule)[0] == want
        assert ref.direct_pr(items, 10)[0] == (hand_matrix[3].tolist() + hand_matrix_test[3].tolist())[i]
        assert ref.opt_enum(items) == A.opt_dp(items)
    assert tuple(E.HAND_TEST[0]) == tuple(hand_test[0]) and E.HAND_TRAIN == tuple(tuple(r) for r in hand_train.tolist())


def test_teil_c_greedy_und_topk(hand_matrix):
    sums = hand_matrix.sum(axis=1).astype(float)
    # erste Wahl: BF (15, Gleichstand mit S1, kleinerer Index); zweite: min(4,5)+min(3,4)+min(4,5)+min(4,3) = 14 mit WF
    assert M.greedy_set(hand_matrix, sums, 2) == [0, 2]
    assert M.set_cost(hand_matrix, [0]) == 15 and M.set_cost(hand_matrix, [0, 1]) == 4 + 3 + 4 + 4 == 15
    assert M.set_cost(hand_matrix, [0, 2]) == 4 + 3 + 4 + 3 == 14 and M.set_cost(hand_matrix, [0, 3]) == 15
    assert M.topk_set(sums, 2) == [0, 3] and M.set_cost(hand_matrix, [0, 3]) == 15       # Güte ohne Vielfalt: kein Gewinn
    pairs = {p: M.set_cost(hand_matrix, list(p)) for p in itertools.combinations(range(4), 2)}
    assert min(pairs.values()) == 14 and sorted(p for p, v in pairs.items() if v == 14) == [(0, 2), (2, 3)]
    h = E.hand_example()
    assert (h["greedy"], h["greedy_steps"], h["second_costs"], h["topk"], h["topk_cost"]) == ([0, 2], [15, 14], [15, 15, 14, 15], [0, 3], 15)


def test_teil_c_einsatz_auf_den_testinstanzen(hand_train, hand_test, hand_matrix, hand_matrix_test):
    g_tr, g_te = hand_matrix[[0, 2]], hand_matrix_test[[0, 2]]
    assert M.oracle_choice(g_te).tolist() == [4, 3] and int(M.oracle_choice(g_te).sum()) == 7          # 7 gegen 8 von Best Fit
    assert M.fixed_member(g_tr) == 0 and int(M.fixed_choice(g_tr, g_te).sum()) == 8
    assert int(M.oracle_choice(hand_matrix_test[[0, 3]]).sum()) == 8                                     # topk {BF, S1}: kein Gewinn
    assert int(M.oracle_choice(hand_matrix_test[[1, 0, 2]]).sum()) == 7                                  # Portfolio, 3 Läufe je Instanz
    # Merkmalswahl: Mittel der ersten drei Größen; Training 125/3, 80/3, 125/3, 135/3; Etiketten BF, BF, BF, WF
    assert M.labels(g_tr).tolist() == [0, 0, 0, 1]
    assert (M.first_features(hand_train, 3, False)[:, 0] * 3).round().tolist() == [125, 80, 125, 135]
    assert (M.first_features(hand_test, 3, False)[:, 0] * 3).round().tolist() == [190, 110]
    # Te1: nächster Nachbar Tr4 (Abstand 55/3) -> WF, 5 Behälter; Te2: Tr1 (Abstand 15/3, Gleichstand mit Tr3, kleinerer Index) -> BF, 4
    assert M.nearest_neighbor(M.first_features(hand_train, 3, False), M.first_features(hand_test, 3, False)).tolist() == [3, 0]
    assert M.feature_pick(hand_train, g_tr, hand_test, 3, False).tolist() == [1, 0]
    assert M.feature_choice(hand_train, g_tr, hand_test, g_te, 3, False).tolist() == [5, 4]
    h = E.hand_example()
    assert (h["oracle"], h["fixed"], h["feature"], h["bf"], h["topk_oracle"], h["portfolio"]) == (7, 8, 9, 8, 8, 7)
    assert h["oracle"] == sum(h["opt_test"])         # die Orakelwahl trifft hier auf beiden Instanzen das Optimum


def test_teil_c_merkmalswahl_in_bruechen(hand_train, hand_test):
    ftr = [Fr(int(sum(x[:3])), 3) for x in hand_train.tolist()]
    assert ftr == [Fr(125, 3), Fr(80, 3), Fr(125, 3), Fr(135, 3)]
    picked = []
    for x in hand_test.tolist():
        f = Fr(int(sum(x[:3])), 3)
        d = [abs(f - t) for t in ftr]
        picked.append(min(range(4), key=lambda t: (d[t], t)))
    assert picked == [3, 0]


def test_teil_c_eigenschaften_der_zielfunktion(hand_matrix):
    subsets = [list(c) for k in range(1, 5) for c in itertools.combinations(range(4), k)]
    assert len(subsets) == 15
    sums = hand_matrix.sum(axis=1)
    n_sub = 0
    for s in subsets:
        assert M.set_cost(hand_matrix, s) <= min(int(sums[r]) for r in s)                    # nie schlechter als jedes Mitglied
        for r in range(4):
            assert M.set_cost(hand_matrix, sorted(set(s) | {r})) <= M.set_cost(hand_matrix, s)      # Monotonie
    for a in subsets:
        for b in subsets:
            if set(a) <= set(b):
                for r in set(range(4)) - set(b):
                    ga = M.set_cost(hand_matrix, a) - M.set_cost(hand_matrix, a + [r])
                    gb = M.set_cost(hand_matrix, b) - M.set_cost(hand_matrix, b + [r])
                    assert ga >= gb                                                             # abnehmender Zusatznutzen
                    n_sub += 1
    assert n_sub == 76


# --- Teil D ----------------------------------------------------------------------------------------------------------------------------
def test_teil_d_zaehlung_in_zwei_einheiten():
    h = E.hand_example()
    n_train, n_test = len(E.HAND_TRAIN), len(E.HAND_TEST)
    assert h["runs"]["pool"] * n_train == 4 * 4 == 16                 # 4 Auswertungen = 16 Platzierungsläufe im Training
    assert h["runs"]["orakel"] * n_test == 2 * 2 == 4                 # Einsatz mit Orakelwahl
    assert h["runs"]["merkmal"] * n_test == 2 and h["runs"]["fest"] * n_test == 2
    assert h["runs"]["portfolio"] == 3
    assert M.runs_per_instance("orakel", 5) == 5 and M.runs_per_instance("merkmal", 5) == 1 and M.runs_per_instance("fest", 3) == 1
