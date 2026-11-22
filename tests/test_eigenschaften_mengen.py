"""Eigenschaften, die für jede Eingabe gelten müssen (ORACLE-PLAYBOOK §5): feste Zufalls-Seeds, 20 bis 50 Zufallsinstanzen je Eigenschaft.

Nicht geprüft wird die Eindeutigkeit der gierigen Menge: bei Gleichständen kann sich die Menge ändern, nur ihr Wert nicht."""

import numpy as np

import hva_algorithm as A
import hva_arms as AR
import hva_experiments as E
import hva_rules as R
import hva_scenario as S
import hva_sets as M
import hva_variation as V
from hva_rng import SplitMix64


def _table(rng, p, k):
    return np.array([[3 + rng.below(6) for _ in range(k)] for _ in range(p)], dtype=np.int64)


def test_gleicher_zufalls_seed_gleiche_ausgabe():
    train = S.training_set("stark", 2, K=5)
    for arm in AR.ARM_ORDER:
        a = AR.run_arm(arm, train, 77, (150,))
        b = AR.run_arm(arm, train, 77, (150,))
        assert a.fits == b.fits and list(a.pool) == list(b.pool)
    assert M.random_sets(30, 5, SplitMix64(3), 4) == M.random_sets(30, 5, SplitMix64(3), 4)
    x = E.variation_experiment("gl_20_70", "frisch", 1, 40, S.test_set("gl_20_70", K=20))
    y = E.variation_experiment("gl_20_70", "frisch", 1, 40, S.test_set("gl_20_70", K=20))
    assert x == y


def test_umnummerieren_der_instanzen_aendert_den_wert_der_menge_nicht():
    rng = SplitMix64(21)
    for _ in range(40):
        p, k = 4 + rng.below(6), 4 + rng.below(6)
        table = _table(rng, p, k)
        perm = list(range(k))
        rng.shuffle(perm)
        fits = table.mean(axis=1)
        members = M.random_set(p, 3, rng)
        assert M.set_cost(table, members) == M.set_cost(table[:, perm], members)
        # der Wert des ersten gierigen Schritts hängt nicht an der Reihenfolge der Instanzen
        g1, g2 = M.greedy_set(table, fits, 1), M.greedy_set(table[:, perm], fits, 1)
        assert M.set_cost(table, g1) == M.set_cost(table[:, perm], g2)
        assert M.set_cost(table, M.greedy_set(table, fits, 3)) == M.set_cost(table[:, perm], M.greedy_set(table[:, perm], fits, 3))


def test_orakelwahl_ist_nie_schlechter_als_ein_mitglied_und_mehr_mitglieder_schaden_nie():
    rng = SplitMix64(22)
    for _ in range(40):
        table = _table(rng, 6, 7)
        order = M.random_set(6, 6, rng)
        prev = None
        for size in range(1, 7):
            cost = M.set_cost(table, order[:size])
            assert cost <= min(int(table[j].sum()) for j in order[:size])
            assert (M.oracle_choice(table[order[:size]]) <= table[order[:size]]).all()
            assert prev is None or cost <= prev
            prev = cost


def test_feste_und_merkmalswahl_liegen_zwischen_orakel_und_schlechtestem_mitglied():
    rng = SplitMix64(23)
    for _ in range(30):
        train = np.array([[1 + rng.below(80) for _ in range(12)] for _ in range(8)], dtype=np.int64)
        test = np.array([[1 + rng.below(80) for _ in range(12)] for _ in range(10)], dtype=np.int64)
        mtr, mte = _table(rng, 3, 8), _table(rng, 3, 10)
        o = M.oracle_choice(mte)
        for chosen in (M.feature_choice(train, mtr, test, mte), M.fixed_choice(mtr, mte)):
            assert (o <= chosen).all() and (chosen <= mte.max(axis=0)).all()


def test_portfolio_mit_orakelwahl_ist_nie_schlechter_als_best_fit():
    for fam in S.TRAIN_FAMILIES:
        sizes = S.test_set(fam, K=40)
        bins = np.array([A.run_known(sizes, r) for r in M.PORTFOLIO])
        assert (M.oracle_choice(bins) <= A.run_known(sizes, "BF")).all()
    assert M.PORTFOLIO[0] == "BF" and set(M.PORTFOLIO) == {"BF", "FF", "WF"}


def test_skalieren_von_groessen_und_kapazitaet_aendert_die_packung_bekannter_regeln_nicht():
    rng = SplitMix64(24)
    for _ in range(30):
        sizes = np.array([[5 + rng.below(60) for _ in range(15)]], dtype=np.int64)
        for name in ("FF", "BF", "WF"):
            a = A.simulate(sizes, A.KNOWN[name], want_assign=True)[1]
            b = A.simulate(3 * sizes, A.KNOWN[name], want_assign=True, capacity=300)[1]
            assert (a == b).all()


def test_s1_mit_a_null_ist_best_fit_auf_jeder_familie():
    for fam in S.TRAIN_FAMILIES:
        sizes = S.test_set(fam, K=30)
        a = A.simulate(sizes, R.make_s_fn(V.BEST_FIT_IND), want_assign=True)[1]
        b = A.simulate(sizes, A.KNOWN["BF"], want_assign=True)[1]
        assert (a == b).all()


def test_reihenfolge_der_trainingsinstanzen_aendert_die_bewertung_nicht():
    """Die Fitness einer Regel ist ein Mittel über die Instanzen: umsortierte Instanzen geben dieselben Behälterzahlen je Instanz."""
    train = S.training_set("gl_20_70", 0, K=10)
    perm = [3, 0, 7, 1, 9, 2, 8, 4, 6, 5]
    rng = SplitMix64(25)
    for _ in range(20):
        ind = R.s_sample(rng)
        a = A.simulate(train, R.make_s_fn(ind))[0]
        b = A.simulate(train[perm], R.make_s_fn(ind))[0]
        assert (a[perm] == b).all()


def test_kinder_folge_ist_folgentreu():
    """Die ersten 100 von 300 Kindern sind die 100 Kinder des kürzeren Laufs: der Regler »Kinder« verlängert nur die Folge."""
    test = S.test_set("gl_20_70", K=20)
    a = E.variation_experiment("gl_20_70", "bf", 0, 60, test)
    b = E.variation_experiment("gl_20_70", "bf", 0, 120, test)
    assert sum(b["counts"].values()) == 120 and sum(a["counts"].values()) == 60
    assert all(a["counts"][c] <= b["counts"][c] for c in V.CLASSES)
    assert sum(t["n"] for t in b["table"].values()) == 120 and a["points"] == b["points"][: len(a["points"])]
