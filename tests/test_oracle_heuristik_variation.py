"""Orakel-Tests: jede Kernroutine gegen einen unabhängigen Rechenweg (ORACLE-PLAYBOOK §2, DEMO-PLAYBOOK §12).

(a) Packung: Stapel-Simulator gegen direkte Regeln und den skalaren Interpreter für Schablonen und Bäume.
(b) Exaktes Optimum: Teilmengen-DP gegen Aufzählung und ganzzahliges Programm; Schranken.
(c) Mengenzielfunktion: gegen naive Schleifen; greedy gegen die Aufzählung aller k-Mengen; Monotonie und abnehmender Zusatznutzen.
(d) Merkmalswahl: 1-nächster Nachbar gegen eine Aufzählung in Brüchen.
(e) Operator: gültige Kinder, Anteile der Operatorarten als Band, `mutate_with_kind` gleich `s_mutate`.
(f) Zählung und Folgentreue der sechs Arme.
(g) Angezeigte Kennzahlen: Bilanz, Delta, Kandidatenstatistik, Spielraum.
Jede Gruppe läuft in wenigen Sekunden; die große Fassung steht in `messungen/orakel_gross.py`."""

import itertools
from fractions import Fraction as Fr

import numpy as np
import pytest

import hva_algorithm as A
import hva_arms as AR
import hva_experiments as E
import hva_rules as R
import hva_scenario as S
import hva_sets as M
import hva_variation as V
import referenz as ref
from hva_rng import SplitMix64


def _instances(seed, count, n_lo=5, n_hi=14, lo=5, hi=75):
    rng = SplitMix64(seed)
    return [[lo + rng.below(hi - lo + 1) for _ in range(n_lo + rng.below(n_hi - n_lo + 1))] for _ in range(count)]


# --- (a) Packung -----------------------------------------------------------------------------------------------------------------------
def test_a_bekannte_regeln_gegen_direkte_regeln():
    for items in _instances(1, 60):
        arr = np.array([items], dtype=np.int64)
        for name in ("FF", "BF", "WF"):
            bins, asg = A.simulate(arr, A.KNOWN[name], want_assign=True)
            assert (int(bins[0]), asg[0].tolist()) == ref.direct_pack(items, name), (name, items)
        assert int(A.run_known(arr, "NF")[0]) == ref.direct_pack(items, "NF")[0]
        assert int(A.run_known(arr, "FFD")[0]) == ref.direct_ffd(items)


def test_a_schablonen_gegen_den_skalaren_interpreter():
    rng = SplitMix64(2)
    for items in _instances(3, 40):
        ind = R.s_sample(rng)
        for _ in range(rng.below(4)):
            ind = R.s_mutate(ind, rng)
        bins, asg = A.simulate(np.array([items], dtype=np.int64), R.make_s_fn(ind), want_assign=True)
        assert (int(bins[0]), asg[0].tolist()) == ref.scalar_simulate(items, ref.s_scalar(ind)), (ind, items)


def test_a_baeume_gegen_den_skalaren_interpreter():
    rng = SplitMix64(4)
    for items in _instances(5, 40):
        t = R.rand_tree_ramped(rng)
        bins, asg = A.simulate(np.array([items], dtype=np.int64), R.make_tree_fn(t), want_assign=True)
        assert (int(bins[0]), asg[0].tolist()) == ref.scalar_simulate(items, ref.tree_scalar(t)), (t, items)


def test_a_lueckenziel_gegen_die_ganzzahlige_direkte_regel():
    """S1(a) als Schablone gegen die direkte Regel in ganzen Zahlen: fängt Gleichstände, die Gleitkomma-Abstände brechen würden."""
    for items in _instances(6, 40, lo=1, hi=14):
        items = [5 * x for x in items]                      # Vielfache von 5: viele exakte Gleichstände
        for a in (0, 5, 10, 15, 20, 30):
            bins, asg = A.simulate(np.array([items], dtype=np.int64), R.make_s_fn(V.s1(a / 100)), want_assign=True)
            assert (int(bins[0]), asg[0].tolist()) == ref.direct_pr(items, a), (a, items)


# --- (b) Optimum und Schranken ---------------------------------------------------------------------------------------------------------
def test_b_optimum_gegen_aufzaehlung_und_schranken():
    for items in _instances(7, 40, n_lo=4, n_hi=8):
        opt = A.opt_dp(items)
        assert opt == ref.opt_enum(items)
        arr = np.array([items], dtype=np.int64)
        lb = int(A.lower_bound(arr)[0])
        ffd = int(A.run_known(arr, "FFD")[0])
        assert lb <= opt <= ffd
        for name in ("FF", "BF", "WF", "NF"):
            assert opt <= int(A.run_known(arr, name)[0])


def test_b_optimum_gegen_ganzzahliges_programm():
    pytest.importorskip("scipy")
    for items in _instances(8, 12, n_lo=8, n_hi=12):
        assert A.opt_dp(items) == ref.opt_milp(items), items


# --- (c) Mengen ------------------------------------------------------------------------------------------------------------------------
def _table(rng, p, k, lo=3, hi=8):
    return np.array([[lo + rng.below(hi - lo + 1) for _ in range(k)] for _ in range(p)], dtype=np.int64)


def _naive_cost(table, members):
    total = Fr(0)
    for i in range(len(table[0])):
        best = None
        for m in members:
            v = Fr(int(table[m][i]))
            if best is None or v < best:
                best = v
        total += best
    return total


def test_c_mengenzielfunktion_gegen_naive_schleifen():
    rng = SplitMix64(9)
    for _ in range(60):
        p, k = 2 + rng.below(9), 1 + rng.below(8)
        table = _table(rng, p, k)
        members = M.random_set(p, 1 + rng.below(p), rng)
        assert M.set_cost(table, members) == _naive_cost(table.tolist(), members)
        assert M.set_min(table, members).tolist() == [min(int(table[m][i]) for m in members) for i in range(k)]


def test_c_greedy_gegen_aufzaehlung_aller_mengen():
    """Die gierige Menge ist nie besser als das Optimum über alle k-Mengen; der erste Schritt ist optimal. Die Lücke wird gezählt."""
    rng = SplitMix64(10)
    worse = 0
    total = 0
    for _ in range(80):
        p, kk = 4 + rng.below(8), 5 + rng.below(6)
        table = _table(rng, p, kk)
        fits = table.mean(axis=1)
        for k in (1, 2, 3, 4):
            g = M.greedy_set(table, fits, k)
            assert len(g) == len(set(g)) == k
            best = min(M.set_cost(table, list(c)) for c in itertools.combinations(range(p), k))
            cost = M.set_cost(table, g)
            assert cost >= best
            if k == 1:
                assert cost == best
            worse += cost > best
            total += 1
    assert total == 320 and 0 < worse < total       # greedy verfehlt das Optimum in einem Teil der Fälle (Scratch-Lauf: siehe README)


def test_c_greedy_schritt_fuer_schritt_gegen_die_definition():
    rng = SplitMix64(11)
    for _ in range(40):
        p, kk = 3 + rng.below(8), 4 + rng.below(6)
        table = _table(rng, p, kk)
        fits = table.mean(axis=1)
        g = M.greedy_set(table, fits, 3)
        chosen = []
        for step in range(min(3, p)):
            cands = [(int(_naive_cost(table.tolist(), chosen + [j])), round(float(fits[j]), 12), j) for j in range(p) if j not in chosen]
            chosen.append(min(cands)[2])
        assert g == chosen


def test_c_monotonie_und_abnehmender_zusatznutzen_auf_zufallstabellen():
    rng = SplitMix64(12)
    for _ in range(12):
        table = _table(rng, 5, 6)
        subsets = [list(c) for k in range(1, 6) for c in itertools.combinations(range(5), k)]
        cost = {tuple(s): M.set_cost(table, s) for s in subsets}
        for a in subsets:
            for b in subsets:
                if set(a) <= set(b):
                    assert cost[tuple(b)] <= cost[tuple(a)]
                    for r in set(range(5)) - set(b):
                        ga = cost[tuple(a)] - cost[tuple(sorted(a + [r]))]
                        gb = cost[tuple(b)] - cost[tuple(sorted(b + [r]))]
                        assert ga >= gb >= 0


def test_c_topk_ist_die_sortierung_nach_fitness_mit_kleinstem_index():
    rng = SplitMix64(13)
    for _ in range(40):
        fits = [rng.below(6) / 4.0 for _ in range(3 + rng.below(9))]
        want = [j for _, j in sorted((f, j) for j, f in enumerate(fits))][:3]
        assert M.topk_set(np.array(fits), 3) == want


def test_c_zufallsmengen_sind_gleichverteilte_teilmengen():
    rng = SplitMix64(14)
    counts = {}
    n = 6000
    for _ in range(n):
        s = tuple(sorted(M.random_set(5, 2, rng)))
        assert len(set(s)) == 2
        counts[s] = counts.get(s, 0) + 1
    assert len(counts) == 10                                    # alle 10 Paare kommen vor
    se = (n * 0.1 * 0.9) ** 0.5
    assert all(abs(c - n / 10) < 4.5 * se for c in counts.values())


# --- (d) Merkmalswahl ------------------------------------------------------------------------------------------------------------------
def test_d_naechster_nachbar_gegen_aufzaehlung_in_bruechen():
    """Ein Merkmal (Summe der ersten m Größen): der nächste Nachbar hängt nicht an der Normierung; Gleichstand: kleinster Index."""
    rng = SplitMix64(15)
    for _ in range(60):
        m = 1 + rng.below(5)
        train = np.array([[10 + 5 * rng.below(8) for _ in range(6)] for _ in range(2 + rng.below(8))], dtype=np.int64)
        test = np.array([[10 + 5 * rng.below(8) for _ in range(6)] for _ in range(5)], dtype=np.int64)
        got = M.nearest_neighbor(M.first_features(train, m, False), M.first_features(test, m, False)).tolist()
        ftr = [Fr(int(sum(x[:m])), m) for x in train.tolist()]
        if len(set(ftr)) == 1:
            continue                                            # Streuung 0: alle Abstände gleich, Index 0
        want = []
        for x in test.tolist():
            f = Fr(int(sum(x[:m])), m)
            want.append(min(range(len(ftr)), key=lambda t: (abs(f - ftr[t]), t)))
        assert got == want


def test_d_zwei_merkmale_gegen_direkte_rechnung_ohne_gleichstaende():
    rng = SplitMix64(16)
    train = np.array([[1 + rng.below(90) for _ in range(12)] for _ in range(20)], dtype=np.int64)
    test = np.array([[1 + rng.below(90) for _ in range(12)] for _ in range(50)], dtype=np.int64)
    ft, fx = M.first_features(train), M.first_features(test)
    assert ft.shape == (20, 2) and ft[0, 0] == pytest.approx(sum(train[0, :10]) / 10)
    mean0 = sum(train[0, :10]) / 10
    assert ft[0, 1] == pytest.approx((sum((x - mean0) ** 2 for x in train[0, :10].tolist()) / 10) ** 0.5)
    mu = [sum(ft[:, c]) / 20 for c in range(2)]
    sd = [(sum((v - mu[c]) ** 2 for v in ft[:, c]) / 20) ** 0.5 for c in range(2)]
    want = []
    for x in fx:
        d = [sum(((x[c] - t[c]) / sd[c]) ** 2 for c in range(2)) for t in ft]
        srt = sorted(d)
        assert srt[1] - srt[0] > 1e-9                         # kein Gleichstand in dieser Stichprobe
        want.append(d.index(srt[0]))
    assert M.nearest_neighbor(ft, fx).tolist() == want


def test_d_merkmalswahl_setzt_etikett_und_lauf_zusammen():
    rng = SplitMix64(17)
    train = np.array([[1 + rng.below(90) for _ in range(12)] for _ in range(8)], dtype=np.int64)
    test = np.array([[1 + rng.below(90) for _ in range(12)] for _ in range(9)], dtype=np.int64)
    mtr = _table(rng, 3, 8)
    mte = _table(rng, 3, 9)
    nn = M.nearest_neighbor(M.first_features(train), M.first_features(test))
    lab = [min(range(3), key=lambda j: (int(mtr[j][i]), j)) for i in range(8)]
    assert M.labels(mtr).tolist() == lab
    assert M.feature_choice(train, mtr, test, mte).tolist() == [int(mte[lab[nn[i]]][i]) for i in range(9)]
    assert M.fixed_member(mtr) == min(range(3), key=lambda j: (int(mtr[j].sum()), j))


# --- (e) Operator ----------------------------------------------------------------------------------------------------------------------
def _valid_s(ind):
    ti, th, t2, th2, mu = ind
    ok = 0 <= ti < len(R.TEMPLATES) and all(lo <= v <= hi for v, (lo, hi) in zip(th, R.TEMPLATES[ti][3])) and len(th) == len(R.TEMPLATES[ti][3])
    if t2 >= 0:
        ok = ok and t2 < len(R.TEMPLATES) and all(lo <= v <= hi for v, (lo, hi) in zip(th2, R.TEMPLATES[t2][3])) and 0.0 <= mu <= 1.0
    else:
        ok = ok and th2 == () and mu == 0.0
    return ok


def test_e_kinder_sind_gueltig_und_gleich_s_mutate_und_die_anteile_stimmen():
    n = 20000
    a, b = SplitMix64(18), SplitMix64(18)
    ind = V.BEST_FIT_IND
    counts = {k: 0 for k in V.KINDS}
    changed = {k: 0 for k in V.KINDS}
    for _ in range(n):
        kind, child = V.mutate_with_kind(ind, a)
        assert child == R.s_mutate(ind, b) and a.state == b.state
        assert _valid_s(child)
        counts[kind] += 1
        changed[kind] += child != ind
        ind = child
    for kind, p in V.KIND_PROBS.items():
        se = (p * (1 - p) / n) ** 0.5
        assert abs(counts[kind] / n - p) < 4.5 * se, (kind, counts[kind])
        assert changed[kind] > 0                                # jede Operatorart erzeugt ein vom Elternteil verschiedenes Kind
    assert V.KIND_PROBS == {"parameter": 0.5, "vorlage": 0.2, "zweiter_term": pytest.approx(0.3)}


def test_e_baumoperatoren_halten_die_hoechsttiefe():
    rng = SplitMix64(19)
    pop = [R.rand_tree_ramped(rng) for _ in range(30)]
    for _ in range(600):
        a, b = pop[rng.below(30)], pop[rng.below(30)]
        child = R.tree_crossover(a, b, rng) if rng.random() < 0.8 else R.tree_mutate_subtree(a, rng)
        assert R.tree_depth(child) <= R.MAX_DEPTH
        pop[rng.below(30)] = child


def test_e_jede_vorlage_weicht_von_best_fit_ab():
    """Nullspaltenregel: jede Vorlage mit nicht trivialem Parameter packt mindestens eine Instanz anders als Best Fit."""
    sizes = S.test_set("gl_20_70", K=30)
    bf = A.run_known(sizes, "BF")
    for ind in ((0, (0.2,), -1, (), 0.0), (1, (0.05, 0.3), -1, (), 0.0), (2, (0.5, 0.2), -1, (), 0.0), (3, (1.5,), -1, (), 0.0), (4, (1.0,), -1, (), 0.0)):
        assert (A.simulate(sizes, R.make_s_fn(ind))[0] != bf).any(), ind
    # vier der fünf Vorlagen enthalten Best Fit als Randfall, die Kantenanpassung (S2) nicht
    for ind, same in (((0, (0.0,), -1, (), 0.0), True), ((2, (0.0, 0.2), -1, (), 0.0), True), ((3, (0.0,), -1, (), 0.0), True), ((4, (0.0,), -1, (), 0.0), True)):
        assert bool((A.simulate(sizes, R.make_s_fn(ind))[0] == bf).all()) is same
    assert (A.simulate(sizes, R.make_s_fn((1, (0.0, 0.3), -1, (), 0.0)))[0] != bf).any()


# --- (f) Zählung und Folgentreue -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("arm", AR.ARM_ORDER)
def test_f_zaehlung_gleich_budget_und_bestwert_ist_das_minimum(arm):
    train = S.training_set("gl_20_70", 0, K=6)
    for budget in (200, 400):
        bud = AR.run_arm(arm, train, AR.arm_seed(arm, "gl_20_70", 0), (200, budget) if budget > 200 else (200,))
        assert bud.n == budget == len(bud.fits) == len(bud.vecs)
        assert bud.at[200][0] == min(bud.fits[:200]) and bud.best[0] == min(bud.fits)
        assert len(bud.pool) + bud.n_dup == budget and len(bud.pool) == len(bud.cache)
        lb = A.lower_bound(train)
        for i in (0, 57, budget - 1):
            assert bud.fits[i] == pytest.approx(float(np.mean(bud.vecs[i] / lb)), rel=1e-12)
    short = AR.run_arm(arm, train, AR.arm_seed(arm, "gl_20_70", 0), (200,))
    assert short.fits == bud.fits[:200]                         # folgentreu: der kürzere Lauf ist der Anfang des längeren


def test_f_auswahl_der_besten_ist_wirklich_abschaltbar():
    """Nullspaltenregel: ZW-S weicht von EA-S ab und GP0-T von GP-T, bei gleichem Zufalls-Seed und gleichem Training."""
    import hva_search as H

    train = S.training_set("gl_20_70", 0, K=6)
    lb = A.lower_bound(train)
    for fn, space in ((H.run_ea_s, "S"), (H.run_gp, "T")):
        a = H.Budget(space, train, lb, (300,))
        fn(a, SplitMix64(5), True)
        b = H.Budget(space, train, lb, (300,))
        fn(b, SplitMix64(5), False)
        assert a.fits != b.fits and a.n == b.n == 300
    assert AR.arm_seed("ZW-S", "gl_20_70", 0) != AR.arm_seed("EA-S", "gl_20_70", 0)


def test_f_arme_der_kopierquelle_benutzen_deren_zufalls_seeds():
    import hva_search as H

    for arm in ("Z-S", "EA-S", "Z-T", "GP-T"):
        assert AR.arm_seed(arm, "clu3", 3) == S.seed_for(S.SEED_METHOD, H.METHOD_ORDER.index(arm), 1, 3)
    assert AR.arm_seed("ZW-S", "clu3", 3) == S.seed_for(21, 1, 1, 3) and AR.arm_seed("GP0-T", "stark", 0) == S.seed_for(21, 4, 2, 0)
    assert len({AR.arm_seed(a, f, lauf) for a in AR.ARM_ORDER for f in S.TRAIN_FAMILIES for lauf in range(20)}) == 6 * 3 * 20


def test_f_test_und_trainingsmengen_ueberlappen_nicht():
    for fam in S.TRAIN_FAMILIES:
        train = {tuple(r) for lauf in range(3) for r in S.training_set(fam, lauf).tolist()}
        test = {tuple(r) for r in S.test_set(fam).tolist()}
        assert len(train & test) == 0


# --- (g) angezeigte Kennzahlen ---------------------------------------------------------------------------------------------------------
def test_g_delta_bilanz_und_spielraum_von_hand():
    bins, bf, lb = np.array([4, 5, 3, 6]), np.array([4, 4, 4, 6]), np.array([4, 4, 3, 5])
    want = 100 * ((4 / 4 + 5 / 4 + 3 / 3 + 6 / 5) / 4 - (4 / 4 + 4 / 4 + 4 / 3 + 6 / 5) / 4)
    assert E.delta_pp(bins, bf, lb) == pytest.approx(want, rel=1e-12)
    assert E.bilanz(bins, bf) == (1, 2, 1)
    sizes = np.array([[40, 75, 70, 15, 15, 25, 30, 25], [55, 65, 70, 20, 30, 65, 30, 40]], dtype=np.int64)
    sp = E.spielraum(sizes)
    # Instanz 1: FF 3, BF 4, LB 3; Instanz 2: FF 5, BF 4, LB 4
    assert sp["FF"] == pytest.approx(100 * ((3 / 3 + 5 / 4) / 2 - 1)) and sp["BF"] == pytest.approx(100 * ((4 / 3 + 4 / 4) / 2 - 1))
    assert (sp["ff_lt_bf"], sp["ff_eq_bf"], sp["ff_gt_bf"], sp["n"], sp["n_items"]) == (1, 0, 1, 2, 8) and sp["lb_mean"] == 3.5


def test_g_kandidatenstatistik_von_hand():
    bf, ff = np.array([4, 3]), np.array([4, 4])
    vecs = [np.array([4, 3]), np.array([4, 4]), np.array([3, 3]), np.array([4, 3]), np.array([5, 5])]
    fits = [3.5, 4.0, 3.0, 3.5, 5.0]
    st = E.candidate_stats(vecs, fits, bf, ff, 3.5)
    assert st == dict(n=5, distinct=4, eq_bf=2, eq_ff=1, eq_any=3, better=1)


def test_g_verlauf_steht_an_der_achse_der_auswertungen():
    s = E.search_run("gl_20_70", "EA-S", 0, budget=200)
    imp = E.improvements(s)
    assert imp[0][0] == 1 and [p[0] for p in imp] == sorted({p[0] for p in imp}) and imp[-1][0] <= 200
    best = np.minimum.accumulate(s["fits"])
    for n_eval, d_train, j in imp:
        assert d_train == pytest.approx(100 * (best[n_eval - 1] - s["bf_fit"]), rel=1e-12, abs=1e-12)
        assert s["pool_fits"][j] == pytest.approx(best[n_eval - 1], rel=1e-12)       # die Regel an dieser Stelle hat genau diese Fitness
    assert len(s["eval_idx"]) == 200 and s["stat"]["n"] == 200 and len(s["pool_inds"]) + s["stat"]["dup"] == 200
    assert s["cps"][-1][1] == pytest.approx(best[-1])


def test_g_mengen_eines_laufs_gegen_direkte_rechnung():
    """Ein kleiner Lauf: die Deltas der Mengen in `run_record` gegen eine direkte Rechnung aus den Behälterzahlen der Mitglieder."""
    tests = {"same60": S.test_set("gl_20_70", K=40)}
    s = E.search_run("gl_20_70", "EA-S", 1, budget=200)
    rec = E.run_record("gl_20_70", "EA-S", 1, tests, budget=200, search=s)
    sets = E.build_sets(s["pool_fits"], s["pool_matrix"], E.set_seed("EA-S", "gl_20_70", 1))
    lb, bf = A.lower_bound(tests["same60"]), A.run_known(tests["same60"], "BF")
    for typ in ("greedy", "topk"):
        mem = sets[typ][:3]
        mte = np.array([A.simulate(tests["same60"], R.make_fn("S", s["pool_inds"][j]))[0] for j in mem])
        direct = 100 * float(np.mean((mte.min(axis=0) - bf) / lb))
        assert rec["sets"][typ]["3"]["T"]["same60"][0] == pytest.approx(direct, rel=1e-9, abs=1e-12)
        assert rec["mem"][typ][:3] == [R.s_str(s["pool_inds"][j]) for j in mem]
    assert rec["sets"]["greedy"]["1"]["T"]["same60"][0] == rec["sets"]["greedy"]["1"]["T"]["same60"][2]      # k = 1: Orakel = fest
    assert rec["sets"]["bfplus"]["1"]["T"]["same60"] == [0.0, 0.0, 0.0]                                       # nur Best Fit
    assert rec["sets"]["greedy"]["3"]["T"]["same60"][0] <= rec["sets"]["greedy"]["1"]["T"]["same60"][0] + 1e-12
    assert rec["pool"] == len(s["pool_inds"]) and rec["stat"]["n"] == 200
    assert rec["ref"]["same60"] == pytest.approx(rec["test"])
