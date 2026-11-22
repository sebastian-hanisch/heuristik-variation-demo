"""Große Fassung der Orakel-Prüfungen (Scratch-Lauf, nicht Teil der Suite): dieselben unabhängigen Rechenwege wie in
tests/test_oracle_heuristik_variation.py, aber mit mindestens 200 Zufallsinstanzen je Gruppe. Gibt die Zahl der Prüfungen selbst aus.

Aufruf (im Demo-Ordner): python messungen/orakel_gross.py > messungen/ausgabe_orakel_gross.txt"""

import itertools
import sys
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import hva_algorithm as A  # noqa: E402
import hva_arms as AR  # noqa: E402
import hva_rules as R  # noqa: E402
import hva_scenario as S  # noqa: E402
import hva_sets as M  # noqa: E402
import hva_variation as V  # noqa: E402
import referenz as ref  # noqa: E402
from hva_rng import SplitMix64  # noqa: E402

N = 0
FEHLER = 0


def ok(cond, msg=""):
    global N, FEHLER
    N += 1
    if not cond:
        FEHLER += 1
        print("FEHLER:", msg)


def instances(seed, count, n_lo=5, n_hi=14, lo=5, hi=75):
    rng = SplitMix64(seed)
    return [[lo + rng.below(hi - lo + 1) for _ in range(n_lo + rng.below(n_hi - n_lo + 1))] for _ in range(count)]


def table(rng, p, k, lo=3, hi=8):
    return np.array([[lo + rng.below(hi - lo + 1) for _ in range(k)] for _ in range(p)], dtype=np.int64)


# (a) Packung
for items in instances(101, 400):
    arr = np.array([items], dtype=np.int64)
    for name in ("FF", "BF", "WF"):
        bins, asg = A.simulate(arr, A.KNOWN[name], want_assign=True)
        ok((int(bins[0]), asg[0].tolist()) == ref.direct_pack(items, name), ("bekannt", name, items))
rng = SplitMix64(102)
for items in instances(103, 300):
    ind = R.s_sample(rng)
    for _ in range(rng.below(5)):
        ind = R.s_mutate(ind, rng)
    bins, asg = A.simulate(np.array([items], dtype=np.int64), R.make_s_fn(ind), want_assign=True)
    ok((int(bins[0]), asg[0].tolist()) == ref.scalar_simulate(items, ref.s_scalar(ind)), ("Schablone", ind, items))
rng = SplitMix64(104)
for items in instances(105, 300):
    t = R.rand_tree_ramped(rng)
    bins, asg = A.simulate(np.array([items], dtype=np.int64), R.make_tree_fn(t), want_assign=True)
    ok((int(bins[0]), asg[0].tolist()) == ref.scalar_simulate(items, ref.tree_scalar(t)), ("Baum", t, items))
for items in instances(106, 200, lo=1, hi=14):
    items = [5 * x for x in items]
    for a in (0, 5, 10, 15, 20, 30, 40):
        bins, asg = A.simulate(np.array([items], dtype=np.int64), R.make_s_fn(V.s1(a / 100)), want_assign=True)
        ok((int(bins[0]), asg[0].tolist()) == ref.direct_pr(items, a), ("Lückenziel", a, items))
n_a = N
print("(a) Packung: %d Prüfungen" % n_a)

# (b) Optimum und Schranken
for items in instances(107, 240, n_lo=4, n_hi=8):
    opt = A.opt_dp(items)
    ok(opt == ref.opt_enum(items), ("Optimum", items))
    arr = np.array([items], dtype=np.int64)
    ok(int(A.lower_bound(arr)[0]) <= opt <= int(A.run_known(arr, "FFD")[0]) and opt <= int(A.run_known(arr, "BF")[0]), ("Schranken", items))
for items in instances(108, 60, n_lo=8, n_hi=12):
    ok(A.opt_dp(items) == ref.opt_milp(items), ("MILP", items))
n_b = N - n_a
print("(b) Optimum: %d Prüfungen" % n_b)

# (c) Mengen
rng = SplitMix64(109)
worse = {1: 0, 2: 0, 3: 0, 4: 0}
total = 0
gap_max = 0
for _ in range(400):
    p, kk = 4 + rng.below(8), 5 + rng.below(6)
    tab = table(rng, p, kk)
    fits = tab.mean(axis=1)
    members = M.random_set(p, 1 + rng.below(p), rng)
    naive = sum(min(Fr(int(tab[m][i])) for m in members) for i in range(kk))
    ok(M.set_cost(tab, members) == naive, "set_cost")
    for k in (1, 2, 3, 4):
        g = M.greedy_set(tab, fits, k)
        best = min(M.set_cost(tab, list(c)) for c in itertools.combinations(range(p), k))
        cost = M.set_cost(tab, g)
        ok(cost >= best and (k > 1 or cost == best), "greedy gegen Aufzählung")
        worse[k] += cost > best
        gap_max = max(gap_max, cost - best)
    total += 1
print("greedy verfehlt das Optimum über alle k-Mengen in %s von je %d Tabellen (k = 1, 2, 3, 4); größte Lücke %d Behälter (Tabellen mit 4 bis 11 Regeln, 5 bis 10 Instanzen, Werte 3 bis 8)" % (
    [worse[k] for k in (1, 2, 3, 4)], total, gap_max))
for _ in range(60):
    tab = table(rng, 5, 6)
    subsets = [list(c) for k in range(1, 6) for c in itertools.combinations(range(5), k)]
    cost = {tuple(s): M.set_cost(tab, s) for s in subsets}
    good = True
    for a in subsets:
        for b in subsets:
            if set(a) <= set(b):
                good = good and cost[tuple(b)] <= cost[tuple(a)]
                for r in set(range(5)) - set(b):
                    good = good and cost[tuple(a)] - cost[tuple(sorted(a + [r]))] >= cost[tuple(b)] - cost[tuple(sorted(b + [r]))] >= 0
    ok(good, "Monotonie und abnehmender Zusatznutzen")
n_c = N - n_a - n_b
print("(c) Mengen: %d Prüfungen" % n_c)

# (d) Merkmalswahl
rng = SplitMix64(110)
for _ in range(300):
    m = 1 + rng.below(5)
    train = np.array([[10 + 5 * rng.below(8) for _ in range(6)] for _ in range(2 + rng.below(8))], dtype=np.int64)
    test = np.array([[10 + 5 * rng.below(8) for _ in range(6)] for _ in range(5)], dtype=np.int64)
    ftr = [Fr(int(sum(x[:m])), m) for x in train.tolist()]
    if len(set(ftr)) == 1:
        continue
    want = [min(range(len(ftr)), key=lambda t: (abs(Fr(int(sum(x[:m])), m) - ftr[t]), t)) for x in test.tolist()]
    ok(M.nearest_neighbor(M.first_features(train, m, False), M.first_features(test, m, False)).tolist() == want, "nächster Nachbar")
n_d = N - n_a - n_b - n_c
print("(d) Merkmalswahl: %d Prüfungen" % n_d)

# (e) Operator
a, b = SplitMix64(111), SplitMix64(111)
ind = V.BEST_FIT_IND
counts = {k: 0 for k in V.KINDS}
n_mut = 100000
good = True
for _ in range(n_mut):
    kind, child = V.mutate_with_kind(ind, a)
    good = good and child == R.s_mutate(ind, b)
    counts[kind] += 1
    ind = child
ok(good, "mutate_with_kind gleich s_mutate")
for kind, p in V.KIND_PROBS.items():
    ok(abs(counts[kind] / n_mut - p) < 4.5 * (p * (1 - p) / n_mut) ** 0.5, ("Anteil", kind, counts[kind]))
print("Anteile der Operatorarten über %d Züge: %s" % (n_mut, {k: round(v / n_mut, 4) for k, v in counts.items()}))
n_e = N - n_a - n_b - n_c - n_d
print("(e) Operator: %d Prüfungen" % n_e)

# (f) Zählung
for fam in S.TRAIN_FAMILIES:
    train = S.training_set(fam, 0, K=6)
    for arm in AR.ARM_ORDER:
        bud = AR.run_arm(arm, train, AR.arm_seed(arm, fam, 0), (200, 400, 1000))
        ok(bud.n == 1000 and len(bud.pool) + bud.n_dup == 1000, ("Zählung", fam, arm))
        ok(all(bud.at[c][0] == min(bud.fits[:c]) for c in (200, 400, 1000)), ("Bestwert bei den Stufen", fam, arm))
n_f = N - n_a - n_b - n_c - n_d - n_e
print("(f) Zählung: %d Prüfungen" % n_f)

print("GESAMT: %d Prüfungen, %d Fehler" % (N, FEHLER))
