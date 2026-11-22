"""Rechnungen der Experimente: ein Lauf eines Arms mit Pool, Mengen aus dem Pool, Einsatz auf Testmengen, Kinder eines Elternteils.

Dieselben Funktionen rechnen die App (live) und die eingefrorenen Messreihen (`generate_precomputed.py`). Delta heißt überall:
Überschuss(Regel oder Menge) minus Überschuss(Best Fit) in Prozentpunkten, positiv = schlechter als Best Fit.
"""

import numpy as np

import hva_algorithm as A
import hva_arms as AR
import hva_constants as C
import hva_rules as R
import hva_scenario as S
import hva_sets as M
import hva_variation as V
from hva_rng import SplitMix64


def delta_pp(bins, bins_bf, lb):
    """Überschuss(Regel) minus Überschuss(Best Fit) in Prozentpunkten."""
    lb = np.asarray(lb)
    return 100.0 * (float(np.mean(np.asarray(bins) / lb)) - float(np.mean(np.asarray(bins_bf) / lb)))


def bilanz(bins, bins_bf):
    """Instanz-Bilanz gegen Best Fit: (weniger, gleich viele, mehr Behälter), als Zählungen."""
    d = np.asarray(bins) - np.asarray(bins_bf)
    return int((d < 0).sum()), int((d == 0).sum()), int((d > 0).sum())


# ---------------------------------------------------------------------------------------------------------------------------------------
# Testmengen
# ---------------------------------------------------------------------------------------------------------------------------------------
def test_sets(familie, test_seed=0, names=C.TESTSETS):
    """Die Testmengen einer Trainingsfamilie: {Name: Größen}. Die fremden Familien sind die beiden anderen bei 60 Gegenständen."""
    out = {}
    others = C.other_families(familie)
    for name in names:
        if name in C.TESTSET_SPEC:
            n, K = C.TESTSET_SPEC[name]
            out[name] = S.test_set(familie, n=n, test_seed=test_seed, K=K)
        else:
            out[name] = S.test_set(others[0 if name == "fremd1" else 1], test_seed=test_seed)
    return out


test_sets.__test__ = False


def test_base(tests):
    """Je Testmenge: untere Schranke und Behälterzahlen von Best Fit."""
    return {name: (A.lower_bound(sz), A.run_known(sz, "BF")) for name, sz in tests.items()}


test_base.__test__ = False


# ---------------------------------------------------------------------------------------------------------------------------------------
# Ein Lauf eines Arms (nur Training)
# ---------------------------------------------------------------------------------------------------------------------------------------
def candidate_stats(vecs, fits, bf_vec, ff_vec, bf_fit):
    """Kandidatenstatistik eines Laufs als Zählungen über alle Auswertungen: dieselben Behälterzahlen wie Best Fit bzw. First Fit
    auf allen Trainingsinstanzen, verschiedene Trainingsvektoren, besser als Best Fit."""
    eq_bf = eq_ff = eq_any = 0
    seen = set()
    for v in vecs:
        b = bool((v == bf_vec).all())
        f = bool((v == ff_vec).all())
        eq_bf += b
        eq_ff += f
        eq_any += b or f
        seen.add(v.tobytes())
    better = sum(1 for x in fits if x < bf_fit)
    return dict(n=len(vecs), distinct=len(seen), eq_bf=eq_bf, eq_ff=eq_ff, eq_any=eq_any, better=better)


def search_run(familie, arm, lauf, K=C.K_TRAIN, budget=C.FROZEN_BUDGET, checkpoints=C.CHECKPOINTS):
    """Ein Lauf des Arms auf K Trainingsinstanzen bis `budget` Auswertungen, mit Pool (alle verschiedenen bewerteten Regeln)."""
    train = S.training_set(familie, lauf, K)
    lb = A.lower_bound(train)
    bf_vec = A.run_known(train, "BF")
    ff_vec = A.run_known(train, "FF")
    bf_fit = A.fitness(bf_vec, lb)
    cps = tuple(sorted({c for c in checkpoints if c < budget} | {budget}))
    bud = AR.run_arm(arm, train, AR.arm_seed(arm, familie, lauf), cps, keep_pool=True)
    stat = candidate_stats(bud.vecs, bud.fits, bf_vec, ff_vec, bf_fit)
    stat["dup"] = bud.n_dup
    inds = [v[0] for v in bud.pool.values()]
    pos = {id(v[1]): j for j, v in enumerate(bud.pool.values())}       # der Zwischenspeicher gibt dasselbe Array-Objekt zurück
    matrix = np.array([v[1] for v in bud.pool.values()], dtype=np.int64)
    return dict(
        familie=familie, arm=arm, space=bud.space, lauf=lauf, K=K, budget=budget, train=train, lb=lb, bf_vec=bf_vec, bf_fit=bf_fit,
        cps=[(c, bud.at[c][0], bud.at[c][1]) for c in cps], fits=np.array(bud.fits), stat=stat,
        eval_idx=np.array([pos[id(v)] for v in bud.vecs], dtype=np.int64), pool_inds=inds, pool_matrix=matrix, pool_fits=np.array([A.fitness(row, lb) for row in matrix]),
    )


def curve_points(fits, bf_fit):
    """Verlauf des Bestwerts über die Auswertungen: (Auswertung, Trainings-Delta des Besten bisher), nur die Verbesserungen."""
    out = []
    best = None
    for i, f in enumerate(fits, start=1):
        if best is None or f < best:
            best = float(f)
            out.append((i, 100.0 * (best - bf_fit)))
    return out


def improvements(search):
    """Die Stellen, an denen der Bestwert sich verbessert hat: Liste von (Auswertung, Trainings-Delta, Pool-Index der Regel)."""
    out = []
    best = None
    for i, f in enumerate(search["fits"]):
        if best is None or f < best:
            best = float(f)
            out.append((i + 1, 100.0 * (best - search["bf_fit"]), int(search["eval_idx"][i])))
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------
# Mengen aus einem Pool und ihr Einsatz
# ---------------------------------------------------------------------------------------------------------------------------------------
class Evaluator:
    """Behälterzahlen von Pool-Regeln (Index) und bekannten Regeln (Name) auf Trainings- und Testmengen, mit Zwischenspeicher."""

    def __init__(self, space, inds, matrix, train, tests):
        self.space = space
        self.inds = inds
        self.matrix = matrix
        self.train = train
        self.tests = tests
        self.base = test_base(tests)
        self._cache = {}
        self._known_train = {}

    def train_bins(self, member):
        if isinstance(member, str):
            if member not in self._known_train:
                self._known_train[member] = A.run_known(self.train, member)
            return self._known_train[member]
        return self.matrix[member]

    def bins(self, member, ts):
        key = (member, ts)
        if key not in self._cache:
            if isinstance(member, str):
                self._cache[key] = A.run_known(self.tests[ts], member)
            else:
                self._cache[key] = A.simulate(self.tests[ts], R.make_fn(self.space, self.inds[member]))[0]
        return self._cache[key]

    def delta(self, bins, ts):
        lb, bf = self.base[ts]
        return delta_pp(bins, bf, lb)

    def choice_bins(self, members, ts):
        """Behälterzahlen der drei Wahlarten für eine Menge auf einer Testmenge."""
        mtr = np.array([self.train_bins(m) for m in members])
        mte = np.array([self.bins(m, ts) for m in members])
        return dict(
            orakel=M.oracle_choice(mte),
            merkmal=M.feature_choice(self.train, mtr, self.tests[ts], mte),
            fest=M.fixed_choice(mtr, mte),
        )

    def choice_deltas(self, members, ts):
        cb = self.choice_bins(members, ts)
        return {w: self.delta(cb[w], ts) for w in C.CHOICES}

    def train_delta(self, members, lb, bf_vec):
        """Orakelwahl auf den Trainingsinstanzen gegen Best Fit (so wurde die greedy-Menge gewählt)."""
        mtr = np.array([self.train_bins(m) for m in members])
        return delta_pp(mtr.min(axis=0), bf_vec, lb)


def set_seed(arm, familie, lauf):
    return S.seed_for(AR.SEED_SETS, AR.ARM_ORDER.index(arm), S.FAMILY_ORDER.index(familie), lauf)


def build_sets(pool_fits, pool_matrix, seed, k_max=max(C.K_OPTIONS), n_random=C.N_RANDOM_SETS):
    """Mitglieder der Mengenarten in voller Länge k_max; die Menge der Größe k ist jeweils der Anfang der Liste.

    Rückgabe: {"greedy": [Indizes], "topk": [Indizes], "zufall": [[Indizes] x n_random]}."""
    return dict(
        greedy=M.greedy_set(pool_matrix, pool_fits, k_max),
        topk=M.topk_set(pool_fits, k_max),
        zufall=M.random_sets(len(pool_fits), k_max, SplitMix64(seed), n_random),
    )


def members_of(sets, typ, k):
    """Liste der Mengen (je eine Liste von Mitgliedern) einer Art bei Größe k; »bfplus« ist Best Fit plus k − 1 beliebige."""
    if typ in ("greedy", "topk"):
        return [sets[typ][:k]]
    if typ == "zufall":
        return [m[:k] for m in sets["zufall"]]
    if typ == "bfplus":
        return [["BF"] + m[: k - 1] for m in sets["zufall"]]
    if typ == "portfolio":
        return [list(M.PORTFOLIO)]
    raise ValueError(typ)


def set_summary(ev, sets, typ, k, lb, bf_vec, names):
    """Eine Mengenart bei Größe k: Trainings-Delta der Orakelwahl, Mitglieder mit den Trainingsbehälterzahlen von Best Fit und je
    Testmenge die Deltas der drei Wahlarten gegen Best Fit. Bei mehreren Mengen (Zufall) das Mittel über die Mengen."""
    groups = members_of(sets, typ, k)
    train = float(np.mean([ev.train_delta(g, lb, bf_vec) for g in groups]))
    eq_bf = float(np.mean([sum(bool((ev.train_bins(m) == bf_vec).all()) for m in g) for g in groups]))
    T = {}
    for ts in names:
        ds = [ev.choice_deltas(g, ts) for g in groups]
        T[ts] = [float(np.mean([d[w] for d in ds])) for w in C.CHOICES]
    return dict(train=train, eq_bf=eq_bf, size=len(groups[0]), T=T)


def run_record(familie, arm, lauf, tests, K=C.K_TRAIN, budget=C.FROZEN_BUDGET, checkpoints=C.CHECKPOINTS, set_types=C.SET_TYPES_FROZEN,
               k_list=C.K_OPTIONS, search=None):
    """Ein Lauf samt Test und Mengen als JSON-taugliches Wörterbuch (so stehen die eingefrorenen Läufe in `hva_precomputed.json`)."""
    s = search_run(familie, arm, lauf, K, budget, checkpoints) if search is None else search
    space = s["space"]
    names = list(tests)
    ev = Evaluator(space, s["pool_inds"], s["pool_matrix"], s["train"], tests)
    key_to_idx = {R.ind_key(space, ind): j for j, ind in enumerate(s["pool_inds"])}
    main = names[0]
    cps = []
    for c, fit, ind in s["cps"]:
        j = key_to_idx[R.ind_key(space, ind)]
        cps.append(dict(n=c, train=100.0 * (fit - s["bf_fit"]), test=ev.delta(ev.bins(j, main), main)))
    _, fit, ind = s["cps"][-1]
    jbest = key_to_idx[R.ind_key(space, ind)]
    sets = build_sets(s["pool_fits"], s["pool_matrix"], set_seed(arm, familie, lauf))
    rec = dict(
        lauf=lauf, cps=cps, train=cps[-1]["train"], test=cps[-1]["test"], rule=R.ind_str(space, ind), ind=R.ind_to_json(space, ind),
        knoten=(R.tree_size(ind) if space == "T" else None),
        bilanz=list(bilanz(ev.bins(jbest, main), ev.base[main][1])), stat=dict(s["stat"]), pool=len(s["pool_inds"]),
        ref={ts: ev.delta(ev.bins(jbest, ts), ts) for ts in names},
        sets={typ: {str(k): set_summary(ev, sets, typ, k, s["lb"], s["bf_vec"], names) for k in k_list} for typ in set_types},
        mem={typ: [R.ind_str(space, s["pool_inds"][j]) for j in sets[typ]] for typ in ("greedy", "topk")},
    )
    return rec


def portfolio_record(familie, lauf, tests, K=C.K_TRAIN):
    """Die feste Menge Best Fit, First Fit, Worst Fit auf den Trainingsinstanzen eines Laufs und allen Testmengen."""
    train = S.training_set(familie, lauf, K)
    lb = A.lower_bound(train)
    bf_vec = A.run_known(train, "BF")
    ev = Evaluator("S", [], np.zeros((0, K), dtype=np.int64), train, tests)
    out = set_summary(ev, {}, "portfolio", len(M.PORTFOLIO), lb, bf_vec, list(tests))
    out["lauf"] = lauf
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------
# Kinder eines Elternteils (Experiment 1)
# ---------------------------------------------------------------------------------------------------------------------------------------
def parent_ind(parent, familie, lauf):
    """Der Elternteil: Best Fit als Schablone, das Lückenziel a = 0,20 oder eine frisch gezogene Schablone (je Familie und Lauf)."""
    if parent == "bf":
        return V.BEST_FIT_IND
    if parent == "s1_20":
        return V.s1(0.2)
    if parent == "frisch":
        return R.s_sample(SplitMix64(S.seed_for(AR.SEED_PARENT, S.FAMILY_ORDER.index(familie), lauf)))
    raise ValueError(parent)


def variation_experiment(familie, parent, lauf, n_children, test, K=C.K_TRAIN):
    """n Kinder eines Elternteils auf den Trainingsinstanzen des Laufs; die auf dem Training besseren zusätzlich auf den Testinstanzen.

    Rückgabe: Zählungen der Klassen, Tabelle nach Operatorart, Punkte (Trainings-, Test-Delta gegen den Elternteil) der besseren Kinder."""
    train = S.training_set(familie, lauf, K)
    lb = A.lower_bound(train)
    lbt = A.lower_bound(test)
    bf_fit = A.fitness(A.run_known(train, "BF"), lb)
    par = parent_ind(parent, familie, lauf)
    par_bins = A.simulate(train, R.make_s_fn(par))[0]
    par_fit = A.fitness(par_bins, lb)
    par_test = A.simulate(test, R.make_s_fn(par))[0]
    rng = SplitMix64(S.seed_for(AR.SEED_CHILDREN, C.PARENTS.index(parent), S.FAMILY_ORDER.index(familie), lauf))
    kids = V.children(par, n_children, rng)
    cache = {}
    tcache = {}
    kinds, classes, points = [], [], []
    better_bf = 0
    better_test = 0
    for kind, child in kids:
        key = R.ind_key("S", child)
        if key not in cache:
            cache[key] = A.simulate(train, R.make_s_fn(child))[0]
        cb = cache[key]
        cls = V.classify(par_bins, cb, lb)
        kinds.append(kind)
        classes.append(cls)
        fit = A.fitness(cb, lb)
        better_bf += fit < bf_fit - V.FIT_TIE
        if cls == "besser":
            if key not in tcache:
                tcache[key] = A.simulate(test, R.make_s_fn(child))[0]
            d_test = delta_pp(tcache[key], par_test, lbt)
            better_test += d_test < -1e-10
            points.append((100.0 * (fit - par_fit), d_test))
    counts = {c: classes.count(c) for c in V.CLASSES}
    return dict(
        familie=familie, parent=parent, lauf=lauf, n=n_children, parent_rule=R.s_str(par), parent_train=100.0 * (par_fit - bf_fit),
        parent_worse_bf=bool(par_fit > bf_fit + V.FIT_TIE), counts=counts, better_test=int(better_test), better_bf=int(better_bf),
        table=V.operator_table(kinds, classes), points=points, distinct=len(cache),
    )


def variation_record(familie, parent, lauf, n_children, test, K=C.K_TRAIN):
    """Wie `variation_experiment`, als kompaktes JSON-taugliches Wörterbuch (ohne die Punkte)."""
    r = variation_experiment(familie, parent, lauf, n_children, test, K)
    return dict(
        lauf=lauf, n=r["n"], c=[r["counts"][c] for c in V.CLASSES], bt=r["better_test"], bbf=r["better_bf"], pw=r["parent_worse_bf"],
        art={k: [r["table"][k]["n"]] + [r["table"][k][c] for c in V.CLASSES] for k in V.KINDS},
    )


# ---------------------------------------------------------------------------------------------------------------------------------------
# Spielraum einer Instanzmenge: bekannte Regeln gegeneinander, Schranken
# ---------------------------------------------------------------------------------------------------------------------------------------
def spielraum(sizes):
    """Überschuss der bekannten Regeln über die untere Schranke (in Prozent) und wie oft First Fit und Best Fit sich unterscheiden."""
    lb = A.lower_bound(sizes)
    bins = {nm: A.run_known(sizes, nm) for nm in ("FF", "BF", "WF", "NF", "FFD")}
    out = {nm: A.excess_pct(b, lb) for nm, b in bins.items()}
    d = bins["FF"] - bins["BF"]
    out.update(ff_lt_bf=int((d < 0).sum()), ff_eq_bf=int((d == 0).sum()), ff_gt_bf=int((d > 0).sum()), n=int(len(lb)),
               n_items=int(np.asarray(sizes).shape[1]), lb_mean=float(np.mean(lb)))
    return out


def spielraum_opt(sizes):
    """Kleine Instanzen gegen das exakte Optimum: Best Fit über dem Optimum, untere Schranke unter dem Optimum (je in Prozent)."""
    opt = np.array([A.opt_dp(row) for row in np.asarray(sizes)], dtype=np.int64)
    lb = A.lower_bound(sizes)
    bf = A.run_known(sizes, "BF")
    return dict(
        bf_ueber_opt=A.excess_pct(bf, opt), lb_unter_opt=100.0 * (1.0 - float(np.mean(lb / opt))), bf_optimal=int((bf == opt).sum()),
        lb_eq_opt=int((lb == opt).sum()), n=int(len(opt)), n_items=int(np.asarray(sizes).shape[1]),
    )


# ---------------------------------------------------------------------------------------------------------------------------------------
# Handbeispiel (Primäransicht): vier Regeln, vier Trainings- und zwei Testinstanzen
# ---------------------------------------------------------------------------------------------------------------------------------------
HAND_A = (40, 75, 70, 15, 15, 25, 30, 25)
HAND_TRAIN = (
    (25, 35, 65, 35, 50, 70, 30, 40),
    (10, 55, 15, 35, 15, 60, 20, 55),
    (45, 65, 15, 75, 15, 75, 25, 55),
    (30, 35, 70, 50, 25, 15, 30, 25),
)
HAND_TEST = ((55, 65, 70, 20, 30, 65, 30, 40), (30, 45, 35, 10, 60, 45, 35, 25))
HAND_RULES = ("BF", "FF", "WF", "S1")       # S1 = Lückenziel mit a = 0,10; Reihenfolge = Index bei Gleichständen
HAND_RULE_LABELS = {"BF": "Best Fit", "FF": "First Fit", "WF": "Worst Fit", "S1": "S1(0,10)"}
HAND_K = 2
HAND_FIRST = 3                               # die Merkmalswahl des Handbeispiels sieht die ersten drei Gegenstände


def hand_fn(name):
    return R.make_s_fn(V.s1(0.10)) if name == "S1" else A.KNOWN[name]


def hand_example():
    """Rechnet das Handbeispiel mit denselben Funktionen wie die Messung: Tabelle, greedy und topk mit k = 2, Einsatz mit drei Wahlarten."""
    train = np.array(HAND_TRAIN, dtype=np.int64)
    test = np.array(HAND_TEST, dtype=np.int64)
    mtr = np.array([A.simulate(train, hand_fn(r))[0] for r in HAND_RULES])
    mte = np.array([A.simulate(test, hand_fn(r))[0] for r in HAND_RULES])
    lb = A.lower_bound(train)
    fits = np.array([A.fitness(row, lb) for row in mtr])
    sums = mtr.sum(axis=1)
    # Im Handbeispiel zählen Summen der Behälterzahlen; die Reihenfolge der Regeln nach Summe ist hier dieselbe wie nach Fitness.
    greedy = M.greedy_set(mtr, sums.astype(float), HAND_K)
    topk = M.topk_set(sums.astype(float), HAND_K)
    g_tr, g_te = mtr[greedy], mte[greedy]
    pick = M.feature_pick(train, g_tr, test, m=HAND_FIRST, with_std=False)
    nn = M.nearest_neighbor(M.first_features(train, HAND_FIRST, False), M.first_features(test, HAND_FIRST, False))
    return dict(
        rules=HAND_RULES, train_bins=mtr, test_bins=mte, train_sums=[int(x) for x in sums], test_sums=[int(x) for x in mte.sum(axis=1)], fits=fits,
        lb_train=[int(x) for x in lb], lb_test=[int(x) for x in A.lower_bound(test)],
        opt_train=[A.opt_dp(x) for x in HAND_TRAIN], opt_test=[A.opt_dp(x) for x in HAND_TEST],
        greedy=greedy, greedy_steps=[M.set_cost(mtr, greedy[: i + 1]) for i in range(len(greedy))], second_costs=[M.set_cost(mtr, [greedy[0], j]) for j in range(len(HAND_RULES))],
        topk=topk, topk_cost=M.set_cost(mtr, topk),
        labels=[int(x) for x in M.labels(g_tr)], nn=[int(x) for x in nn], pick=[int(x) for x in pick],
        feat_train=[int(sum(x[:HAND_FIRST])) for x in HAND_TRAIN], feat_test=[int(sum(x[:HAND_FIRST])) for x in HAND_TEST],
        oracle=int(M.oracle_choice(g_te).sum()), feature=int(M.feature_choice(train, g_tr, test, g_te, HAND_FIRST, False).sum()),
        fixed=int(M.fixed_choice(g_tr, g_te).sum()), bf=int(mte[0].sum()), topk_oracle=int(M.oracle_choice(mte[topk]).sum()),
        portfolio=int(M.oracle_choice(mte[[HAND_RULES.index(r) for r in M.PORTFOLIO]]).sum()),
        runs=dict(orakel=M.runs_per_instance("orakel", HAND_K), merkmal=1, fest=1, portfolio=len(M.PORTFOLIO), pool=len(HAND_RULES)),
    )
