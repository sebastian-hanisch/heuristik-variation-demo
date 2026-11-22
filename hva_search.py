"""Suche im Regelraum: Zählung der Auswertungen, vier Suchverfahren, parametrische Gegner.

Zähleinheit: eine **Auswertung** = eine Regel einmal auf dem ganzen Satz der Trainingsinstanzen. Jede bewertete Regel zählt, auch wenn ihr
Ergebnis aus dem Zwischenspeicher kommt. Die Verfahren sind folgentreu: ein Lauf mit kleinerem Budget ist der Anfang des Laufs mit größerem
Budget (gleicher Zufalls-Seed), deshalb lässt sich der Bestwert bei jeder Stufe aus einem Lauf ablesen.
"""

import numpy as np

import hva_algorithm as A
import hva_rules as R
from hva_rng import SplitMix64


class Budget:
    """Zählt Auswertungen, hält den Bestwert bei den Stufen, den Verlauf und (optional) alle verschiedenen Kandidaten.

    Attribute nach dem Lauf:
        n          Zahl der Auswertungen
        best       (Fitness, Individuum, Schlüssel) des Besten bisher (bei Gleichstand der frühere)
        at         {Stufe: best zu diesem Zeitpunkt}
        fits       Fitness jeder Auswertung in Reihenfolge
        vecs       Behälterzahlen jeder Auswertung auf den Trainingsinstanzen, Liste von (K,)-Arrays
        n_dup      Auswertungen, die aus dem Zwischenspeicher kamen
        cache      {Schlüssel: Behälterzahlen (K,)} aller verschiedenen Kandidaten
        pool       {Schlüssel: (Individuum, Behälterzahlen)} in der Reihenfolge des ersten Auftretens, nur mit keep_pool
    """

    def __init__(self, space, sizes, lb, checkpoints=(200, 1000), keep_pool=False):
        self.space = space
        self.sizes = sizes
        self.lb = lb
        self.cp = tuple(sorted(checkpoints))
        self.n = 0
        self.best = None
        self.at = {}
        self.fits = []
        self.vecs = []
        self.n_dup = 0
        self.cache = {}
        self.pool = {} if keep_pool else None

    def done(self):
        return self.n >= self.cp[-1]

    def eval(self, ind):
        """Eine Auswertung: Regel auf allen Trainingsinstanzen; Rückgabe ist die Trainingsfitness (kleiner ist besser)."""
        key = R.ind_key(self.space, ind)
        if key in self.cache:
            bins = self.cache[key]
            self.n_dup += 1
        else:
            bins = A.simulate(self.sizes, R.make_fn(self.space, ind))[0]
            self.cache[key] = bins
            if self.pool is not None:
                self.pool[key] = (ind, bins)
        fit = A.fitness(bins, self.lb)
        self.n += 1
        self.fits.append(fit)
        self.vecs.append(bins)
        if self.best is None or fit < self.best[0]:
            self.best = (fit, ind, key)
        if self.n in self.cp:
            self.at[self.n] = self.best
        return fit


def run_random(space, bud, rng):
    """Zufallssuche: unabhängige Ziehungen aus dem Raum."""
    while not bud.done():
        bud.eval(R.sample_ind(space, rng))


def run_11(space, bud, rng):
    """(1+1)-Suche: ein Elter, ein Kind je Schritt; das Kind ersetzt den Elter, wenn es nicht schlechter ist."""
    parent = R.sample_ind(space, rng)
    pf = bud.eval(parent)
    while not bud.done():
        child = R.mutate_ind(space, parent, rng)
        cf = bud.eval(child)
        if cf <= pf:
            parent, pf = child, cf


def tournament(fits, k, rng):
    """Turnier: k Teilnehmer mit Zurücklegen ziehen, der mit der kleinsten Fitness gewinnt (Gleichstand: der zuerst gezogene)."""
    idx = [rng.below(len(fits)) for _ in range(k)]
    best = idx[0]
    for i in idx[1:]:
        if fits[i] < fits[best]:
            best = i
    return best


def run_gp(bud, rng, select=True, pop_n=30, tour=3, p_cross=0.8):
    """Population von Bäumen: Elite 1, Turnier 3, Teilbaum-Kreuzung 0,8, sonst Teilbaum-Mutation.

    Mit select=False werden die Eltern gleichverteilt gezogen und es gibt keine Elite (Variation ohne Auswahl der Besten)."""
    pop = []
    fits = []
    for _ in range(pop_n):
        if bud.done():
            return
        t = R.rand_tree_ramped(rng)
        pop.append(t)
        fits.append(bud.eval(t))
    while not bud.done():
        new = []
        nfit = []
        if select:
            ie = int(np.argmin(fits))
            new.append(pop[ie])
            nfit.append(fits[ie])
        while len(new) < pop_n and not bud.done():
            if select:
                a = pop[tournament(fits, tour, rng)]
                b = pop[tournament(fits, tour, rng)]
            else:
                a = pop[rng.below(pop_n)]
                b = pop[rng.below(pop_n)]
            if rng.random() < p_cross:
                child = R.tree_crossover(a, b, rng)
            else:
                child = R.tree_mutate_subtree(a, rng)
            new.append(child)
            nfit.append(bud.eval(child))
        if len(new) == pop_n:
            pop, fits = new, nfit


def run_ea_s(bud, rng, select=True, mu=10, lam=10, tour=2):
    """(10 + 10)-Evolution über Schablonen: Turnier 2, ein Kind je Elter durch `s_mutate`, Plus-Selektion der zehn Besten.

    Mit select=False werden die Eltern gleichverteilt gezogen und die Kinder ersetzen die Eltern (Variation ohne Auswahl der Besten)."""
    pop = []
    fits = []
    for _ in range(mu):
        if bud.done():
            return
        ind = R.s_sample(rng)
        pop.append(ind)
        fits.append(bud.eval(ind))
    while not bud.done():
        kids = []
        kfit = []
        for _ in range(lam):
            if bud.done():
                break
            par = pop[tournament(fits, tour, rng)] if select else pop[rng.below(len(pop))]
            ch = R.s_mutate(par, rng)
            kids.append(ch)
            kfit.append(bud.eval(ch))
        if len(kids) < lam:
            break
        if select:
            allp = pop + kids
            allf = fits + kfit
            order = sorted(range(len(allp)), key=lambda i: allf[i])[:mu]     # stabil: bei Gleichstand der ältere
            pop = [allp[i] for i in order]
            fits = [allf[i] for i in order]
        else:
            pop, fits = kids[:mu], kfit[:mu]


# Kürzel: (Raum, Funktion(bud, rng), Anzeigename)
METHODS = {
    "Z-W": ("W", lambda b, r: run_random("W", b, r), "Zufallssuche Gewichte (Z-W)"),
    "11-W": ("W", lambda b, r: run_11("W", b, r), "(1+1)-Suche Gewichte (11-W)"),
    "Z-T": ("T", lambda b, r: run_random("T", b, r), "Zufallssuche Bäume (Z-T)"),
    "11-T": ("T", lambda b, r: run_11("T", b, r), "(1+1)-Suche Bäume (11-T)"),
    "GP-T": ("T", lambda b, r: run_gp(b, r, True), "Population Bäume (GP-T, 30, Turnier 3)"),
    "Z-S": ("S", lambda b, r: run_random("S", b, r), "Zufallssuche Schablonen (Z-S)"),
    "EA-S": ("S", lambda b, r: run_ea_s(b, r, True), "(10 + 10)-Evolution Schablonen (EA-S)"),
}
METHOD_ORDER = ("Z-W", "11-W", "Z-T", "11-T", "GP-T", "Z-S", "EA-S")    # Position = Index im Zufalls-Seed


def run_search(method, sizes, seed, checkpoints=(200, 1000), keep_pool=False):
    """Ein Lauf eines Suchverfahrens auf den Trainingsinstanzen `sizes` bis zur größten Stufe; Rückgabe: das Budget-Objekt."""
    space, fn, _ = METHODS[method]
    bud = Budget(space, sizes, A.lower_bound(sizes), checkpoints, keep_pool)
    fn(bud, SplitMix64(seed))
    return bud


# ---------------------------------------------------------------------------------------------------------------------------------------
# Parametrische Gegner: Gitter nach der Trainingsfitness
# ---------------------------------------------------------------------------------------------------------------------------------------
PR_GRID = tuple(float(x) for x in np.linspace(0.0, 0.4, 21))            # a
PR2_LAM_GRID = tuple(float(x) for x in np.linspace(0.0, 1.0, 8))        # lambda
PR2_M_GRID = tuple(float(x) for x in np.linspace(0.02, 0.4, 8))         # m
GRID_TIE = 1e-12        # ein späterer Gitterpunkt gewinnt nur, wenn er um mehr als das besser ist


def grid_pr(sizes, lb=None):
    """PR(a) auf dem 21-Werte-Gitter: 21 Auswertungen. Rückgabe: (Index des besten a, Liste der Fitnesswerte); Gleichstand: kleinstes a."""
    lb = A.lower_bound(sizes) if lb is None else lb
    fits = [A.fitness(A.simulate(sizes, A.score_pr(a))[0], lb) for a in PR_GRID]
    best = 0
    for j in range(1, len(fits)):
        if fits[j] < fits[best] - GRID_TIE:
            best = j
    return best, fits


def grid_pr2(sizes, lb=None):
    """PR2(lambda, m) auf dem 8-mal-8-Gitter: 64 Auswertungen. Rückgabe: ((i, j) des besten Punkts, Matrix der Fitnesswerte [i][j]).

    Gleichstand: der frühere Punkt in der Reihenfolge lambda außen, m innen (also zuerst lambda = 0, das ist Best Fit)."""
    lb = A.lower_bound(sizes) if lb is None else lb
    fits = [[A.fitness(A.simulate(sizes, A.score_pr2(lam, m))[0], lb) for m in PR2_M_GRID] for lam in PR2_LAM_GRID]
    best = (0, 0)
    for i in range(len(PR2_LAM_GRID)):
        for j in range(len(PR2_M_GRID)):
            if fits[i][j] < fits[best[0]][best[1]] - GRID_TIE:
                best = (i, j)
    return best, fits


def best_fixed(sizes, lb=None):
    """Beste feste Regel unter Best Fit, First Fit, Worst Fit nach der Trainingsfitness: 3 Auswertungen; Gleichstand in dieser Reihenfolge."""
    lb = A.lower_bound(sizes) if lb is None else lb
    fits = {nm: A.fitness(A.run_known(sizes, nm), lb) for nm in ("BF", "FF", "WF")}
    name = min(("BF", "FF", "WF"), key=lambda k: fits[k])
    return name, fits
