"""Heuristik-Mengen: Güte einer Menge, drei Arten, eine Menge zu bilden, und drei Arten, im Einsatz daraus zu wählen.

Eine Tabelle `matrix` hat die Form (Regeln, Instanzen) und enthält Behälterzahlen (ganze Zahlen). Eine Menge ist eine Liste von
Zeilenindizes. Jede Funktion ist eine Einheit mit expliziten Ein- und Ausgaben; Zufall nur über den übergebenen Generator.
"""

import numpy as np

from hva_algorithm import snap

PORTFOLIO = ("BF", "FF", "WF")      # feste Menge bekannter Regeln, Reihenfolge = Index bei Gleichständen
N_FIRST = 10                        # die Merkmalswahl sieht nur die ersten zehn Gegenstände
CHOICES = ("orakel", "merkmal", "fest")


def set_min(matrix, members):
    """Je Instanz die kleinste Behälterzahl über die Mitglieder der Menge."""
    return np.asarray(matrix)[list(members)].min(axis=0)


def set_cost(matrix, members):
    """Güte einer Menge: Summe über die Instanzen des Minimums über die Mitglieder (ganze Zahl, kleiner ist besser)."""
    return int(set_min(matrix, members).sum())


def greedy_set(matrix, fits, k):
    """Gierige Menge: wiederholt die Regel aufnehmen, die die Summe der Instanz-Minima am stärksten senkt.

    Gleichstand: die Regel mit der besseren Einzelfitness, danach der kleinste Index. Die Fitness wird dafür auf 12 Stellen gerundet."""
    matrix = np.asarray(matrix, dtype=np.int64)
    fits = snap(fits)
    chosen = []
    big = 10 ** 9
    cur = np.full(matrix.shape[1], big, dtype=np.int64)
    for _ in range(min(k, matrix.shape[0])):
        tot = np.minimum(matrix, cur[None, :]).sum(axis=1)
        tot[chosen] = np.iinfo(np.int64).max
        cand = np.flatnonzero(tot == tot.min())
        best = int(cand[np.argmin(fits[cand])])     # argmin nimmt bei gleicher Fitness den kleinsten Index
        chosen.append(best)
        cur = np.minimum(cur, matrix[best])
    return chosen


def topk_set(fits, k):
    """Die k Regeln mit der kleinsten Einzelfitness (auf 12 Stellen gerundet); Gleichstand: kleinster Index."""
    fits = snap(fits)
    return [int(j) for j in np.argsort(fits, kind="stable")[:k]]


def random_set(n_pool, k, rng):
    """k verschiedene Indizes aus 0..n_pool-1, gleichverteilt (Anfang einer Fisher-Yates-Mischung); weniger, wenn der Pool kleiner ist."""
    idx = list(range(n_pool))
    k = min(k, n_pool)
    for i in range(k):
        j = i + rng.below(n_pool - i)
        idx[i], idx[j] = idx[j], idx[i]
    return idx[:k]


def random_sets(n_pool, k, rng, n):
    """n Zufallsmengen der Größe k."""
    return [random_set(n_pool, k, rng) for _ in range(n)]


# ---------------------------------------------------------------------------------------------------------------------------------------
# Wahl im Einsatz. `members_train` (k, K) und `members_test` (k, T): Behälterzahlen der Mitglieder der Menge.
# ---------------------------------------------------------------------------------------------------------------------------------------
def oracle_choice(members_test):
    """Orakelwahl: alle Mitglieder laufen, je Instanz zählt die beste Packung. Rückgabe: Behälterzahlen (T,)."""
    return np.asarray(members_test).min(axis=0)


def fixed_member(members_train):
    """Feste Wahl: das Mitglied mit der kleinsten Summe der Behälterzahlen auf dem Training (Gleichstand: kleinster Index)."""
    return int(np.argmin(np.asarray(members_train, dtype=np.int64).sum(axis=1)))


def fixed_choice(members_train, members_test):
    return np.asarray(members_test)[fixed_member(members_train)]


def labels(members_train):
    """Etikett je Trainingsinstanz: das Mitglied mit der kleinsten Behälterzahl dort (Gleichstand: kleinster Index)."""
    return np.asarray(members_train).argmin(axis=0)


def first_features(sizes, m=N_FIRST, with_std=True):
    """Merkmale einer Instanz aus den ersten m Gegenständen: Mittel und (mit with_std) Streuung der Größen. Form (K, 1 oder 2)."""
    f = np.asarray(sizes, dtype=np.float64)[:, :m]
    cols = [f.mean(axis=1)]
    if with_std:
        cols.append(f.std(axis=1))
    return np.stack(cols, axis=1)


def nearest_neighbor(train_feat, test_feat):
    """1-nächster Nachbar je Testinstanz: Merkmale mit Mittel und Streuung der Trainingsinstanzen normiert, quadrierter Abstand
    (auf 12 Stellen gerundet), Gleichstand: kleinster Index."""
    train_feat = np.asarray(train_feat, dtype=np.float64)
    test_feat = np.asarray(test_feat, dtype=np.float64)
    mu = train_feat.mean(axis=0)
    sd = train_feat.std(axis=0) + 1e-9
    a = (train_feat - mu) / sd
    b = (test_feat - mu) / sd
    dist = snap(((b[:, None, :] - a[None, :, :]) ** 2).sum(axis=2))
    return dist.argmin(axis=1)


def feature_pick(train_sizes, members_train, test_sizes, m=N_FIRST, with_std=True):
    """Merkmalswahl: je Testinstanz das Etikett des nächsten Nachbarn unter den Trainingsinstanzen. Rückgabe: Mitgliedsindex (T,)."""
    nn = nearest_neighbor(first_features(train_sizes, m, with_std), first_features(test_sizes, m, with_std))
    return labels(members_train)[nn]


def feature_choice(train_sizes, members_train, test_sizes, members_test, m=N_FIRST, with_std=True):
    """Behälterzahlen (T,) der Merkmalswahl: ein Lauf je Instanz mit der gewählten Regel."""
    pick = feature_pick(train_sizes, members_train, test_sizes, m, with_std)
    mt = np.asarray(members_test)
    return mt[pick, np.arange(mt.shape[1])]


def runs_per_instance(wahl, k):
    """Platzierungsläufe je Instanz im Einsatz: k bei der Orakelwahl, sonst 1."""
    return int(k) if wahl == "orakel" else 1
