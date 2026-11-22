"""Instanzen: drei erzeugte Familien (plus die Anzeigefamilie »perfekt«), Zufalls-Seeds je Zweck, Handinstanzen.

Zufall nur über SplitMix64 (`hva_rng.py`); die Normalzahlen sind auf jeder Plattform bitgleich. Ein Zufalls-Seed entsteht aus einem Zweck
und ganzen Zahlen über eine SplitMix64-Kette (`seed_for`), nie über `hash()`.
"""

import numpy as np

from hva_algorithm import CAPACITY
from hva_rng import MASK, SplitMix64

# Familie: (Art, Parameter ...)
FAMILIES = {
    "gl_20_70": ("uni", 20, 70),              # gleichverteilt auf den ganzen Zahlen 20 bis 70
    "clu3": ("clu", (20, 35, 55), 2.0),       # drei Häufungen (je gleich wahrscheinlich), Normalstreuung 2
    "stark": ("nor", 40.0, 22.0),             # Normal(40, 22), gerundet und auf 1..100 begrenzt
    "perfekt": ("plant",),                    # volle Behälter zerschnitten: Optimum = Zahl der Behälter (nur Anzeige)
}
FAMILY_ORDER = ("gl_20_70", "clu3", "stark", "perfekt")   # Position = Index im Zufalls-Seed
TRAIN_FAMILIES = ("gl_20_70", "clu3", "stark")

SEED_TRAIN, SEED_TEST, SEED_METHOD, SEED_ORACLE = 11, 12, 13, 14   # Zwecke der Zufalls-Seeds

N_TRAIN, K_TRAIN, K_TEST = 60, 20, 200

HAND = {
    "I*": (60, 35, 55, 65, 30, 65, 45, 10),
    "A": (55, 50, 60, 15, 40, 45, 25, 10),
    "B": (50, 60, 35, 15, 45, 20, 20, 40),
    "U": (15, 55, 70, 15, 40, 15, 30, 40),
}


def seed_for(zweck, *ints):
    """Zufalls-Seed aus einem Zweck und ganzen Zahlen (SplitMix64-Kette): getrennte Zwecke geben getrennte Ströme."""
    h = SplitMix64(int(zweck)).next_u64()
    for v in ints:
        h = SplitMix64((h ^ ((int(v) + 1) * 0x9E3779B97F4A7C15)) & MASK).next_u64()
    return h


def _round_clip(v, capacity):
    """Auf die nächste ganze Zahl runden (Hälften zur geraden Zahl) und auf 1..Kapazität begrenzen."""
    return min(capacity, max(1, int(round(v))))


def _plant_instance(n, rng, capacity):
    """Zerschneidet volle Behälter in je 2 bis 5 Stücke, bis genau n Gegenstände da sind; mischt die Reihenfolge. Bliebe sonst ein einzelnes Reststück, bekommt der letzte Behälter alle Reststücke (dann bis zu 6)."""
    items = []
    bins = 0
    left = n
    while left > 0:
        k = min(2 + rng.below(4), left)
        if left - k == 1:           # kein einzelnes Reststück: der letzte Behälter bekommt es dazu
            k = left
        if k == 1:
            pieces = [capacity]
        else:
            pool = list(range(1, capacity))
            for i in range(k - 1):  # k - 1 verschiedene Schnittstellen ziehen
                j = i + rng.below(len(pool) - i)
                pool[i], pool[j] = pool[j], pool[i]
            cuts = sorted(pool[: k - 1])
            edges = [0] + cuts + [capacity]
            pieces = [edges[i + 1] - edges[i] for i in range(k)]
        items.extend(pieces)
        left -= len(pieces)
        bins += 1
    rng.shuffle(items)
    return items, bins


def gen_sizes(familie, n, K, rng, capacity=CAPACITY):
    """Ganzzahlige Größen (K, n) in 1..Kapazität; für »perfekt« zusätzlich das exakte Optimum (K,), sonst None.

    Zeile für Zeile, Gegenstand für Gegenstand gezogen: die ersten K' Zeilen hängen nicht von K ab."""
    spec = FAMILIES[familie]
    kind = spec[0]
    out = np.empty((K, n), dtype=np.int64)
    opt = None
    if kind == "uni":
        lo, hi = spec[1], spec[2]
        for k in range(K):
            for t in range(n):
                out[k, t] = lo + rng.below(hi - lo + 1)
    elif kind == "clu":
        centers, sd = spec[1], spec[2]
        for k in range(K):
            for t in range(n):
                c = centers[rng.below(len(centers))]
                out[k, t] = _round_clip(c + sd * rng.normal(), capacity)
    elif kind == "nor":
        mu, sd = spec[1], spec[2]
        for k in range(K):
            for t in range(n):
                out[k, t] = _round_clip(mu + sd * rng.normal(), capacity)
    elif kind == "plant":
        opt = np.empty(K, dtype=np.int64)
        for k in range(K):
            items, bins = _plant_instance(n, rng, capacity)
            out[k] = items
            opt[k] = bins
    else:
        raise ValueError(familie)
    return out, opt


def training_set(familie, lauf, K=K_TRAIN, n=N_TRAIN):
    """Trainingsinstanzen eines Laufs (Zweck 11 mit Familie, Lauf, K, n)."""
    rng = SplitMix64(seed_for(SEED_TRAIN, FAMILY_ORDER.index(familie), lauf, K, n))
    return gen_sizes(familie, n, K, rng)[0]


def test_set(familie, n=N_TRAIN, test_seed=0, K=K_TEST, with_opt=False):
    """Frische Testinstanzen (Zweck 12 mit Familie, n, Test-Zufalls-Seed); fest je (Familie, n, Test-Zufalls-Seed)."""
    rng = SplitMix64(seed_for(SEED_TEST, FAMILY_ORDER.index(familie), n, test_seed))
    sizes, opt = gen_sizes(familie, n, K, rng)
    return (sizes, opt) if with_opt else sizes


test_set.__test__ = False   # kein pytest-Test, auch wenn der Name so beginnt
