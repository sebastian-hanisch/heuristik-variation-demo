"""Ein Schritt der Variation im Raum der Schablonen, mit Angabe der Operatorart, und die Einordnung der Kinder gegen den Elternteil.

Der Operator ist `hva_rules.s_mutate` (Parameter stören 0,5; Vorlage wechseln 0,2; zweiten Term anfügen, ändern oder entfernen 0,3).
`mutate_with_kind` liefert dasselbe Kind aus demselben Zufallsstrom und nennt dazu die Art.
"""

import numpy as np

import hva_algorithm as A
import hva_rules as R
from hva_rng import SplitMix64

KINDS = ("parameter", "vorlage", "zweiter_term")
KIND_LABELS = {"parameter": "Parameter stören", "vorlage": "Vorlage wechseln", "zweiter_term": "zweiten Term anfügen, ändern oder entfernen"}
KIND_PROBS = {"parameter": R.P_S_PARAM, "vorlage": R.P_S_SWAP, "zweiter_term": 1.0 - R.P_S_PARAM - R.P_S_SWAP}
FIT_TIE = 1e-12     # zwei Fitnesswerte gelten als gleich, wenn sie sich um höchstens das unterscheiden

BEST_FIT_IND = (0, (0.0,), -1, (), 0.0)         # S1 mit a = 0: Best Fit


def s1(a):
    """Schablone »Lückenziel« mit Parameter a (Anteil der Kapazität), ohne zweiten Term."""
    return (0, (float(a),), -1, (), 0.0)


def kind_of(u):
    """Operatorart zur ersten Zufallszahl u des Operators."""
    if u < R.P_S_PARAM:
        return "parameter"
    if u < R.P_S_PARAM + R.P_S_SWAP:
        return "vorlage"
    return "zweiter_term"


def mutate_with_kind(ind, rng):
    """(Art, Kind): dasselbe Kind wie `R.s_mutate(ind, rng)` aus demselben Zufallsstrom; die Art folgt aus der ersten Zufallszahl."""
    u = SplitMix64(rng.state).random()      # Vorschau auf die erste Zufallszahl, der Strom selbst bleibt unberührt
    return kind_of(u), R.s_mutate(ind, rng)


def children(parent, n, rng):
    """n Kinder desselben Elternteils: Liste von (Art, Kind)."""
    return [mutate_with_kind(parent, rng) for _ in range(n)]


def classify(parent_bins, child_bins, lb):
    """Ein Kind gegen den Elternteil auf denselben Instanzen: »besser« (kleinere Fitness), »gleich« (dieselben Behälterzahlen auf
    allen Instanzen), »gleichwertig« (andere Behälterzahlen, gleiche Fitness), »schlechter«."""
    parent_bins = np.asarray(parent_bins)
    child_bins = np.asarray(child_bins)
    if (parent_bins == child_bins).all():
        return "gleich"
    d = A.fitness(child_bins, lb) - A.fitness(parent_bins, lb)
    if d < -FIT_TIE:
        return "besser"
    if d > FIT_TIE:
        return "schlechter"
    return "gleichwertig"


CLASSES = ("besser", "gleich", "gleichwertig", "schlechter")


def classify_children(parent_bins, child_bins_list, lb):
    """Zählung der vier Klassen über alle Kinder."""
    out = {c: 0 for c in CLASSES}
    for cb in child_bins_list:
        out[classify(parent_bins, cb, lb)] += 1
    return out


def operator_table(kinds, classes):
    """Je Operatorart: Zahl der Kinder und Zählung der Klassen. `kinds` und `classes` sind gleich lange Listen."""
    out = {k: dict(n=0, **{c: 0 for c in CLASSES}) for k in KINDS}
    for k, c in zip(kinds, classes):
        out[k]["n"] += 1
        out[k][c] += 1
    return out
