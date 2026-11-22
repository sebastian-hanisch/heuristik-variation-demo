"""Die sechs Arme der Erzeugung: frisch ziehen, Variation ohne Auswahl der Besten, Variation mit Auswahl, je für Schablonen und Bäume.

Alle Arme laufen über dasselbe `Budget` (`hva_search.py`): gleiche Zahl der Auswertungen, gleiche Zählung, gleiche Trainingsinstanzen.
Der Unterschied zwischen »ZW-S« und »EA-S« (und zwischen »GP0-T« und »GP-T«) ist nur der Schalter `select`: ohne ihn werden die Eltern
gleichverteilt gezogen, und die Kinder ersetzen die Eltern.
"""

import hva_algorithm as A
import hva_scenario as S
import hva_search as H
from hva_rng import SplitMix64

SEED_ARM = 21       # Zweck des Zufalls-Seeds der beiden Arme ohne Auswahl der Besten
SEED_SETS = 22      # Zufallsmengen
SEED_CHILDREN = 23  # Kinder der Variation
SEED_PARENT = 24    # frische Schablone als Elternteil

# Kürzel: (Raum, Funktion(bud, rng), Anzeigename, Kurzbeschreibung)
ARMS = {
    "Z-S": ("S", lambda b, r: H.run_random("S", b, r), "Schablonen: frisch ziehen (Z-S)",
            "Jede Auswertung zieht eine neue Schablone mit zufälligen Parametern; nichts wird verändert, nichts behalten."),
    "ZW-S": ("S", lambda b, r: H.run_ea_s(b, r, False), "Schablonen: Variation ohne Auswahl (ZW-S)",
             "Zehn Schablonen; je Runde zehn Kinder durch den Vorlagen-Operator, Eltern gleichverteilt gezogen, die Kinder ersetzen alle Eltern."),
    "EA-S": ("S", lambda b, r: H.run_ea_s(b, r, True), "Schablonen: Variation und Auswahl (EA-S)",
             "Wie ZW-S, aber die Eltern gewinnen ein Turnier der Größe 2, und aus Eltern und Kindern bleiben die zehn Besten."),
    "Z-T": ("T", lambda b, r: H.run_random("T", b, r), "Bäume: frisch ziehen (Z-T)",
            "Jede Auswertung zieht einen neuen Ausdrucksbaum der Tiefe 2 bis 4."),
    "GP0-T": ("T", lambda b, r: H.run_gp(b, r, False), "Bäume: Kreuzung ohne Auswahl (GP0-T)",
              "30 Bäume; Teilbaum-Kreuzung (0,8) oder Teilbaum-Mutation (0,2), Eltern gleichverteilt gezogen, keine Elite."),
    "GP-T": ("T", lambda b, r: H.run_gp(b, r, True), "Bäume: Kreuzung und Auswahl (GP-T)",
             "Wie GP0-T, aber die Eltern gewinnen ein Turnier der Größe 3, und der Beste bleibt erhalten."),
}
ARM_ORDER = ("Z-S", "ZW-S", "EA-S", "Z-T", "GP0-T", "GP-T")     # Position = Index im Zufalls-Seed der Arme ohne Auswahl

# Vergleiche (A, B, Art): Delta = B minus A, negativ = B besser
COMPARISONS = (
    ("Z-S", "ZW-S", "Variation allein"),
    ("ZW-S", "EA-S", "Auswahl der Besten"),
    ("Z-S", "EA-S", "beides"),
    ("Z-T", "GP0-T", "Variation allein"),
    ("GP0-T", "GP-T", "Auswahl der Besten"),
    ("Z-T", "GP-T", "beides"),
    ("EA-S", "GP-T", "Operator gegen Baumkreuzung"),
)


def arm_space(arm):
    return ARMS[arm][0]


def arm_seed(arm, familie, lauf):
    """Zufalls-Seed der Suche. Die vier Arme, die es auch in der Kopierquelle gibt (Z-S, EA-S, Z-T, GP-T), benutzen deren Zweck 13
    und Verfahrensindex: derselbe Lauf wie dort. ZW-S und GP0-T haben den eigenen Zweck 21. Der Seed hängt nicht an der Zahl der
    Trainingsinstanzen."""
    fi = S.FAMILY_ORDER.index(familie)
    if arm in H.METHOD_ORDER:
        return S.seed_for(S.SEED_METHOD, H.METHOD_ORDER.index(arm), fi, lauf)
    return S.seed_for(SEED_ARM, ARM_ORDER.index(arm), fi, lauf)


def run_arm(arm, sizes, seed, checkpoints=(200, 1000), keep_pool=True):
    """Ein Lauf eines Arms auf den Trainingsinstanzen `sizes` bis zur größten Stufe; Rückgabe: das Budget-Objekt (mit Pool)."""
    space, fn = ARMS[arm][0], ARMS[arm][1]
    bud = H.Budget(space, sizes, A.lower_bound(sizes), checkpoints, keep_pool)
    fn(bud, SplitMix64(seed))
    return bud
