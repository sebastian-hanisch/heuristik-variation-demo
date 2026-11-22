"""Konstanten: Maßstab der Messung, Schwellen der Verdikte, Regler-Grenzen, Namen, Farben und Presets."""

import hva_arms as AR
import hva_scenario as S

# --- Maßstab der Messung ---------------------------------------------------------------------------------------------------------------
CAPACITY = S.CAPACITY
N_ITEMS = S.N_TRAIN            # 60 Gegenstände je Instanz
K_TRAIN = S.K_TRAIN            # 20 Trainingsinstanzen
K_TRAIN_BIG = 100              # größere Trainingsmenge (Experiment 5)
K_TEST = S.K_TEST              # 200 frische Testinstanzen
N_RUNS_FROZEN = 20             # eingefrorene Läufe je Zelle
N_RUNS_BIG = 10                # eingefrorene Läufe mit 100 Trainingsinstanzen
CHECKPOINTS = tuple(range(100, 1001, 100))     # Stufen, an denen der Bestwert festgehalten wird
FROZEN_BUDGET = 1000           # die Pools der eingefrorenen Mengen stammen aus Läufen mit 1 000 Auswertungen
N_RANDOM_SETS = 5              # Zufallsmengen je Lauf (das Mittel je Lauf zählt)
FROZEN_CHILDREN = 300          # Kinder je Elternteil und Trainingssatz in der eingefrorenen Messung der Variation

# --- Schwellen der Verdikte ------------------------------------------------------------------------------------------------------------
TOL_TEST = 0.2                 # Toleranz der Entscheidungsregel, Prozentpunkte
TOL_TRAIN = 0.1                # Anzeigeschwelle »Trainingsvorsprung«, Prozentpunkte (Konvention, nicht aus einer Messung abgeleitet)
K_MIN = 15                     # mindestens so viele von 20 Läufen müssen auf derselben Seite liegen
ANTEIL_SELTEN = 0.05           # »selten«: unter 5 % der Kinder besser als der Elternteil (Konvention dieser Demo)
HALB = 0.5                     # »weniger als die Hälfte der Gewinner auf dem Training gewinnt auch auf frischen Instanzen«

# --- Regler ----------------------------------------------------------------------------------------------------------------------------
FAMILY_LABELS = {
    "gl_20_70": "gleichverteilt 20 bis 70",
    "clu3": "drei Häufungen um 20, 35 und 55",
    "stark": "breit gestreut (Normal 40, Streuung 22)",
}
FAMILIES = tuple(FAMILY_LABELS)
DEFAULT_FAMILY = "gl_20_70"
ARMS = AR.ARM_ORDER
ARM_LABELS = {a: AR.ARMS[a][2] for a in ARMS}
DEFAULT_ARM = "EA-S"
SPACE_LABELS = {"T": "Ausdrucksbäume", "S": "Schablonen"}
POOL_ARMS = ("EA-S", "GP-T")   # Arme mit Auswahl der Besten: ihre Pools sind die Quelle der Mengen in den Befunden
BIG_ARMS = ("Z-S", "EA-S", "GP-T")
BIG_FAMILIES = ("gl_20_70", "stark")

BUDGET_MIN, BUDGET_MAX, BUDGET_STEP, DEFAULT_BUDGET = 100, 1000, 100, 400      # je Auswertung ein Simulatorlauf über 20 mal 60 Gegenstände, linear im Budget
THREE_ARM_BUDGET_MAX = 400     # der Knopf »drei Schablonen-Arme« rechnet höchstens mit diesem Budget
K_OPTIONS = (1, 3, 5)
DEFAULT_K = 3
SET_TYPES = ("greedy", "topk", "zufall")
SET_LABELS = {"greedy": "greedy (Vielfalt)", "topk": "die k besten (topk)", "zufall": "k beliebige (Zufall)"}
SET_TYPES_FROZEN = ("greedy", "topk", "zufall", "bfplus")
SET_LABELS_LONG = {
    "greedy": "greedy: nach Vielfalt auf dem Training gewählt",
    "topk": "topk: die k nach Trainingsfitness besten",
    "zufall": "k beliebige Regeln des Pools",
    "bfplus": "Best Fit plus k − 1 beliebige Regeln des Pools",
    "portfolio": "Best Fit, First Fit, Worst Fit",
}
DEFAULT_SET = "greedy"
CHOICES = ("orakel", "merkmal", "fest")
CHOICE_LABELS = {
    "orakel": "Orakel (alle k laufen, die beste Packung zählt)",
    "merkmal": "Merkmale (1 Lauf)",
    "fest": "fest (1 Lauf)",
}
CHOICE_SHORT = {"orakel": "Orakelwahl", "merkmal": "Merkmalswahl", "fest": "feste Wahl"}
DEFAULT_CHOICE = "orakel"
TESTSETS = ("same60", "n30", "n120", "fremd1", "fremd2")
TESTSET_SPEC = {"same60": (60, 200), "n30": (30, 200), "n120": (120, 100)}      # (Gegenstände, Instanzen) in der Trainingsfamilie
DEFAULT_TESTSET = "same60"
PARENTS = ("bf", "s1_20", "frisch")
PARENT_LABELS = {"bf": "Best Fit (S1, a = 0)", "s1_20": "Lückenziel a = 0,20", "frisch": "frische Schablone"}
DEFAULT_PARENT = "bf"
KINDER_MIN, KINDER_MAX, KINDER_STEP, DEFAULT_KINDER = 100, 600, 100, 300           # je Kind ein Simulatorlauf über 20 mal 60 Gegenstände
HAND_STEP_MIN, HAND_STEP_MAX, DEFAULT_HAND_STEP = 0, 3, 3
TEST_SEED_MIN, TEST_SEED_MAX, DEFAULT_TEST_SEED = 0, 9, 0
RUN_SEED_MIN, RUN_SEED_MAX, DEFAULT_RUN_SEED = 0, 999, 0


def other_families(familie):
    """Die beiden anderen Familien in fester Reihenfolge (»fremde Familie 1« und »2«)."""
    return tuple(f for f in FAMILIES if f != familie)


def testset_label(name, familie):
    if name == "same60":
        return "gleiche Familie, 60 Gegenstände"
    if name == "n30":
        return "gleiche Familie, 30 Gegenstände"
    if name == "n120":
        return "gleiche Familie, 120 Gegenstände"
    other = other_families(familie)[0 if name == "fremd1" else 1]
    return "fremde Familie %s: %s" % ("1" if name == "fremd1" else "2", FAMILY_LABELS[other])


TESTSET_LABELS = {
    "same60": "gleiche Familie, 60 Gegenstände", "n30": "gleiche Familie, 30 Gegenstände", "n120": "gleiche Familie, 120 Gegenstände",
    "fremd1": "fremde Familie 1", "fremd2": "fremde Familie 2",
}

# --- Farben ----------------------------------------------------------------------------------------------------------------------------
COLOR_BF = "#2F6B65"
COLOR_RULE = "#d1495b"
COLOR_LINE = "#4c78a8"
COLOR_OTHER = "#edae49"
COLOR_GREY = "rgba(120,120,120,0.55)"
COLOR_BARS = ("#4c78a8", "#2F6B65", "#edae49", "#d1495b", "#8d6a9f")

# --- Presets ---------------------------------------------------------------------------------------------------------------------------
DEFAULTS = {
    "familie": DEFAULT_FAMILY, "arm": DEFAULT_ARM, "budget": DEFAULT_BUDGET, "k": DEFAULT_K, "menge": DEFAULT_SET, "wahl": DEFAULT_CHOICE,
    "testmenge": DEFAULT_TESTSET, "elternteil": DEFAULT_PARENT, "kinder": DEFAULT_KINDER, "hand_schritt": DEFAULT_HAND_STEP,
    "test_seed": DEFAULT_TEST_SEED, "lauf_seed": DEFAULT_RUN_SEED,
}


def _preset(**kw):
    out = dict(DEFAULTS)
    out.update(kw)
    return out


PRESETS = {
    "Eine Menge von Hand": _preset(),
    "Variation an Best Fit": _preset(hand_schritt=0, budget=200),
    "Variation und Auswahl": _preset(budget=600),
    "Bäume statt Vorlagen": _preset(arm="GP-T", budget=600),
    "Menge mit Orakelwahl": _preset(budget=1000, lauf_seed=1),
    "Menge ohne Orakel": _preset(budget=1000, lauf_seed=1, wahl="merkmal"),
    "Häufungen: wenig Vielfalt": _preset(familie="clu3", budget=1000),
}
PRESET_HELP = {
    "Eine Menge von Hand": "Die Voreinstellung: das Handbeispiel mit vier Regeln bis zum Einsatz auf den zwei Testinstanzen, darunter ein Lauf mit 400 Auswertungen.",
    "Variation an Best Fit": "Elternteil Best Fit, 300 Kinder, gleichverteilte Familie, dazu ein kurzer Lauf mit 200 Auswertungen; das Handbeispiel zeigt nur die Tabelle. Experiment 1 zählt, wie viele Kinder besser sind.",
    "Variation und Auswahl": "Schablonen mit Variation und Auswahl der Besten, 600 Auswertungen, ein Lauf. Die 20 eingefrorenen Läufe stehen im Vergleich darunter.",
    "Bäume statt Vorlagen": "Ausdrucksbäume mit Teilbaum-Kreuzung und Auswahl der Besten, 600 Auswertungen, ein Lauf.",
    "Menge mit Orakelwahl": "Pool aus 1 000 Auswertungen (Zufalls-Seed der Läufe 1), daraus drei Regeln nach Vielfalt; je Instanz zählt die beste der drei Packungen.",
    "Menge ohne Orakel": "Dieselbe Menge, aber eine Regel je Instanz nach den ersten zehn Gegenständen gewählt: ein Lauf je Instanz.",
    "Häufungen: wenig Vielfalt": "Familie mit drei Häufungen: First Fit und Best Fit brauchen auf den meisten Instanzen gleich viele Behälter, und die Menge gewinnt weniger als auf der gleichverteilten Familie.",
}
