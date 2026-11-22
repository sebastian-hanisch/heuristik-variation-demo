"""Auswertung: Zahlenformat, Entscheidungsregel, Tabellen aus den eingefrorenen Messreihen und die Verdikte als reine Funktionen.

Delta heißt überall: Überschuss(Regel oder Menge) minus Überschuss(Vergleichsgröße) in Prozentpunkten, positiv = schlechter.
Vergleichsgröße ist Best Fit, wenn nichts anderes dabeisteht; bei Mengen zusätzlich die beste Einzelregel desselben Laufs.
"""

import json
import math
from pathlib import Path

import hva_arms as AR
import hva_constants as C
import hva_variation as V

PRECOMPUTED = Path(__file__).resolve().parent / "hva_precomputed.json"
LABELS = ("besser", "gleich", "schlechter")
WAHL_INDEX = {w: i for i, w in enumerate(C.CHOICES)}


# ---------------------------------------------------------------------------------------------------------------------------------------
# Zahlenformat
# ---------------------------------------------------------------------------------------------------------------------------------------
def de(x, stellen=2):
    """Zahl mit Dezimalkomma; eine gerundete Null hat kein Vorzeichen."""
    v = round(float(x), stellen)
    if v == 0:
        v = 0.0
    return f"{v:.{stellen}f}".replace(".", ",").replace("-", "−")


def de_pm(x, stellen=2):
    """Wie `de`, mit Pluszeichen bei positiven Werten (für Differenzen)."""
    v = round(float(x), stellen)
    return ("+" if v > 0 else "") + de(v, stellen)


def ganz(n):
    """Ganze Zahl mit schmalem Leerzeichen als Tausendertrenner."""
    return f"{int(n):,}".replace(",", " ")


def pct(k, n, stellen=1):
    """Anteil k von n in Prozent als Text; bei n = 0 ein Strich statt einer Division durch null."""
    return "–" if n == 0 else de(100.0 * k / n, stellen)


# ---------------------------------------------------------------------------------------------------------------------------------------
# Entscheidungsregel
# ---------------------------------------------------------------------------------------------------------------------------------------
def median(values):
    v = sorted(float(x) for x in values)
    n = len(v)
    return (v[n // 2 - 1] + v[n // 2]) / 2.0 if n % 2 == 0 else v[n // 2]


def urteil(deltas, tol=C.TOL_TEST, k_min=C.K_MIN):
    """Entscheidungsregel: »besser«, wenn der Median <= -tol ist und mindestens k_min Läufe ein Delta < 0 haben; »schlechter«
    spiegelbildlich; sonst »gleich«. Nullen zählen für keine Seite. Rückgabe: (Urteil, Median, Läufe < 0, Läufe > 0)."""
    med = median(deltas)
    neg = sum(1 for x in deltas if x < 0)
    pos = sum(1 for x in deltas if x > 0)
    if med <= -tol and neg >= k_min:
        return "besser", med, neg, pos
    if med >= tol and pos >= k_min:
        return "schlechter", med, neg, pos
    return "gleich", med, neg, pos


def sign_test_p(k, n):
    """Einseitiger Vorzeichentest: Wahrscheinlichkeit für mindestens k von n Läufen auf einer Seite, wenn beide Seiten gleich wahrscheinlich sind."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def bonferroni_k(n, cells, alpha=0.05):
    """Kleinste Zahl von Läufen auf einer Seite, mit der der einseitige Vorzeichentest auch nach Bonferroni-Ausgleich über `cells`
    Zellen unter alpha bleibt; None, wenn selbst n von n nicht reicht."""
    for k in range(n + 1):
        if sign_test_p(k, n) <= alpha / cells:
            return k
    return None


# ---------------------------------------------------------------------------------------------------------------------------------------
# Eingefrorene Messreihen: Lader und Tabellen
# ---------------------------------------------------------------------------------------------------------------------------------------
def load_precomputed(path=PRECOMPUTED):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def cell_deltas(runs, budget=None, which="test"):
    """Delta je Lauf einer Zelle gegen Best Fit, bei `budget` (eine Stufe) oder am Ende des Laufs."""
    if budget is None:
        return [r[which] for r in runs]
    return [next(c[which] for c in r["cps"] if c["n"] == budget) for r in runs]


def arm_table(pre, budget=C.FROZEN_BUDGET):
    """Die 18 Zellen Arm mal Familie gegen Best Fit: Median Training, Median Test, Zählungen, Urteil."""
    rows = []
    for fam in C.FAMILIES:
        for arm in C.ARMS:
            runs = pre["m1"][fam][arm]
            lab, med, neg, pos = urteil(cell_deltas(runs, budget))
            tr = cell_deltas(runs, budget, "train")
            rows.append(dict(familie=fam, arm=arm, train=median(tr), train_neg=sum(1 for x in tr if x < 0), test=med, neg=neg, pos=pos,
                             urteil=lab, n=len(runs)))
    return rows


def comparison_rows(pre, familie, budget=C.FROZEN_BUDGET):
    """Die sieben Vergleiche B gegen A einer Familie, gepaart je Lauf: Delta = B minus A, negativ = B besser. Je Vergleich Urteil,
    Median und Zählungen für Training und Test."""
    rows = []
    for a, b, art in AR.COMPARISONS:
        ra, rb = pre["m1"][familie][a], pre["m1"][familie][b]
        out = dict(a=a, b=b, art=art, n=len(ra))
        for which in ("train", "test"):
            da, db = cell_deltas(ra, budget, which), cell_deltas(rb, budget, which)
            lab, med, neg, pos = urteil([y - x for x, y in zip(da, db)])
            out[which] = dict(urteil=lab, median=med, neg=neg, pos=pos)
        rows.append(out)
    return rows


def count_labels(rows, which=None):
    """Zahl der Zeilen je Urteil (bei Vergleichszeilen mit `which` = »train« oder »test«)."""
    labs = [r[which]["urteil"] if which else r["urteil"] for r in rows]
    return {lab: labs.count(lab) for lab in LABELS}


def set_deltas(pre, familie, arm, typ, k, wahl, testmenge="same60", ref="bf", block="m1"):
    """Delta je Lauf einer Menge: gegen Best Fit (ref = »bf«) oder gegen die beste Einzelregel desselben Laufs (ref = »regel«).

    None, wenn die Zelle nicht gemessen ist (zum Beispiel Häufungen mit 100 Trainingsinstanzen)."""
    try:
        runs = pre[block][familie][arm]
        out = []
        for r in runs:
            d = r["sets"][typ][str(k)]["T"][testmenge][WAHL_INDEX[wahl]]
            out.append(d - r["ref"][testmenge] if ref == "regel" else d)
        return out
    except KeyError:
        return None


def set_train_deltas(pre, familie, arm, typ, k, block="m1"):
    """Trainings-Delta der Orakelwahl einer Menge gegen Best Fit, je Lauf."""
    return [r["sets"][typ][str(k)]["train"] for r in pre[block][familie][arm]]


def portfolio_deltas(pre, familie, wahl="orakel", testmenge="same60", block="portfolio"):
    """Delta der festen Menge Best Fit, First Fit, Worst Fit gegen Best Fit, je Lauf (die Orakelwahl hängt nicht am Lauf)."""
    try:
        return [r["T"][testmenge][WAHL_INDEX[wahl]] for r in pre[block][familie]]
    except KeyError:
        return None


def cell(deltas):
    """Zusammenfassung einer Zelle: Urteil (nur bei 20 Läufen), Median, Läufe < 0, Läufe > 0, Zahl der Läufe; None bleibt None."""
    if deltas is None:
        return None
    lab, med, neg, pos = urteil(deltas)
    return dict(urteil=lab if len(deltas) == C.N_RUNS_FROZEN else None, median=med, neg=neg, pos=pos, n=len(deltas))


def variation_cell(pre, familie, parent):
    """Summe über die 20 Trainingssätze: Kinder, Klassen, auf dem Test ebenfalls besser, besser als Best Fit, Tabelle nach Operatorart."""
    recs = pre["m4"][familie][parent]
    out = dict(n=sum(r["n"] for r in recs), better_test=sum(r["bt"] for r in recs), better_bf=sum(r["bbf"] for r in recs),
               parent_worse=sum(1 for r in recs if r["pw"]), saetze=len(recs))
    for i, c in enumerate(V.CLASSES):
        out[c] = sum(r["c"][i] for r in recs)
    out["art"] = {k: dict(n=sum(r["art"][k][0] for r in recs), **{c: sum(r["art"][k][i + 1] for r in recs) for i, c in enumerate(V.CLASSES)})
                  for k in V.KINDS}
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------
# Verdikte: reine Funktionen (Art, Text); die App ruft getattr(st, Art)(Text)
# ---------------------------------------------------------------------------------------------------------------------------------------
def verdict_hand(bf, orakel, merkmal, fest, topk_orakel, k, falsch):
    """Handbeispiel: Behälter auf den zwei Testinstanzen. `falsch` = Zahl der Testinstanzen, auf denen die Merkmalswahl die schlechtere Regel nimmt."""
    text = (f"Auf den zwei Testinstanzen braucht Best Fit {bf} Behälter; die Orakelwahl über die gierig gewählte Menge {orakel} "
            f"({k} Läufe je Instanz), die Merkmalswahl {merkmal} (1 Lauf je Instanz, {falsch} von 2 Mal die schlechtere Regel), die feste Wahl {fest}. ")
    if topk_orakel >= bf:
        text += (f"Die Menge aus den {k} Regeln mit der kleinsten Trainingssumme bringt auch mit Orakelwahl nichts ({topk_orakel} Behälter): "
                 "ihre beiden Regeln packen die vier Trainingsinstanzen mit denselben Behälterzahlen.")
    else:
        text += f"Die Menge aus den {k} Regeln mit der kleinsten Trainingssumme braucht mit Orakelwahl {topk_orakel} Behälter."
    return "info", text


def verdict_variation(parent_worse_bf, n, besser, gleich, schlechter, besser_test, gleichwertig=0, k_train=C.K_TRAIN,
                      selten=C.ANTEIL_SELTEN, halb=C.HALB):
    """Experiment 1: Kinder eines Elternteils auf dem Training, die besseren auch auf den Testinstanzen."""
    rest = f"; {gleichwertig} weitere sind bei anderen Behälterzahlen gleich gut" if gleichwertig else ""
    if n == 0:
        return "info", "Keine Kinder gerechnet."
    zahlen = (f"Von {ganz(n)} Kindern des Elternteils sind {ganz(besser)} ({pct(besser, n)} %) auf dem Training besser, {ganz(gleich)} ({pct(gleich, n)} %) "
              f"packen alle {k_train} Trainingsinstanzen mit denselben Behälterzahlen, {ganz(schlechter)} ({pct(schlechter, n)} %) sind schlechter{rest}. ")
    if besser == 0:
        test = "Kein Kind ist besser, also gibt es auch nichts auf den Testinstanzen zu prüfen."
    else:
        test = f"Von den {ganz(besser)} besseren sind {ganz(besser_test)} auch auf den Testinstanzen besser als der Elternteil."
    if parent_worse_bf:
        return "info", ("Der Elternteil ist auf dem Training schlechter als Best Fit. " + zahlen + test
                        + " Ob die Verbesserungen am Abstand des Elternteils zu Best Fit liegen, ist nicht gemessen.")
    if besser / n < selten:
        text = zahlen + test
        if besser > 0 and besser_test / besser < halb:
            text += " Weniger als die Hälfte der Gewinner auf dem Training gewinnt auch auf frischen Instanzen."
        return "warning", text
    return "info", zahlen + test


def verdict_lauf8(train_delta, test_delta, tol_test=C.TOL_TEST, tol_train=C.TOL_TRAIN):
    """Experiment 2, Live-Lauf: ein Lauf, kein Signifikanztest."""
    kopf = "(ein Lauf, kein Signifikanztest) "
    if test_delta <= -tol_test:
        return "success", (kopf + f"Auf den frischen Instanzen besser als Best Fit ({de_pm(test_delta)} Punkte). Das ist ein Lauf; "
                           "das Urteil über die 20 eingefrorenen Läufe steht darunter.")
    if train_delta <= -tol_train and test_delta >= tol_test:
        return "warning", (kopf + f"Auf dem Training {de(abs(train_delta))} Punkte besser als Best Fit, auf den frischen Instanzen "
                           f"{de(test_delta)} Punkte **schlechter**.")
    if train_delta <= -tol_train:
        return "warning", (kopf + f"Auf dem Training {de(abs(train_delta))} Punkte besser als Best Fit, auf den frischen Instanzen innerhalb der Toleranz "
                           f"von {de(tol_test, 1)} Punkten ({de_pm(test_delta)} Punkte): kein Gewinn nachweisbar.")
    return "info", (kopf + f"Auch auf dem Training kein Vorsprung von mindestens {de(tol_train, 1)} Punkten ({de_pm(train_delta)}); "
                    f"auf den frischen Instanzen {de_pm(test_delta)} Punkte.")


def verdict_zerlegung(familie_label, rows, budget, tol=C.TOL_TEST, k_min=C.K_MIN):
    """Experiment 2, eingefroren: die sieben Vergleiche einer Familie bei einem Budget."""
    n = rows[0]["n"]
    ct, cs = count_labels(rows, "train"), count_labels(rows, "test")
    var = [r for r in rows if r["art"] == "Variation allein"]
    var_nicht = sum(1 for r in var if r["train"]["urteil"] != "besser")
    text = (f"Über {n} eingefrorene Läufe, Familie »{familie_label}«, {ganz(budget)} Auswertungen: auf dem Training {ct['besser']} von {len(rows)} "
            f"Vergleichen »besser«, {ct['schlechter']} »schlechter«; auf den Testinstanzen {cs['besser']} »besser«, {cs['schlechter']} »schlechter« "
            f"(Regel: Median ≤ −{de(tol, 1)} Punkte und mindestens {k_min} von {n} Läufen besser). Die Variation ohne Auswahl der Besten war auf dem "
            f"Training in {var_nicht} von {len(var)} Vergleichen nicht »besser« als frisches Ziehen.")
    return ("warning" if ct["besser"] > 0 and cs["besser"] == 0 else "info"), text


def _u(z):
    """Urteil und Zählung einer eingefrorenen Zelle als Halbsatz."""
    if z is None:
        return "nicht gemessen"
    if z["urteil"] is None:
        return f"Median {de_pm(z['median'])} Punkte über {z['n']} Läufe, {z['neg']} davon unter null"
    return f"Median {de_pm(z['median'])} Punkte über {z['n']} Läufe, {z['neg']} davon unter null, Urteil »{z['urteil']}«"


def verdict_menge(k, live_orakel, live_merkmal, live_fest, z_orakel, z_merkmal, z_fest):
    """Experiment 3: die drei Wahlarten einer Menge gegen Best Fit. `live_*`: Deltas dieses Laufs; `z_*`: eingefrorene Zellen (oder None)."""
    kopf = (f"Dieser Lauf, gegen Best Fit: Orakelwahl über {k} Regeln {de_pm(live_orakel)} Punkte ({k} Läufe je Instanz), "
            f"Merkmalswahl {de_pm(live_merkmal)} Punkte (1 Lauf je Instanz), feste Wahl {de_pm(live_fest)} Punkte (1 Lauf je Instanz). ")
    if z_orakel is None:
        return "info", kopf + "Für diese Einstellung gibt es keine eingefrorenen Läufe."
    frozen = f"Eingefroren: Orakelwahl {_u(z_orakel)}; Merkmalswahl {_u(z_merkmal)}; feste Wahl {_u(z_fest)}. "
    if z_orakel["urteil"] == "besser":
        schluss = ("Der Gewinn gilt nur mit der Orakelwahl: dafür laufen im Einsatz alle Regeln der Menge auf jeder Instanz, und erst danach steht fest, "
                   "welche Packung zählt.")
        kind = "warning" if z_merkmal is not None and z_merkmal["urteil"] == "schlechter" else "info"
        if kind == "warning":
            schluss += " Die Wahl nach Merkmalen mit einem Lauf je Instanz ist schlechter als Best Fit."
        return kind, kopf + frozen + schluss
    return "info", kopf + frozen + "Kein Gewinn nach der Entscheidungsregel, auch nicht mit der Orakelwahl."


def verdict_vielfalt(k, d_greedy, d_topk, d_zufall, d_portfolio, quelle, tol_train=C.TOL_TRAIN):
    """Experiment 4: Orakelwahl über vier Mengenarten gegen Best Fit. `quelle` beschreibt, woher die Zahlen stammen."""
    text = (f"Orakelwahl gegen Best Fit ({quelle}): gierig nach Vielfalt gewählt {de_pm(d_greedy)} Punkte, die {k} nach Trainingsfitness besten "
            f"{de_pm(d_topk)}, {k} beliebige Regeln des Pools {de_pm(d_zufall)}, Best Fit mit First Fit und Worst Fit {de_pm(d_portfolio)} Punkte. ")
    if d_greedy < d_topk - tol_train:
        text += (f"Die nach Vielfalt gewählte Menge liegt {de(d_topk - d_greedy)} Punkte vor den {k} besten Regeln: "
                 "die besten Einzelregeln sind zusammen nicht die beste Menge.")
    else:
        text += f"Die nach Vielfalt gewählte Menge liegt hier nicht um mindestens {de(tol_train, 1)} Punkte vor den {k} besten Regeln."
    if d_zufall <= -tol_train:
        text += (f" Schon {k} beliebige Regeln des Pools gewinnen {de(abs(d_zufall))} Punkte. Ob das nur das Minimum über verschieden streuende "
                 "Packungen ist, ist eine Deutung, keine Messung.")
        return "warning", text
    return "info", text


def verdict_poolquelle(k, medians, familie_label, testmenge_label, tol=C.TOL_TRAIN):
    """Experiment 4: die gierige Menge mit Orakelwahl aus den Pools aller sechs Arme. `medians`: {Arm: Median des Deltas gegen Best Fit}.

    Verglichen wird der Pool des frischen Ziehens (Z-S) mit dem Pool aus Variation und Auswahl (EA-S)."""
    liste = ", ".join(f"{arm} {de_pm(medians[arm])}" for arm in C.ARMS)
    text = (f"Gierige Menge mit Orakelwahl aus den Pools der sechs Arme (k = {k}, Familie »{familie_label}«, Testmenge »{testmenge_label}«, "
            f"Median über 20 eingefrorene Läufe, Punkte gegen Best Fit): {liste}. ")
    d = medians["Z-S"] - medians["EA-S"]
    if d <= tol + 1e-9:
        return "warning", text + (f"Der Pool des frischen Ziehens (Z-S: keine Variation, keine Auswahl der Besten) liefert eine Menge, die nicht um mindestens "
                                  f"{de(tol, 1)} Punkte hinter der aus dem Pool von EA-S liegt (Abstand {de_pm(d)}): die Suche war für diesen Gewinn in diesem Vergleich nicht erforderlich.")
    return "info", text + f"Die Menge aus dem Pool von EA-S liegt {de(d)} Punkte vor der aus dem Pool des frischen Ziehens (Z-S)."


def verdict_wechsel(pool, familie_label, testmenge_label, k, z20, z100, haeufungen_fremd):
    """Experiment 5: die gierige Menge mit Orakelwahl auf einer anderen Testmenge und mit 100 Trainingsinstanzen."""
    text = f"Pool {pool}, auf »{familie_label}« trainiert, Testmenge »{testmenge_label}«, {k} Regeln, Orakelwahl gegen Best Fit: {_u(z20)}."
    if z100 is not None:
        text += f" Mit 100 Trainingsinstanzen: {_u(z100)}."
    else:
        text += " Mit 100 Trainingsinstanzen: für diese Zelle nicht gemessen."
    if haeufungen_fremd and z20 is not None and z20["urteil"] != "besser":
        return "warning", text + " Auf Häufungen trainiert gewinnt diese Menge auf der fremden Familie nichts gegen Best Fit."
    if z20 is not None and z20["urteil"] == "besser":
        return "info", text + f" Auch dieser Gewinn kostet {k} Läufe je Instanz."
    return "info", text
