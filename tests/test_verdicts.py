"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Schwelle der Art und jede Zahl im Text steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zahlen, keine Zufalls-Seeds."""

import hva_constants as C
import hva_evaluation as ev


def test_schwellen_sind_die_festgelegten():
    assert (C.TOL_TEST, C.TOL_TRAIN, C.K_MIN, C.ANTEIL_SELTEN, C.HALB) == (0.2, 0.1, 15, 0.05, 0.5)
    assert round(ev.sign_test_p(15, 20), 4) == 0.0207
    assert ev.sign_test_p(20, 20) == 1 / 2 ** 20 and ev.sign_test_p(0, 20) == 1.0
    # Bonferroni über 42 Zellen (21 Vergleiche, Training und Test): 0,05 / 42 = 0,00119; 17 von 20 gibt 0,00129, 18 von 20 gibt 0,00020
    assert ev.bonferroni_k(20, 42) == 18 and ev.bonferroni_k(20, 1) == 15 and ev.bonferroni_k(3, 1000) is None


def test_zahlenformat():
    assert ev.de(-0.004) == "0,00" and ev.de(-0.75) == "−0,75" and ev.de_pm(0.2) == "+0,20" and ev.de_pm(-0.2) == "−0,20" and ev.de_pm(0) == "0,00"
    assert ev.ganz(54000) == "54 000" and ev.pct(83, 6000) == "1,4"


def test_verdict_hand_nennt_jede_zahl():
    kind, text = ev.verdict_hand(8, 7, 9, 8, 8, 2, 2)
    assert kind == "info"
    for teil in ("Best Fit 8 Behälter", "Menge 7 (2 Läufe je Instanz)", "Merkmalswahl 9 (1 Lauf je Instanz, 2 von 2 Mal die schlechtere Regel)", "feste Wahl 8",
                 "bringt auch mit Orakelwahl nichts (8 Behälter)"):
        assert teil in text, teil
    assert "braucht mit Orakelwahl 7 Behälter" in ev.verdict_hand(8, 7, 9, 8, 7, 2, 1)[1]


def test_verdict_variation_zweige():
    # selten (83 von 6000 = 1,4 % < 5 %) und weniger als die Hälfte hält (13 von 83)
    kind, text = ev.verdict_variation(False, 6000, 83, 3412, 2505, 13)
    assert kind == "warning"
    for teil in ("Von 6 000 Kindern", "83 (1,4 %)", "3 412 (56,9 %)", "2 505 (41,8 %)", "Von den 83 besseren sind 13 auch", "Weniger als die Hälfte"):
        assert teil in text, teil
    # selten, aber mindestens die Hälfte hält: kein Zusatzsatz
    kind, text = ev.verdict_variation(False, 100, 4, 50, 46, 2)
    assert kind == "warning" and "Weniger als die Hälfte" not in text
    # genau 5 % ist nicht mehr selten
    assert ev.verdict_variation(False, 100, 5, 50, 45, 0)[0] == "info"
    assert ev.verdict_variation(False, 100, 4, 50, 45, 1, gleichwertig=1)[1].count("1 weitere sind bei anderen Behälterzahlen gleich gut") == 1
    # Elternteil schlechter als Best Fit
    kind, text = ev.verdict_variation(True, 300, 150, 30, 120, 140)
    assert kind == "info" and text.startswith("Der Elternteil ist auf dem Training schlechter als Best Fit.") and "150 (50,0 %)" in text and "140 auch" in text


def test_verdict_lauf8_zweige():
    assert ev.verdict_lauf8(-0.5, -0.2)[0] == "success" and "−0,20 Punkte" in ev.verdict_lauf8(-0.5, -0.2)[1]
    kind, text = ev.verdict_lauf8(-0.1, 0.2)
    assert kind == "warning" and "0,10 Punkte besser" in text and "0,20 Punkte **schlechter**" in text
    kind, text = ev.verdict_lauf8(-0.76, 0.19)
    assert kind == "warning" and "kein Gewinn nachweisbar" in text and "(+0,19 Punkte)" in text and "0,2 Punkten" in text
    kind, text = ev.verdict_lauf8(-0.09, 0.3)
    assert kind == "info" and "Auch auf dem Training kein Vorsprung von mindestens 0,1 Punkten (−0,09)" in text and "+0,30 Punkte" in text
    assert all("ein Lauf, kein Signifikanztest" in ev.verdict_lauf8(a, b)[1] for a, b in ((-1, -1), (-1, 1), (-1, 0), (0, 0)))


def _row(art, train, test, n=20):
    return dict(a="A", b="B", art=art, n=n, train=dict(urteil=train, median=0.0, neg=0, pos=0), test=dict(urteil=test, median=0.0, neg=0, pos=0))


def test_verdict_zerlegung_zaehlt_und_waehlt_die_art():
    rows = [_row("Variation allein", "gleich", "gleich"), _row("Auswahl der Besten", "besser", "gleich"), _row("beides", "besser", "gleich"),
            _row("Variation allein", "schlechter", "gleich"), _row("Auswahl der Besten", "gleich", "gleich"), _row("beides", "gleich", "gleich"),
            _row("Operator gegen Baumkreuzung", "gleich", "schlechter")]
    kind, text = ev.verdict_zerlegung("gleichverteilt 20 bis 70", rows, 1000)
    assert kind == "warning"                                         # auf dem Training besser, auf dem Test nirgends
    for teil in ("Über 20 eingefrorene Läufe", "»gleichverteilt 20 bis 70«", "1 000 Auswertungen", "auf dem Training 2 von 7 Vergleichen »besser«, 1 »schlechter«",
                 "auf den Testinstanzen 0 »besser«, 1 »schlechter«", "Median ≤ −0,2 Punkte und mindestens 15 von 20", "in 2 von 2 Vergleichen nicht »besser«"):
        assert teil in text, teil
    rows[1] = _row("Auswahl der Besten", "besser", "besser")
    assert ev.verdict_zerlegung("x", rows, 200)[0] == "info"
    rows = [_row("Variation allein", "besser", "gleich")] + [_row("beides", "gleich", "gleich")] * 6
    assert "in 0 von 1 Vergleichen nicht »besser«" in ev.verdict_zerlegung("x", rows, 200)[1]
    assert ev.verdict_zerlegung("x", [_row("beides", "gleich", "gleich")] * 7, 200)[0] == "info"


def _z(urteil, median, neg, n=20):
    return dict(urteil=urteil, median=median, neg=neg, pos=n - neg, n=n)


def test_verdict_menge_zweige():
    kind, text = ev.verdict_menge(3, -0.75, 0.5, 0.2, _z("besser", -0.75, 20), _z("schlechter", 0.5, 2), _z("gleich", 0.1, 8))
    assert kind == "warning"
    for teil in ("Orakelwahl über 3 Regeln −0,75 Punkte (3 Läufe je Instanz)", "Merkmalswahl +0,50 Punkte (1 Lauf je Instanz)", "feste Wahl +0,20 Punkte",
                 "Median −0,75 Punkte über 20 Läufe, 20 davon unter null, Urteil »besser«", "Urteil »schlechter«", "Der Gewinn gilt nur mit der Orakelwahl",
                 "schlechter als Best Fit"):
        assert teil in text, teil
    kind, text = ev.verdict_menge(5, -0.3, 0.1, 0.0, _z("besser", -0.3, 16), _z("gleich", 0.1, 9), _z("gleich", 0.0, 10))
    assert kind == "info" and "Der Gewinn gilt nur mit der Orakelwahl" in text and "schlechter als Best Fit" not in text
    kind, text = ev.verdict_menge(3, -0.1, 0.1, 0.0, _z("gleich", -0.1, 12), _z("gleich", 0.1, 9), _z("gleich", 0.0, 10))
    assert kind == "info" and "Kein Gewinn nach der Entscheidungsregel" in text
    kind, text = ev.verdict_menge(1, 0.1, 0.1, 0.1, None, None, None)
    assert kind == "info" and "keine eingefrorenen Läufe" in text
    # zehn Läufe: Zählung ohne Urteil
    assert "über 10 Läufe, 10 davon unter null" in ev._u(dict(urteil=None, median=-1.0, neg=10, pos=0, n=10)) and "Urteil" not in ev._u(dict(urteil=None, median=-1.0, neg=10, pos=0, n=10))


def test_verdict_vielfalt_zweige():
    kind, text = ev.verdict_vielfalt(3, -0.75, 0.01, -0.21, -0.08, "Median über 20 Läufe")
    assert kind == "warning"
    for teil in ("(Median über 20 Läufe)", "gewählt −0,75 Punkte", "besten +0,01", "3 beliebige Regeln des Pools −0,21", "Worst Fit −0,08 Punkte",
                 "liegt 0,76 Punkte vor den 3 besten", "Schon 3 beliebige Regeln des Pools gewinnen 0,21 Punkte", "eine Deutung, keine Messung"):
        assert teil in text, teil
    kind, text = ev.verdict_vielfalt(3, -0.75, 0.01, -0.09, -0.08, "q")
    assert kind == "info" and "Deutung" not in text                  # −0,09 liegt über der Anzeigeschwelle −0,1
    assert ev.verdict_vielfalt(3, -0.75, 0.01, -0.1, 0.0, "q")[0] == "warning"          # genau −0,1 zählt
    kind, text = ev.verdict_vielfalt(5, -0.1, -0.05, 0.3, 0.0, "q")
    assert kind == "info" and "nicht um mindestens 0,1 Punkte vor den 5 besten" in text


def test_verdict_poolquelle_zweige():
    med = {"Z-S": -1.2, "ZW-S": -1.08, "EA-S": -0.92, "Z-T": -0.48, "GP0-T": -0.34, "GP-T": -0.47}
    kind, text = ev.verdict_poolquelle(3, med, "gleichverteilt", "gleiche Familie")
    assert kind == "warning"
    for teil in ("k = 3", "»gleichverteilt«", "»gleiche Familie«", "Z-S −1,20, ZW-S −1,08, EA-S −0,92, Z-T −0,48, GP0-T −0,34, GP-T −0,47", "(Abstand −0,28)",
                 "die Suche war für diesen Gewinn in diesem Vergleich nicht erforderlich"):
        assert teil in text, teil
    med["Z-S"] = -0.82                                              # genau 0,1 Punkte hinter EA-S: zählt noch als »nicht dahinter«
    assert ev.verdict_poolquelle(3, med, "f", "t")[0] == "warning"
    med["Z-S"] = -0.5
    kind, text = ev.verdict_poolquelle(5, med, "f", "t")
    assert kind == "info" and "liegt 0,42 Punkte vor der aus dem Pool des frischen Ziehens" in text and "nicht erforderlich" not in text


def test_verdict_wechsel_zweige():
    kind, text = ev.verdict_wechsel("GP-T", "drei Häufungen", "fremde Familie 1", 3, _z("gleich", 0.32, 4), None, True)
    assert kind == "warning" and "Pool GP-T" in text and "Median +0,32 Punkte" in text and "nicht gemessen" in text and "gewinnt diese Menge auf der fremden Familie nichts" in text
    kind, text = ev.verdict_wechsel("EA-S", "gleichverteilt", "gleiche Familie", 3, _z("besser", -0.75, 20), _z(None, -1.23, 10, 10), False)
    assert kind == "info" and "Mit 100 Trainingsinstanzen: Median −1,23 Punkte über 10 Läufe, 10 davon unter null." in text and "kostet 3 Läufe je Instanz" in text
    kind, text = ev.verdict_wechsel("EA-S", "x", "y", 5, _z("gleich", -0.1, 12), None, False)
    assert kind == "info" and "kostet" not in text and "nichts gegen Best Fit" not in text
    # Häufungen auf fremder Familie, aber doch »besser«: keine Warnung
    assert ev.verdict_wechsel("EA-S", "x", "y", 5, _z("besser", -0.4, 18), None, True)[0] == "info"


def test_cell_gibt_bei_zehn_laeufen_kein_urteil():
    z = ev.cell([-1.0] * 10)
    assert z == dict(urteil=None, median=-1.0, neg=10, pos=0, n=10) and ev.cell(None) is None
    assert ev.cell([-1.0] * 20)["urteil"] == "besser"
