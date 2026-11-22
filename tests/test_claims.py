"""Aussagen der README gegen die Messreihen: jede Befundzeile wird aus `hva_precomputed.json` neu gerechnet (`messungen/befunde.py`) und
wörtlich in der README gesucht. Dazu die Kernaussagen als Zählungen über das ganze Raster, Bänder um die gemessenen Werte und der Abgleich
der Grenzen-Texte zwischen App und README.

Die Datei selbst ist durch tests/test_neu_rechnen.py abgesichert (Ausschnitt neu gerechnet). Zählungen stehen exakt da, Mediane auf zwei
Stellen gerundet. Die Bänder sind um die Messung dieser Demo gelegt (Standardfehler des Medians grob über die Streuung der 20 Läufe)."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

import hva_constants as C
import hva_evaluation as ev
import hva_texts as T

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "messungen"))

import befunde  # noqa: E402

README = (ROOT / "README.md").read_text(encoding="utf-8")
PRE = ev.load_precomputed()
LINES = befunde.lines(PRE)
F = C.FAMILIES
med = ev.median


def sd(fam, arm, typ, k, wahl, ts="same60", ref="bf", block="m1"):
    return ev.set_deltas(PRE, fam, arm, typ, k, wahl, ts, ref, block)


@pytest.mark.parametrize("key", list(LINES))
def test_befundzeile_steht_woertlich_in_der_readme(key):
    assert LINES[key] in README, f"nicht in der README: {LINES[key]!r}"


def test_grenzen_text_ist_in_app_und_readme_derselbe():
    assert T.GRENZEN.strip() in README


def test_testzahl_der_readme_stimmt_mit_der_suite_ueberein():
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "tests"], cwd=ROOT, capture_output=True, text=True, timeout=300).stdout
    n = int(re.search(r"(\d+) tests? collected", out).group(1))
    assert f"{n} Tests (`pytest --collect-only -q tests`)" in README


# --- Kernaussagen als Zählungen über das ganze Raster ---------------------------------------------------------------------------------
def test_kein_arm_ist_auf_frischen_instanzen_besser_als_best_fit():
    for budget in C.CHECKPOINTS:
        assert ev.count_labels(ev.arm_table(PRE, budget))["besser"] == 0, budget
    rows = ev.arm_table(PRE)
    assert all(r["train"] < 0 for r in rows)                       # jeder Median auf dem Training liegt unter null
    assert all(r["test"] > -C.TOL_TEST for r in rows)


def test_kein_vergleich_ist_auf_frischen_instanzen_entschieden():
    """Variation allein, Auswahl der Besten, beides, Operator gegen Baumkreuzung: auf dem Test bei keiner Stufe »besser«."""
    for budget in C.CHECKPOINTS:
        for fam in F:
            assert ev.count_labels(ev.comparison_rows(PRE, fam, budget), "test")["besser"] == 0, (budget, fam)


def test_variation_allein_ist_auf_dem_training_in_keiner_zelle_besser():
    for budget in C.CHECKPOINTS:
        for fam in F:
            rows = [r for r in ev.comparison_rows(PRE, fam, budget) if r["art"] == "Variation allein"]
            assert len(rows) == 2 and all(r["train"]["urteil"] != "besser" for r in rows), (budget, fam)


def test_gierige_menge_mit_orakelwahl_gewinnt_und_die_merkmalswahl_nicht():
    for arm in C.POOL_ARMS:
        for fam in F:
            for k in (3, 5):
                lab, m, neg, _ = ev.urteil(sd(fam, arm, "greedy", k, "orakel"))
                assert lab == "besser" and neg >= 18, (arm, fam, k)
                assert ev.urteil(sd(fam, arm, "greedy", k, "merkmal"))[0] != "besser"
                assert med(sd(fam, arm, "greedy", k, "merkmal")) > 0
                assert ev.urteil(sd(fam, arm, "greedy", k, "fest"))[0] != "besser"
                # mehr Regeln, mehr Gewinn mit Orakelwahl; die Menge ist auf dem Training besser als auf dem Test
                assert med(sd(fam, arm, "greedy", 5, "orakel")) <= med(sd(fam, arm, "greedy", 3, "orakel")) + 1e-9
                assert med(ev.set_train_deltas(PRE, fam, arm, "greedy", k)) < med(sd(fam, arm, "greedy", k, "orakel")) < 0


def test_die_besten_einzelregeln_sind_als_menge_nicht_besser_als_best_fit():
    for arm in C.POOL_ARMS:
        for fam in F:
            for k in (3, 5):
                assert ev.urteil(sd(fam, arm, "topk", k, "orakel"))[0] == "gleich", (arm, fam, k)
                assert med(sd(fam, arm, "greedy", k, "orakel")) < med(sd(fam, arm, "topk", k, "orakel")) - C.TOL_TRAIN


def test_aussagen_der_preset_hilfen_und_der_einfachen_worte():
    """»Häufungen: die Menge gewinnt weniger als auf der gleichverteilten Familie« und »First Fit und Best Fit auf den meisten Instanzen gleich«;
    »eine einfache Wahl war schlechter als die feste Regel« (In einfachen Worten)."""
    for arm in C.ARMS:
        for k in (3, 5):
            assert med(sd("clu3", arm, "greedy", k, "orakel")) > med(sd("gl_20_70", arm, "greedy", k, "orakel")) or arm == "GP0-T", (arm, k)
    assert med(sd("clu3", "EA-S", "greedy", 3, "orakel")) > med(sd("gl_20_70", "EA-S", "greedy", 3, "orakel"))
    sp = PRE["spielraum"]["clu3"]["same60"]
    assert sp["ff_eq_bf"] > sp["n"] / 2
    assert "weniger als auf der gleichverteilten Familie" in C.PRESET_HELP["Häufungen: wenig Vielfalt"]
    labs = [ev.urteil(sd(f, a, "greedy", k, "merkmal"))[0] for a in C.POOL_ARMS for f in F for k in (3, 5)]
    fest = [ev.urteil(sd(f, a, "greedy", k, "fest"))[0] for a in C.POOL_ARMS for f in F for k in (3, 5)]
    assert labs.count("schlechter") == 12 and "war in der Messung schlechter als Best Fit" in T.EINFACH
    assert fest.count("besser") == 0 and "eine feste Wahl nicht besser" in T.EINFACH


def test_orakelwahl_ueber_die_feste_menge_bekannter_regeln_gewinnt_weniger_als_die_toleranz():
    for fam in F:
        d = ev.portfolio_deltas(PRE, fam)
        assert len(set(d)) == 1 and -C.TOL_TEST < d[0] <= 0          # hängt nicht am Lauf, nie schlechter als Best Fit


def test_zaehleinheiten_und_groesse_der_messung():
    n_m1 = sum(r["stat"]["n"] for fam in F for arm in C.ARMS for r in PRE["m1"][fam][arm])
    n_m3 = sum(r["stat"]["n"] for fam in C.BIG_FAMILIES for arm in C.BIG_ARMS for r in PRE["m3"][fam][arm])
    n_m4 = sum(r["n"] for fam in F for p in C.PARENTS for r in PRE["m4"][fam][p])
    n_m7 = sum(r["stat"]["n"] for r in PRE["m7"]["gl_20_70"]["EA-S"])
    assert (n_m1, n_m3, n_m4, n_m7) == (360000, 60000, 54000, 20000)
    for teil in ("360 Läufe mit je 1 000 Auswertungen", "60 Läufe mit 100 Trainingsinstanzen", "54 000 Kinder", "2 Läufe mit 10 000 Auswertungen",
                 f"{ev.ganz(n_m1 + n_m3 + n_m4 + n_m7)} gezählte Auswertungen"):
        assert teil in README, teil


def test_entscheidungsregel_und_mehrfachvergleich():
    assert f"p = {ev.de(ev.sign_test_p(15, 20), 4)}" in README and ev.de(ev.sign_test_p(15, 20), 4) == "0,0207"
    assert ev.bonferroni_k(20, 42) == 18
    assert ev.de(42 * ev.sign_test_p(15, 20), 2) == "0,87" and "0,87" in README


def test_handbeispiel_zahlen_der_readme():
    import hva_experiments as E

    h = E.hand_example()
    assert (h["bf"], h["oracle"], h["feature"], h["fixed"], h["topk_oracle"], h["greedy_steps"], h["topk_cost"]) == (8, 7, 9, 8, 8, [15, 14], 15)
    for teil in ("Best Fit 8 Behälter", "Orakelwahl über {Best Fit, Worst Fit} 7", "Merkmalswahl 9", "feste Wahl 8", "Trainingssumme 15", "Summe der Instanz-Minima 14"):
        assert teil in README, teil


def test_aussagen_nach_unabhaengiger_pruefung():
    """Sätze, die nach der unabhängigen Prüfung eingeschränkt wurden: Parameter-Operator, Wahlregeln, Menge nur mit Orakelwahl."""
    import hva_rng
    import hva_rules as R

    # Der Operator stört alle Parameter des gewählten Terms (S2 hat zwei Parameter): in jedem Schritt ändern sich beide
    rng = hva_rng.SplitMix64(7)
    th = (0.075, 0.35)
    for _ in range(50):
        neu = R._perturb(th, 1, rng)
        assert len(neu) == 2 and neu[0] != th[0] and neu[1] != th[1]
    assert len(R.TEMPLATES[1][3]) == 2 and len(R.TEMPLATES[2][3]) == 2 and all(len(R.TEMPLATES[i][3]) == 1 for i in (0, 3, 4))
    assert "die Parameter des ersten oder zweiten Terms stören (alle Parameter des gewählten Terms" in README
    assert "alle Parameter des gewählten Terms" in T.MATHE
    # genau zwei Wahlregeln mit einem Lauf je Instanz sind gemessen; mit keiner gewinnt die gierige Menge, mit der Orakelwahl in allen 12 Zellen
    assert C.CHOICES == ("orakel", "merkmal", "fest")
    zellen = [(a, f, k) for a in C.POOL_ARMS for f in F for k in (3, 5)]
    assert len(zellen) == 12
    assert all(ev.urteil(sd(f, a, "greedy", k, "orakel"))[0] == "besser" for a, f, k in zellen)
    assert not any(ev.urteil(sd(f, a, "greedy", k, wahl))[0] == "besser" for a, f, k in zellen for wahl in ("merkmal", "fest"))
    assert [ev.urteil(sd(f, a, "greedy", k, "fest"))[0] for a, f, k in zellen].count("gleich") == 10
    assert "gewinnt mit Orakelwahl (k Läufe je Instanz); mit den zwei gemessenen Wahlregeln mit einem Lauf je Instanz gewinnt sie nicht" in README
    assert "Gemessen sind zwei Wahlen mit einem Lauf je Instanz" in T.GRENZEN
    # die Menge ohne Auswahl der Besten (frisches Ziehen) gewinnt ebenfalls, in allen drei Familien
    for arm in ("Z-S", "Z-T"):
        assert all(ev.urteil(sd(f, arm, "greedy", 3, "orakel"))[0] == "besser" for f in F)
    assert "Pools ohne Auswahl der Besten" in README
