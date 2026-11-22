"""End-to-End über Streamlits AppTest: Standard, jeder Button, jeder Regler an beiden Grenzen, jede Auswahl jedes Auswahlfelds,
Worst Case am Slider-Maximum, Permalink, der Knopf von Experiment 2 und die Verdikt-Zeilen je Preset.

Expander werden per Titel angesprochen, nie per Index (DEMO-PLAYBOOK §2). Die Bedientests laufen mit kleinem Budget (Adresszeile),
damit jeder Lauf kurz bleibt; die Presets und der Worst Case rechnen mit ihren echten Werten."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import hva_constants as C
import hva_evaluation as ev
import hva_experiments as E
import hva_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
PRE = ev.load_precomputed()
QUICK = dict(budget="100", kinder="100")


def _fresh_app(**query) -> AppTest:
    at = AppTest.from_file(APP_PATH)
    for k, v in query.items():
        at.query_params[k] = v
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return [e.value for kind in ("success", "info", "warning", "error") for e in getattr(at, kind)]


def test_app_loads_without_exception():
    at = _fresh_app()
    assert not at.error
    assert at.title[0].value.startswith("🧬 Heuristiken variieren und zu Mengen bündeln")


def test_expander_und_abschnitte_per_titel():
    at = _fresh_app(**QUICK)
    labels = [e.label for e in at.expander]
    for titel in ("💬 In einfachen Worten", "So funktioniert der Kreislauf aus Variation und Auswahl", "🔬 Alle Arme und Mengen, 20 Läufe (eingefroren)",
                  "📐 Mathematische Formulierung"):
        assert titel in labels, titel
    assert labels[0] == "💬 In einfachen Worten" and labels[-1] == "📐 Mathematische Formulierung"
    subs = [s.value for s in at.subheader]
    assert len([s for s in subs if s.startswith("🔬 Experiment")]) == 5 and "🚧 Wo die Annahmen enden" in subs
    einfach = next(e for e in at.expander if e.label == "💬 In einfachen Worten")
    text = " ".join(m.value for m in einfach.markdown)
    assert 80 <= len(text.split()) <= 130 and not any(ch.isdigit() for ch in text)
    alle = " ".join(m.value for m in at.markdown)
    assert all(f"{i}. " in alle for i in (1, 2, 3)) and "Kernfragen" in alle


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app(**QUICK).button]
    assert len(labels) == len(C.PRESETS) + 2, labels            # Presets, »Neue Zufalls-Seeds«, Knopf von Experiment 2
    for index, label in enumerate(labels):
        at = _fresh_app(**QUICK)
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app(**QUICK).slider]
    assert len(labels) == 4, labels                              # Budget, Kinder, Schritt des Handbeispiels, Test-Zufalls-Seed
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app(**QUICK)
            slider = at.slider[index]
            slider.set_value(int(slider.min if edge == "min" else slider.max)).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_jede_auswahl_jedes_auswahlfelds_und_auswahlreglers():
    at0 = _fresh_app(**QUICK)
    assert {s.key for s in at0.selectbox} == {"familie", "testmenge", "arm", "menge", "wahl", "elternteil"} and {s.key for s in at0.select_slider} == {"k"}
    for key in ("familie", "testmenge", "arm", "menge", "wahl", "elternteil"):
        for option in P.options(key):
            at = _fresh_app(**QUICK)
            at.selectbox(key=key).set_value(option).run(timeout=TIMEOUT)
            assert not at.exception, (key, option, [str(e) for e in at.exception])
    for option in P.options("k"):
        at = _fresh_app(**QUICK)
        at.select_slider(key="k").set_value(option).run(timeout=TIMEOUT)
        assert not at.exception, option
    for value in P.bounds("lauf_seed"):
        at = _fresh_app(**QUICK)
        at.number_input(key="lauf_seed").set_value(value).run(timeout=TIMEOUT)
        assert not at.exception, value


def test_worst_case_am_slider_maximum():
    """Größtes Budget, meiste Kinder, größte Menge, größte Instanzen, Bäume; kein Zeit-Assert, nur das Timeout."""
    at = _fresh_app(arm="GP0-T", budget="1000", kinder="600", k="5", test="n120", fam="stark", eltern="frisch", menge="zufall")
    assert not at.exception and not at.error
    at = _fresh_app(arm="ZW-S", budget="1000", kinder="600", k="5", test="fremd2", fam="clu3", wahl="merkmal")
    assert not at.exception and not at.error


def test_knopf_der_drei_arme_zeigt_die_tabelle_und_verfaellt_bei_neuer_einstellung():
    def has_table(at):
        return any("Endregel" in d.value.columns for d in at.dataframe)

    at = _fresh_app(budget="200", kinder="100")
    assert not has_table(at)
    at.button(key="drei_button").click().run(timeout=TIMEOUT)
    assert has_table(at)
    df = next(d.value for d in at.dataframe if "Endregel" in d.value.columns)
    assert len(df) == 3 and list(df["Arm"]) == [C.ARM_LABELS[a] for a in ("Z-S", "ZW-S", "EA-S")]
    at.slider(key="budget").set_value(300).run(timeout=TIMEOUT)
    assert not has_table(at)                                      # gehört zu den alten Einstellungen
    at = _fresh_app(budget="500", kinder="100")
    assert at.button(key="drei_button").disabled
    assert any("auf Budgets bis 400 Auswertungen begrenzt" in c.value for c in at.caption)


def test_permalink_wird_begrenzt_und_eingerastet():
    at = _fresh_app(fam="clu3", arm="unbekannt", budget="149", k="4", seed="5000", schritt="99", kinder="100", wahl="merkmal", test="n30")
    ss = at.session_state
    assert (ss["familie"], ss["arm"], ss["budget"], ss["lauf_seed"], ss["hand_schritt"], ss["wahl"], ss["testmenge"]) == ("clu3", "EA-S", 100, 999, 3, "merkmal", "n30")
    assert ss["k"] in (3, 5)


def test_live_lauf_mit_seed_0_rechnet_den_ersten_eingefrorenen_lauf_nach():
    at = _fresh_app(budget="1000", kinder="100")
    frozen = PRE["m1"][C.DEFAULT_FAMILY][C.DEFAULT_ARM][0]
    wert = {m.label: m.value for m in at.metric}
    assert wert["Training"] == f"{ev.de_pm(frozen['train'])} Punkte" and wert["Test"] == f"{ev.de_pm(frozen['test'])} Punkte"
    assert wert["verschiedene Regeln im Pool"] == ev.ganz(frozen["pool"])
    assert any(frozen["rule"] in c.value for c in at.caption)
    live = frozen["sets"]["greedy"]["3"]["T"]["same60"]
    assert wert["Orakelwahl über 3 Regeln, dieser Lauf"] == f"{ev.de_pm(live[0])} Punkte" and wert["Merkmalswahl, dieser Lauf"] == f"{ev.de_pm(live[1])} Punkte"


def test_neue_zufalls_seeds_erhoeht_den_seed_um_eins():
    at = _fresh_app(**QUICK)
    at.button(key="seed_button").click().run(timeout=TIMEOUT)
    assert at.session_state["lauf_seed"] == 1
    at = _fresh_app(seed="999", **QUICK)
    at.button(key="seed_button").click().run(timeout=TIMEOUT)
    assert at.session_state["lauf_seed"] == 0


def test_hand_schritte_bauen_die_menge_auf():
    at = _fresh_app(schritt="0", **QUICK)
    tab = next(d.value for d in at.dataframe if "Summe Training" in d.value.columns)
    assert list(tab["Summe Training"]) == [15, 16, 17, 15] and "Te1" not in tab.columns and "Minima mit Best Fit" not in tab.columns
    assert not any("Best Fit allein" == m.label for m in at.metric)
    at = _fresh_app(schritt="2", **QUICK)
    tab = next(d.value for d in at.dataframe if "Summe Training" in d.value.columns)
    assert list(tab["Minima mit Best Fit"]) == [15, 15, 14, 15] and "Te1" not in tab.columns
    assert any("**Best Fit, Worst Fit**" in m.value and "**14** Behälter" in m.value for m in at.markdown)
    at = _fresh_app(schritt="3", **QUICK)
    tab = next(d.value for d in at.dataframe if "Summe Training" in d.value.columns)
    assert list(tab["Summe Test"]) == [8, 9, 8, 9]
    wert = {m.label: m.value for m in at.metric}
    assert (wert["Best Fit allein"], wert["Orakelwahl"], wert["Merkmalswahl"], wert["feste Wahl"]) == ("8 Behälter", "7 Behälter", "9 Behälter", "8 Behälter")
    assert any("16 Platzierungsläufe" in c.value and "Orakelwahl 2 Läufe je Instanz" in c.value for c in at.caption)


def _expected(p):
    """Die Verdikte eines Presets aus den reinen Funktionen, mit denselben Rechnungen wie die App."""
    fam, arm, budget, k, menge, tm, lauf, ts = p["familie"], p["arm"], p["budget"], p["k"], p["menge"], p["testmenge"], p["lauf_seed"], p["test_seed"]
    tests = E.test_sets(fam, ts)
    s = E.search_run(fam, arm, lauf, budget=budget)
    rec = E.run_record(fam, arm, lauf, {"same60": tests["same60"]}, budget=budget, set_types=("greedy", "topk", "zufall"), k_list=(k,), search=s)
    rec_t = rec if tm == "same60" else E.run_record(fam, arm, lauf, {tm: tests[tm]}, budget=budget, set_types=("greedy", "topk", "zufall"), k_list=(k,), search=s)
    live = rec_t["sets"][menge][str(k)]["T"][tm]
    out = {}
    out["zerlegung"] = ev.verdict_zerlegung(C.FAMILY_LABELS[fam], ev.comparison_rows(PRE, fam, budget), budget)
    out["lauf"] = ev.verdict_lauf8(rec["train"], rec["test"])
    z = {w: ev.cell(ev.set_deltas(PRE, fam, arm, menge, k, w, tm)) for w in C.CHOICES}
    out["menge"] = ev.verdict_menge(k, live[0], live[1], live[2], z["orakel"], z["merkmal"], z["fest"])
    med = {t: ev.median(ev.set_deltas(PRE, fam, arm, t, k, "orakel", tm)) for t in C.SET_TYPES}
    out["vielfalt"] = ev.verdict_vielfalt(k, med["greedy"], med["topk"], med["zufall"], ev.median(ev.portfolio_deltas(PRE, fam, "orakel", tm)),
                                          f"Median über 20 eingefrorene Läufe, Pool {arm} mit 1 000 Auswertungen, Testmenge »{C.testset_label(tm, fam)}«")
    out["pool"] = ev.verdict_poolquelle(k, {a: ev.median(ev.set_deltas(PRE, fam, a, "greedy", k, "orakel", tm)) for a in C.ARMS},
                                        C.FAMILY_LABELS[fam], C.testset_label(tm, fam))
    z20 = ev.cell(ev.set_deltas(PRE, fam, arm, "greedy", k, "orakel", tm))
    z100 = ev.cell(ev.set_deltas(PRE, fam, arm, "greedy", k, "orakel", tm, block="m3")) if tm == "same60" else None
    out["wechsel"] = ev.verdict_wechsel(arm, C.FAMILY_LABELS[fam], C.testset_label(tm, fam), k, z20, z100, fam == "clu3" and tm.startswith("fremd"))
    v = E.variation_experiment(fam, p["elternteil"], lauf, p["kinder"], tests["same60"])
    c = v["counts"]
    out["variation"] = ev.verdict_variation(v["parent_worse_bf"], v["n"], c["besser"], c["gleich"], c["schlechter"], v["better_test"], c["gleichwertig"])
    return out


# Je Preset die gemessene Art von vier Verdikten: (Zerlegung über 20 eingefrorene Läufe, Live-Lauf, Menge, Variation).
# Ein Preset ohne Effekt meldet keinen Effekt; die Arten stammen aus der Rechnung und sind hier festgehalten.
ERWARTET = {
    "Eine Menge von Hand": ("info", "warning", "warning", "warning"),            # bei 400 Auswertungen ist auch auf dem Training kein Vergleich »besser«
    "Variation an Best Fit": ("info", "warning", "warning", "warning"),          # 2 von 300 Kindern besser: »selten«
    "Variation und Auswahl": ("warning", "warning", "warning", "warning"),       # Training entschieden, Test nicht
    "Bäume statt Vorlagen": ("warning", "warning", "warning", "warning"),
    "Menge mit Orakelwahl": ("warning", "warning", "warning", "warning"),        # Orakelwahl »besser«, Merkmalswahl »schlechter«
    "Menge ohne Orakel": ("warning", "warning", "warning", "warning"),
    "Häufungen: wenig Vielfalt": ("info", "info", "warning", "warning"),         # kein Trainingsvorsprung des Laufs, kein Vergleich entschieden
}


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_zeigt_die_verdikte_aus_der_rechnung(name):
    """Die App zeigt je Preset genau die Verdikt-Zeilen, die die reinen Funktionen für die Werte dieses Presets liefern."""
    p = C.PRESETS[name]
    at = _fresh_app()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception
    shown = _texts(at)
    exp = _expected(p)
    for key, (kind, text) in exp.items():
        assert text in [e.value for e in getattr(at, kind)], (name, key, kind, text)
    for key in ("zerlegung", "lauf", "menge", "vielfalt", "wechsel"):
        assert shown.count(exp[key][1]) == 2, (name, key)          # Kernfragen und Experiment zeigen dieselbe Zeile
    assert shown.count(exp["variation"][1]) == 1 and shown.count(exp["pool"][1]) == 1
    hand_gezeigt = sum("Auf den zwei Testinstanzen braucht Best Fit 8 Behälter" in t for t in shown)
    assert hand_gezeigt == (1 if p["hand_schritt"] == 3 else 0)
    assert set(ERWARTET) == set(C.PRESETS)
    assert (exp["zerlegung"][0], exp["lauf"][0], exp["menge"][0], exp["variation"][0]) == ERWARTET[name], (name, [exp[k][0] for k in ("zerlegung", "lauf", "menge", "variation")])
