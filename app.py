"""Heuristiken variieren und zu Mengen bündeln - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Online-Packen mit Platzierungsfunktionen: ein Vorlagen-Operator verändert Regeln, sechs Arme trennen Variation und Auswahl der Besten,
und aus dem Pool eines Laufs entstehen Mengen von Regeln mit drei Wahlarten im Einsatz. Aufbau nach DEMO-PLAYBOOK §2.

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import hva_arms as AR
import hva_constants as C
import hva_evaluation as ev
import hva_experiments as E
import hva_rules as R
import hva_sets as M
import hva_texts as T
import hva_variation as VA
import hva_visualization as V
from hva_layout import dunkles_design, kopf_css
from hva_presets import bounds, init_session_state_defaults, load_permalink_settings, next_run_seed, options, sidebar_presets, sync_query_params

st.set_page_config(page_title="Heuristik-Variation – Sebastian Hanisch", layout="wide")
de, de_pm, ganz, pct = ev.de, ev.de_pm, ev.ganz, ev.pct


# --- zwischengespeicherte Rechnungen (Schlüssel: alle Eingaben samt Zufalls-Seeds) --------------------------------------------------------
@st.cache_data(show_spinner=False)
def _pre():
    return ev.load_precomputed()


@st.cache_data(show_spinner=False, max_entries=24)
def _test(fam, test_seed, name):
    """Eine Testmenge (die Normalzahlen der Familien mit Häufungen und breiter Streuung sind teuer: je Menge einmal)."""
    return E.test_sets(fam, test_seed, names=(name,))[name]


@st.cache_data(show_spinner=False, max_entries=8)
def _search(fam, arm, lauf, budget):
    return E.search_run(fam, arm, lauf, budget=budget)


@st.cache_data(show_spinner=False, max_entries=24)
def _run_view(fam, arm, lauf, budget, test_seed):
    """Ein Lauf: Verlauf des Bestwerts mit Test-Delta an den Verbesserungsstellen, Endregel, Zählungen."""
    s = _search(fam, arm, lauf, budget)
    tests = {"same60": _test(fam, test_seed, "same60")}
    evl = E.Evaluator(s["space"], s["pool_inds"], s["pool_matrix"], s["train"], tests)
    pts = [(n, d, evl.delta(evl.bins(j, "same60"), "same60")) for n, d, j in E.improvements(s)]
    jbest = E.improvements(s)[-1][2]
    ind = s["pool_inds"][jbest]
    return dict(points=pts, train=pts[-1][1], test=pts[-1][2], rule=R.ind_str(s["space"], ind), space=s["space"],
                knoten=R.tree_size(ind) if s["space"] == "T" else None, stat=s["stat"], pool=len(s["pool_inds"]),
                bilanz=E.bilanz(evl.bins(jbest, "same60"), evl.base["same60"][1]))


@st.cache_data(show_spinner=False, max_entries=24)
def _set_view(fam, arm, lauf, budget, test_seed, k, testmenge):
    """Die Mengenarten aus dem Pool des Laufs bei Größe k auf einer Testmenge: Deltas der drei Wahlarten, Mitglieder, Bilanz."""
    s = _search(fam, arm, lauf, budget)
    tests = {testmenge: _test(fam, test_seed, testmenge)}
    evl = E.Evaluator(s["space"], s["pool_inds"], s["pool_matrix"], s["train"], tests)
    sets = E.build_sets(s["pool_fits"], s["pool_matrix"], E.set_seed(arm, fam, lauf))
    jbest = E.improvements(s)[-1][2]
    out = {"ref": evl.delta(evl.bins(jbest, testmenge), testmenge), "pool": len(s["pool_inds"]), "types": {}}
    for typ in C.SET_TYPES + ("portfolio",):
        kk = len(M.PORTFOLIO) if typ == "portfolio" else k
        summ = E.set_summary(evl, sets, typ, kk, s["lb"], s["bf_vec"], [testmenge])
        entry = dict(deltas=dict(zip(C.CHOICES, summ["T"][testmenge])), train=summ["train"], eq_bf=summ["eq_bf"], size=summ["size"])
        if typ in ("greedy", "topk"):
            mem = sets[typ][:k]
            entry["members"] = [(R.ind_str(s["space"], s["pool_inds"][j]), int(s["pool_matrix"][j].sum())) for j in mem]
            entry["bilanz"] = E.bilanz(evl.choice_bins(mem, testmenge)["orakel"], evl.base[testmenge][1])
        out["types"][typ] = entry
    out["bf_train_sum"] = int(s["bf_vec"].sum())
    return out


@st.cache_data(show_spinner=False, max_entries=24)
def _variation(fam, parent, lauf, kinder, test_seed):
    return E.variation_experiment(fam, parent, lauf, kinder, _test(fam, test_seed, "same60"))


@st.cache_data(show_spinner=False)
def _spielraum(fam, test_seed):
    return E.spielraum(_test(fam, test_seed, "same60"))


@st.cache_data(show_spinner=False)
def _hand():
    return E.hand_example()


PRE = _pre()

# --- Kopf, Einleitung ------------------------------------------------------------------------------------------------------------------------
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title(T.TITEL)
st.markdown(T.EINLEITUNG)

with st.expander("💬 In einfachen Worten"):
    st.markdown(T.EINFACH)

with st.expander("So funktioniert der Kreislauf aus Variation und Auswahl", expanded=True):
    st.markdown(T.SO_FUNKTIONIERT)
    st.graphviz_chart(T.KREISLAUF_DOT)
    st.markdown("**Die fünf Vorlagen**")
    st.markdown(T.VORLAGEN_TABELLE)
    st.markdown(T.ZUORDNUNG)

st.caption("🔗 Die Adresszeile oben spiegelt die aktuellen Einstellungen wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("📦 Instanzen")
    fam = st.selectbox("Instanzfamilie", options("familie"), key="familie", format_func=lambda f: C.FAMILY_LABELS[f],
                       help="Aus dieser Verteilung stammen Trainings- und Testinstanzen (je 60 Gegenstände, Kapazität 100).")
    testmenge = st.selectbox("Testmenge", options("testmenge"), key="testmenge", format_func=lambda t: C.testset_label(t, st.session_state.get("familie", C.DEFAULT_FAMILY)),
                             help="Auf welchen frischen Instanzen die Mengen in Experiment 3 bis 5 eingesetzt werden. Experiment 1 und 2 testen immer auf der gleichen Familie mit 60 Gegenständen.")
    st.header("🧬 Erzeugen")
    arm = st.selectbox("Arm", options("arm"), key="arm", format_func=lambda a: C.ARM_LABELS[a],
                       help="Sechs Arme: frisch ziehen, Variation ohne Auswahl, Variation mit Auswahl, je für Schablonen und Bäume. Die Erklärung steht in Experiment 2.")
    budget = st.slider("Budget (Auswertungen)", *bounds("budget"), step=C.BUDGET_STEP, key="budget",
                       help="Eine Auswertung = eine Regel einmal auf allen 20 Trainingsinstanzen. Der Pool der Mengen stammt aus diesem Lauf.")
    st.header("🧺 Mengen")
    k = st.select_slider("Mengengröße k", options("k"), key="k", help="Zahl der Regeln in der Menge; mit k = 1 ist die Menge eine einzelne Regel.")
    menge = st.selectbox("Menge", options("menge"), key="menge", format_func=lambda m: C.SET_LABELS[m],
                         help="Wie die k Regeln aus dem Pool gewählt werden. Die feste Menge Best Fit, First Fit, Worst Fit steht in Experiment 4 immer daneben.")
    wahl = st.selectbox("Wahl im Einsatz", options("wahl"), key="wahl", format_func=lambda w: C.CHOICE_LABELS[w],
                        help="Hebt in Experiment 3 eine der drei Wahlarten hervor; gerechnet werden immer alle drei.")
    st.header("🔀 Variation")
    elternteil = st.selectbox("Elternteil", options("elternteil"), key="elternteil", format_func=lambda p: C.PARENT_LABELS[p],
                              help="Wirkt nur auf Experiment 1: von dieser Regel aus werden die Kinder erzeugt.")
    kinder = st.slider("Kinder", *bounds("kinder"), step=C.KINDER_STEP, key="kinder", help="Wirkt nur auf Experiment 1: je Kind eine Auswertung.")
    st.header("✋ Handbeispiel")
    hand_step = st.slider("Schritt des Handbeispiels", *bounds("hand_schritt"), key="hand_schritt",
                          help="0 = Tabelle, 1 = erste Wahl, 2 = zweite Wahl, 3 = Einsatz auf den Testinstanzen.")
    st.header("🎲 Zufall")
    test_seed = st.slider("Test-Zufalls-Seed", *bounds("test_seed"), key="test_seed",
                          help="Wählt eine von zehn Testmengen. Die eingefrorenen Läufe nutzen die Testmenge 0.")
    lauf = st.number_input("Zufalls-Seed der Läufe", *bounds("lauf_seed"), step=1, key="lauf_seed",
                           help="Bestimmt die Trainingsinstanzen und den Zufallsstrom der Suche. Mit 0 rechnet die App den ersten eingefrorenen Lauf nach.")
    st.button("🎲 Neue Zufalls-Seeds", width="stretch", on_click=next_run_seed, key="seed_button")

fam, arm, budget, k, lauf, test_seed, kinder, hand_step = str(fam), str(arm), int(budget), int(k), int(lauf), int(test_seed), int(kinder), int(hand_step)
sync_query_params({"familie": fam, "arm": arm, "budget": budget, "k": k, "menge": menge, "wahl": wahl, "testmenge": testmenge, "elternteil": elternteil,
                   "kinder": kinder, "hand_schritt": hand_step, "test_seed": test_seed, "lauf_seed": lauf})
fam_label = C.FAMILY_LABELS[fam]
ts_label = C.testset_label(testmenge, fam)

# --- Rechnungen dieses Laufs ------------------------------------------------------------------------------------------------------------------
with st.spinner(f"Rechne einen Lauf mit {ganz(budget)} Auswertungen (je Auswertung 20 Trainingsinstanzen) und {kinder} Kinder …"):
    run = _run_view(fam, arm, lauf, budget, test_seed)
    sv = _set_view(fam, arm, lauf, budget, test_seed, k, testmenge)
    var = _variation(fam, elternteil, lauf, kinder, test_seed)
hand = _hand()

# Verdikte (reine Funktionen, hva_evaluation)
rows_zerl = ev.comparison_rows(PRE, fam, budget)
v_zerl = ev.verdict_zerlegung(fam_label, rows_zerl, budget)
v_lauf = ev.verdict_lauf8(run["train"], run["test"])
live = sv["types"][menge]["deltas"]
z_cells = {w: ev.cell(ev.set_deltas(PRE, fam, arm, menge, k, w, testmenge)) for w in C.CHOICES}
v_menge = ev.verdict_menge(k, live["orakel"], live["merkmal"], live["fest"], z_cells["orakel"], z_cells["merkmal"], z_cells["fest"])
med = {typ: ev.median(ev.set_deltas(PRE, fam, arm, typ, k, "orakel", testmenge)) for typ in C.SET_TYPES}
med_port = ev.median(ev.portfolio_deltas(PRE, fam, "orakel", testmenge))
v_viel = ev.verdict_vielfalt(k, med["greedy"], med["topk"], med["zufall"], med_port,
                             f"Median über 20 eingefrorene Läufe, Pool {arm} mit 1 000 Auswertungen, Testmenge »{ts_label}«")
z20 = ev.cell(ev.set_deltas(PRE, fam, arm, "greedy", k, "orakel", testmenge))
z100 = ev.cell(ev.set_deltas(PRE, fam, arm, "greedy", k, "orakel", testmenge, block="m3")) if testmenge == "same60" else None
v_pool = ev.verdict_poolquelle(k, {a: ev.median(ev.set_deltas(PRE, fam, a, "greedy", k, "orakel", testmenge)) for a in C.ARMS}, fam_label, ts_label)
v_wechsel = ev.verdict_wechsel(arm, fam_label, ts_label, k, z20, z100, fam == "clu3" and testmenge.startswith("fremd"))
falsch = sum(1 for i, p in enumerate(hand["pick"]) if hand["test_bins"][hand["greedy"][p]][i] > min(hand["test_bins"][j][i] for j in hand["greedy"]))
v_hand = ev.verdict_hand(hand["bf"], hand["oracle"], hand["feature"], hand["fixed"], hand["topk_oracle"], E.HAND_K, falsch)
c = var["counts"]
v_var = ev.verdict_variation(var["parent_worse_bf"], var["n"], c["besser"], c["gleich"], c["schlechter"], var["better_test"], c["gleichwertig"])

# --- Primäransicht: eine Menge von Hand -------------------------------------------------------------------------------------------------------
st.markdown("## ✋ Eine Menge von Hand")
st.caption(T.E0_CAPTION)
names = [E.HAND_RULE_LABELS[r] for r in hand["rules"]]
tab = pd.DataFrame({
    "Regel": names,
    **{f"Tr{i + 1}": [int(x) for x in hand["train_bins"][:, i]] for i in range(4)},
    "Summe Training": hand["train_sums"],
})
if hand_step >= 3:
    tab["Te1"] = [int(x) for x in hand["test_bins"][:, 0]]
    tab["Te2"] = [int(x) for x in hand["test_bins"][:, 1]]
    tab["Summe Test"] = hand["test_sums"]
if hand_step >= 2:
    tab["Minima mit Best Fit"] = hand["second_costs"]
st.dataframe(tab, hide_index=True, width="stretch")
st.markdown(T.E0_SCHRITTE[hand_step])
if hand_step >= 1:
    g = hand["greedy"][: min(hand_step, 2)]
    st.markdown("Gierige Menge bisher: **" + ", ".join(names[j] for j in g) + f"**, Summe der Instanz-Minima auf dem Training: **{hand['greedy_steps'][len(g) - 1]}** Behälter.")
if hand_step >= 3:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best Fit allein", f"{hand['bf']} Behälter", delta="Referenz", delta_color="off")
    m2.metric("Orakelwahl", f"{hand['oracle']} Behälter", delta=f"{hand['oracle'] - hand['bf']:+d} gegen Best Fit", delta_color="inverse")
    m3.metric("Merkmalswahl", f"{hand['feature']} Behälter", delta=f"{hand['feature'] - hand['bf']:+d} gegen Best Fit", delta_color="inverse")
    m4.metric("feste Wahl", f"{hand['fixed']} Behälter", delta=f"{hand['fixed'] - hand['bf']:+d} gegen Best Fit" if hand["fixed"] != hand["bf"] else "wie Best Fit",
              delta_color="inverse" if hand["fixed"] != hand["bf"] else "off")
    st.plotly_chart(V.build_hand(["Best Fit", "Orakelwahl {BF, WF}", "Merkmalswahl", "feste Wahl", "Orakelwahl {BF, S1}"],
                                 [hand["bf"], hand["oracle"], hand["feature"], hand["fixed"], hand["topk_oracle"]], hand["bf"]),
                    width="stretch", key="hand_chart")
    st.caption(T.E0_ZAEHLUNG.format(pool=hand["runs"]["pool"], laeufe=hand["runs"]["pool"] * len(E.HAND_TRAIN), orakel=hand["runs"]["orakel"],
                                    portfolio=hand["runs"]["portfolio"]))
    getattr(st, v_hand[0])(v_hand[1])

# --- Kernfragen -------------------------------------------------------------------------------------------------------------------------------
st.markdown("## ❓ Kernfragen")
st.markdown(f"**1. {T.KERNFRAGEN[0]}**")
k1, k2 = st.columns(2)
k1.metric("Training, dieser Lauf", f"{de_pm(run['train'])} Punkte", delta="gegen Best Fit", delta_color="off")
k2.metric("200 frische Instanzen, dieser Lauf", f"{de_pm(run['test'])} Punkte", delta="gegen Best Fit", delta_color="off")
getattr(st, v_zerl[0])(v_zerl[1])
getattr(st, v_lauf[0])(v_lauf[1])
st.markdown(f"**2. {T.KERNFRAGEN[1]}**")
k3, k4 = st.columns(2)
k3.metric(f"Orakelwahl über {k} Regeln, dieser Lauf", f"{de_pm(live['orakel'])} Punkte", delta=f"{M.runs_per_instance('orakel', k)} Läufe je Instanz", delta_color="off")
k4.metric("Merkmalswahl, dieser Lauf", f"{de_pm(live['merkmal'])} Punkte", delta="1 Lauf je Instanz", delta_color="off")
getattr(st, v_menge[0])(v_menge[1])
getattr(st, v_viel[0])(v_viel[1])
st.markdown(f"**3. {T.KERNFRAGEN[2]}**")
getattr(st, v_wechsel[0])(v_wechsel[1])

st.markdown("---")

# --- Experiment 1 -----------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 1: Was eine Variation anrichtet")
st.markdown(T.E1_INTRO)
st.caption(f"Elternteil: {C.PARENT_LABELS[elternteil]} = {var['parent_rule']}; auf dem Training {de_pm(var['parent_train'])} Punkte gegen Best Fit. "
           f"Familie »{fam_label}«, Zufalls-Seed der Läufe {lauf}, {var['n']} Kinder, davon {var['distinct']} verschiedene Regeln.")
e1a, e1b = st.columns(2)
labels_c = ["besser", "gleiche Behälterzahlen", "anders, gleich gut", "schlechter"]
e1a.plotly_chart(V.build_children([c[x] for x in VA.CLASSES], labels_c), width="stretch", key="e1_bar")
if var["points"]:
    e1b.plotly_chart(V.build_children_scatter(var["points"]), width="stretch", key="e1_scatter")
else:
    e1b.info("Kein Kind ist auf dem Training besser als der Elternteil; das Streudiagramm bleibt leer.")
m1, m2, m3 = st.columns(3)
m1.metric("Kinder besser als der Elternteil", f"{pct(c['besser'], var['n'])} %", delta=f"{c['besser']} von {var['n']}", delta_color="off")
m2.metric("davon auch auf dem Test besser", f"{var['better_test']} von {c['besser']}", delta="200 frische Instanzen", delta_color="off")
m3.metric("Kinder besser als Best Fit (Training)", f"{pct(var['better_bf'], var['n'])} %", delta=f"{var['better_bf']} von {var['n']}", delta_color="off")
st.dataframe(pd.DataFrame([
    {"Operatorart": VA.KIND_LABELS[kd], "Kinder": var["table"][kd]["n"], "besser": var["table"][kd]["besser"],
     "gleiche Behälterzahlen": var["table"][kd]["gleich"], "anders, gleich gut": var["table"][kd]["gleichwertig"], "schlechter": var["table"][kd]["schlechter"],
     "Anteil besser (%)": (100.0 * var["table"][kd]["besser"] / var["table"][kd]["n"]) if var["table"][kd]["n"] else None}
    for kd in VA.KINDS]), hide_index=True, width="stretch",
    column_config={"Anteil besser (%)": st.column_config.NumberColumn(format="%.1f %%")})
st.caption("Die Tabelle nach Operatorart gilt für diesen einen Trainingssatz. Die Summe über 20 Trainingssätze steht im Expander mit den eingefrorenen Läufen.")
getattr(st, v_var[0])(v_var[1])
vc = ev.variation_cell(PRE, fam, elternteil)
st.caption(f"Eingefroren, 20 Trainingssätze mit je 300 Kindern, Familie »{fam_label}«, Elternteil {C.PARENT_LABELS[elternteil]}: {ganz(vc['besser'])} von {ganz(vc['n'])} "
           f"Kindern besser ({pct(vc['besser'], vc['n'])} %), {ganz(vc['gleich'])} mit denselben Behälterzahlen ({pct(vc['gleich'], vc['n'])} %); von den besseren "
           f"{ganz(vc['better_test'])} auch auf den Testinstanzen besser.")

st.markdown("---")

# --- Experiment 2 -----------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 2: Variation allein oder Auswahl der Besten?")
st.markdown(T.E2_INTRO)
st.markdown(T.ARME_KOPF + "\n".join(f"| {a} | {C.SPACE_LABELS[AR.ARMS[a][0]]} | {AR.ARMS[a][3]} |" for a in C.ARMS))
st.plotly_chart(V.build_curve(run["points"], budget, C.TOL_TEST), width="stretch", key=f"e2_curve_{arm}_{budget}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Training", f"{de_pm(run['train'])} Punkte", delta="gegen Best Fit", delta_color="off")
m2.metric("Test", f"{de_pm(run['test'])} Punkte", delta="gegen Best Fit", delta_color="off")
m3.metric("verschiedene Regeln im Pool", ganz(run["pool"]), delta=f"von {ganz(run['stat']['n'])} Auswertungen", delta_color="off")
m4.metric("Auswertungen besser als Best Fit (Training)", f"{pct(run['stat']['better'], run['stat']['n'])} %", delta=f"{run['stat']['better']} von {ganz(run['stat']['n'])}", delta_color="off")
knoten = f" Der Baum der Endregel hat {run['knoten']} Knoten." if run["knoten"] is not None else ""
st.caption(f"Endregel dieses Laufs: {run['rule']}.{knoten} Instanz-Bilanz gegen Best Fit auf den 200 Testinstanzen: {run['bilanz'][0]} mit weniger, "
           f"{run['bilanz'][1]} mit gleich vielen, {run['bilanz'][2]} mit mehr Behältern. {run['stat']['eq_bf']} der {ganz(run['stat']['n'])} Auswertungen "
           "trafen eine Regel mit denselben Trainingsbehälterzahlen wie Best Fit.")
getattr(st, v_lauf[0])(v_lauf[1])
st.dataframe(pd.DataFrame([
    {"Vergleich (B gegen A)": f"{r['a']} → {r['b']}", "misst": r["art"],
     "Training: Median B − A": de_pm(r["train"]["median"]), "Training: Läufe B besser": f"{r['train']['neg']} von {r['n']}", "Training: Urteil": r["train"]["urteil"],
     "Test: Median B − A": de_pm(r["test"]["median"]), "Test: Läufe B besser": f"{r['test']['neg']} von {r['n']}", "Test: Urteil": r["test"]["urteil"]}
    for r in rows_zerl]), hide_index=True, width="stretch")
st.caption(f"Eingefroren: 20 Läufe je Arm, Familie »{fam_label}«, Bestwert nach {ganz(budget)} Auswertungen, gepaart je Lauf (gleiche Trainingsinstanzen). "
           "Negativ heißt: B ist besser als A. Das Budget folgt dem Regler.")
getattr(st, v_zerl[0])(v_zerl[1])
st.markdown(T.E2_ZAEHLEINHEIT)
drei_key = (fam, lauf, budget, test_seed)
if budget > C.THREE_ARM_BUDGET_MAX:
    st.caption(f"Der Knopf rechnet drei Läufe und ist deshalb auf Budgets bis {C.THREE_ARM_BUDGET_MAX} Auswertungen begrenzt. Den Regler »Budget« senken, um ihn zu benutzen.")
if st.button("Die drei Schablonen-Arme mit diesem Budget rechnen (Z-S, ZW-S, EA-S)", key="drei_button", disabled=budget > C.THREE_ARM_BUDGET_MAX):
    with st.spinner(f"Rechne drei Läufe mit je {ganz(budget)} Auswertungen …"):
        st.session_state["drei_arme"] = (drei_key, {a: _run_view(fam, a, lauf, budget, test_seed) for a in ("Z-S", "ZW-S", "EA-S")})
drei = st.session_state.get("drei_arme")
if drei is not None and drei[0] == drei_key:
    st.dataframe(pd.DataFrame([{"Arm": C.ARM_LABELS[a], "Training (Punkte gegen Best Fit)": de_pm(r["train"]), "Test (Punkte gegen Best Fit)": de_pm(r["test"]),
                                "verschiedene Regeln": r["pool"], "Endregel": r["rule"]} for a, r in drei[1].items()]), hide_index=True, width="stretch")
    st.caption("Drei Läufe auf denselben Trainingsinstanzen, je ein Lauf je Arm: ein Beispiel, kein Urteil.")

st.markdown("---")

# --- Experiment 3 -----------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 3: Heuristik-Mengen mit Orakelwahl, Merkmalswahl und fester Wahl")
st.markdown(T.E3_INTRO)
if arm not in C.POOL_ARMS:
    st.info(f"Der Pool stammt aus dem Arm {arm}. Ohne Auswahl der Besten enthält er überwiegend Regeln, die nie nach ihrer Güte gewählt wurden; die Menge wird trotzdem gebildet.")
cur = sv["types"][menge]
order = [wahl] + [w for w in C.CHOICES if w != wahl]
st.plotly_chart(V.build_choices([C.CHOICE_SHORT[w] for w in order], [cur["deltas"][w] for w in order], [cur["deltas"][w] - sv["ref"] for w in order],
                                [M.runs_per_instance(w, cur["size"]) for w in order], C.TOL_TEST), width="stretch", key=f"e3_bar_{menge}_{k}_{wahl}")
if "members" in cur:
    st.markdown(f"**Mitglieder der Menge** ({C.SET_LABELS[menge]}, k = {k}; Summe der Behälter auf den 20 Trainingsinstanzen, Best Fit: {sv['bf_train_sum']}):")
    st.dataframe(pd.DataFrame([{"Regel": r, "Behälter auf dem Training": s_} for r, s_ in cur["members"]]), hide_index=True, width="stretch")
    b = cur["bilanz"]
    nb = sum(b)
    st.caption(f"Instanz-Bilanz der Orakelwahl gegen Best Fit auf der Testmenge »{ts_label}«: {pct(b[0], nb)} % der {nb} Instanzen mit weniger, "
               f"{pct(b[1], nb)} % mit gleich vielen, {pct(b[2], nb)} % mit mehr Behältern.")
else:
    st.caption(f"Bei »{C.SET_LABELS[menge]}« stehen hier Mittel über {C.N_RANDOM_SETS} Zufallsmengen aus dem Pool; einzelne Mitglieder werden nicht gezeigt.")
st.caption(f"Zwei Zähleinheiten, nie addiert: {ganz(budget)} Auswertungen für den Pool ({sv['pool']} verschiedene Regeln) und im Einsatz "
           f"{M.runs_per_instance('orakel', cur['size'])} Läufe je Instanz bei der Orakelwahl, 1 Lauf je Instanz bei Merkmalswahl und fester Wahl. "
           f"Die beste Einzelregel dieses Laufs liegt auf dieser Testmenge bei {de_pm(sv['ref'])} Punkten gegen Best Fit. "
           "Die eingefrorenen Zellen im Verdikt stammen aus Pools mit 1 000 Auswertungen.")
getattr(st, v_menge[0])(v_menge[1])

st.markdown("---")

# --- Experiment 4 -----------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 4: Was die Menge kostet: Vielfalt, Güte und Einsatzaufwand")
st.markdown(T.E4_INTRO)
typen = list(C.SET_TYPES) + ["portfolio"]
st.plotly_chart(V.build_set_types([C.SET_LABELS_LONG[t] for t in typen], [sv["types"][t]["deltas"]["orakel"] for t in typen], typen.index(menge), C.TOL_TEST),
                width="stretch", key=f"e4_bar_{menge}_{k}")
st.dataframe(pd.DataFrame([
    {"Menge": C.SET_LABELS_LONG[t], "Orakelwahl gegen Best Fit, dieser Lauf": de_pm(sv["types"][t]["deltas"]["orakel"]),
     "Median über 20 eingefrorene Läufe": de_pm(med_port if t == "portfolio" else med[t]),
     "Läufe je Instanz": sv["types"][t]["size"], "Auswertungen im Training": 3 if t == "portfolio" else budget,
     "Mitglieder mit den Trainingsbehälterzahlen von Best Fit": de(sv["types"][t]["eq_bf"], 1)}
    for t in typen]), hide_index=True, width="stretch")
sp = _spielraum(fam, test_seed)
st.caption(f"Vielfalt der Familie »{fam_label}« auf den 200 Testinstanzen (60 Gegenstände): First Fit und Best Fit brauchen auf {sp['ff_lt_bf'] + sp['ff_gt_bf']} "
           f"von {sp['n']} Instanzen verschieden viele Behälter. Das beschreibt die Familie; eine Ursache für den Gewinn der Menge ist damit nicht belegt.")
getattr(st, v_viel[0])(v_viel[1])
getattr(st, v_pool[0])(v_pool[1])

st.markdown("---")

# --- Experiment 5 -----------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 5: Wo es endet: andere Instanzen und mehr Trainingsinstanzen")
st.markdown(T.E5_INTRO)
rows5 = []
for ts in C.TESTSETS:
    for w in C.CHOICES:
        z = ev.cell(ev.set_deltas(PRE, fam, arm, "greedy", k, w, ts))
        zb = ev.cell(ev.set_deltas(PRE, fam, arm, "greedy", k, w, ts, block="m3")) if ts == "same60" else None
        rows5.append({"Testmenge": C.testset_label(ts, fam), "Wahl": C.CHOICE_SHORT[w], "Läufe je Instanz": M.runs_per_instance(w, k),
                      "20 Trainingsinstanzen: Median": de_pm(z["median"]), "Läufe unter null": f"{z['neg']} von {z['n']}", "Urteil": z["urteil"],
                      "100 Trainingsinstanzen: Median": de_pm(zb["median"]) if zb else "nicht gemessen",
                      "Läufe unter null (100)": f"{zb['neg']} von {zb['n']}" if zb else "nicht gemessen"})
st.dataframe(pd.DataFrame(rows5), hide_index=True, width="stretch")
st.caption(f"Eingefroren: gierige Menge aus dem Pool {arm} (1 000 Auswertungen), k = {k}, auf »{fam_label}« trainiert, Delta gegen Best Fit in Prozentpunkten. "
           "Mit 100 Trainingsinstanzen gibt es 10 Läufe für die Arme Z-S, EA-S und GP-T auf zwei Familien und nur für die Testmenge der gleichen Familie; "
           "bei 10 Läufen steht kein Urteil. Die Testmenge mit 120 Gegenständen hat 100 Instanzen, alle anderen 200.")
getattr(st, v_wechsel[0])(v_wechsel[1])

st.markdown("---")

# --- Alle eingefrorenen Läufe -----------------------------------------------------------------------------------------------------------------
with st.expander("🔬 Alle Arme und Mengen, 20 Läufe (eingefroren)"):
    st.markdown("**Arme gegen Best Fit** (Bestwert nach 1 000 Auswertungen, Test auf 200 Instanzen der gleichen Familie, Delta in Prozentpunkten)")
    at_rows = ev.arm_table(PRE)
    st.dataframe(pd.DataFrame([{"Familie": C.FAMILY_LABELS[r["familie"]], "Arm": r["arm"], "Training: Median": de_pm(r["train"]),
                                "Test: Median": de_pm(r["test"]), "Läufe besser": f"{r['neg']} von {r['n']}", "Läufe schlechter": f"{r['pos']} von {r['n']}",
                                "Urteil": r["urteil"]} for r in at_rows]), hide_index=True, width="stretch")
    cl = ev.count_labels(at_rows)
    st.caption(f"{len(at_rows)} Zellen: {cl['besser']} »besser«, {cl['gleich']} »gleich«, {cl['schlechter']} »schlechter«.")
    st.markdown("**Mengen mit Orakelwahl** (Pools mit Auswahl der Besten, Testmenge der gleichen Familie, Median über 20 Läufe)")
    rows_m = []
    for f in C.FAMILIES:
        for a in C.POOL_ARMS:
            for kk in (3, 5):
                row = {"Familie": C.FAMILY_LABELS[f], "Pool": a, "k": kk}
                for t in C.SET_TYPES_FROZEN:
                    z = ev.cell(ev.set_deltas(PRE, f, a, t, kk, "orakel"))
                    row[C.SET_LABELS_LONG[t]] = f"{de_pm(z['median'])} ({z['neg']}/20, {z['urteil']})"
                zm = ev.cell(ev.set_deltas(PRE, f, a, "greedy", kk, "merkmal"))
                row["greedy mit Merkmalswahl"] = f"{de_pm(zm['median'])} ({zm['pos']}/20 schlechter, {zm['urteil']})"
                rows_m.append(row)
    st.dataframe(pd.DataFrame(rows_m), hide_index=True, width="stretch")
    st.caption("Je Zelle: Median des Deltas gegen Best Fit, Läufe unter null von 20, Urteil. Bei den Zufallsmengen zählt je Lauf das Mittel über fünf Mengen.")
    st.markdown("**Kinder der Variation** (20 Trainingssätze mit je 300 Kindern je Zelle)")
    rows_v = []
    for f in C.FAMILIES:
        for p in C.PARENTS:
            z = ev.variation_cell(PRE, f, p)
            rows_v.append({"Familie": C.FAMILY_LABELS[f], "Elternteil": C.PARENT_LABELS[p], "Kinder": z["n"], "besser (%)": 100.0 * z["besser"] / z["n"],
                           "gleiche Behälterzahlen (%)": 100.0 * z["gleich"] / z["n"], "auch auf dem Test besser": f"{z['better_test']} von {z['besser']}",
                           "besser als Best Fit (%)": 100.0 * z["better_bf"] / z["n"]})
    st.dataframe(pd.DataFrame(rows_v), hide_index=True, width="stretch",
                 column_config={c_: st.column_config.NumberColumn(format="%.1f %%") for c_ in ("besser (%)", "gleiche Behälterzahlen (%)", "besser als Best Fit (%)")})

# --- Grenzen ----------------------------------------------------------------------------------------------------------------------------------
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(T.GRENZEN)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(T.MATHE)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Hyperheuristiken: Auswählen und Erzeugen](https://sebastianhanisch.net/konzepte-hyperheuristiken.html)."
)
