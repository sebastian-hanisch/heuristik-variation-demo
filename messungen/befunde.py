"""Befundzeilen der README, aus `hva_precomputed.json` gerechnet. Nicht Teil der Laufzeit-Importe.

Jede Zeile der Befundliste in der README ist eine Ausgabe dieser Datei; `tests/test_claims.py` rechnet die Zeilen neu und sucht sie wörtlich
in der README. Aufruf (im Demo-Ordner): python messungen/befunde.py
Reihenfolge der Familien in Dreiergruppen: gleichverteilt / Häufungen / breit gestreut. Delta in Prozentpunkten gegen Best Fit, negativ = besser."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hva_constants as C  # noqa: E402
import hva_evaluation as ev  # noqa: E402
import hva_variation as V  # noqa: E402

F = C.FAMILIES
med = ev.median
pm = ev.de_pm
de = ev.de


def tri(values, fmt=pm):
    return " / ".join(fmt(v) for v in values)


def zaehl(labels):
    return f"„besser“ in {labels.count('besser')}, „gleich“ in {labels.count('gleich')}, „schlechter“ in {labels.count('schlechter')}"


def lines(pre):
    out = {}
    # --- Arme gegen Best Fit --------------------------------------------------------------------------------------------------------
    rows = ev.arm_table(pre)
    cl = ev.count_labels(rows)
    out["arme"] = (f"**Arme gegen Best Fit** (Bestwert nach 1 000 Auswertungen, 18 Zellen aus 6 Armen und 3 Familien, je 20 Läufe): {cl['besser']} Zellen „besser“, "
                   f"{cl['gleich']} „gleich“, {cl['schlechter']} „schlechter“. Der Median liegt auf dem Training zwischen {pm(min(r['train'] for r in rows))} und "
                   f"{pm(max(r['train'] for r in rows))} Punkten, auf den 200 frischen Instanzen zwischen {pm(min(r['test'] for r in rows))} und {pm(max(r['test'] for r in rows))}.")
    alle = [r for fam in F for arm in C.ARMS for r in pre["m1"][fam][arm]]
    out["laeufe"] = (f"Von den {len(alle)} Läufen liegen auf dem Training {sum(r['train'] < 0 for r in alle)} unter null (besser als Best Fit), auf den frischen Instanzen "
                     f"{sum(r['test'] < 0 for r in alle)}. Läufe, die dort um mindestens 0,2 Punkte besser sind als Best Fit: {sum(r['test'] <= -0.2 for r in alle)}; "
                     f"Läufe, die um mindestens 0,2 Punkte schlechter sind: {sum(r['test'] >= 0.2 for r in alle)}.")
    for arm in C.ARMS:
        out[f"arm_{arm}"] = (f"{arm}: Training {tri(med(ev.cell_deltas(pre['m1'][f][arm], None, 'train')) for f in F)}, "
                             f"Test {tri(med(ev.cell_deltas(pre['m1'][f][arm])) for f in F)}")
    # --- Variation gegen Auswahl ----------------------------------------------------------------------------------------------------
    comp = {fam: ev.comparison_rows(pre, fam, 1000) for fam in F}
    tr = [r["train"]["urteil"] for fam in F for r in comp[fam]]
    te = [r["test"]["urteil"] for fam in F for r in comp[fam]]
    out["vergleiche"] = (f"**21 Vergleiche** (7 je Familie, gepaart je Lauf, 1 000 Auswertungen): auf den frischen Instanzen {zaehl(te)}; auf dem Training {zaehl(tr)}.")
    stufen = {w: [r[w]["urteil"] for b in C.CHECKPOINTS for fam in F for r in ev.comparison_rows(pre, fam, b)] for w in ("train", "test")}
    out["stufen"] = (f"Über alle zehn Budgetstufen von 100 bis 1 000 Auswertungen ({len(stufen['test'])} Zellen je Seite): auf den frischen Instanzen {zaehl(stufen['test'])}; "
                     f"auf dem Training {zaehl(stufen['train'])}.")

    def zeile(a, b):
        return [next(r for r in comp[fam] if (r["a"], r["b"]) == (a, b)) for fam in F]

    for key, name, pairs in (("variation", "Variation allein (frisch ziehen → Variation ohne Auswahl)", (("Z-S", "ZW-S"), ("Z-T", "GP0-T"))),
                             ("auswahl", "Auswahl der Besten (Variation ohne → mit Auswahl)", (("ZW-S", "EA-S"), ("GP0-T", "GP-T")))):
        s_, t_ = zeile(*pairs[0]), zeile(*pairs[1])
        labs = [r["train"]["urteil"] for r in s_ + t_]
        labs_te = [r["test"]["urteil"] for r in s_ + t_]
        out[key] = (f"**{name}**, Median der Differenz auf dem Training: Schablonen {tri(r['train']['median'] for r in s_)} "
                    f"(Läufe mit Vorteil {tri((r['train']['neg'] for r in s_), str)} von 20), Bäume {tri(r['train']['median'] for r in t_)} "
                    f"(Läufe mit Vorteil {tri((r['train']['neg'] for r in t_), str)} von 20); „besser“ in {labs.count('besser')} von 6 Zellen, „schlechter“ in "
                    f"{labs.count('schlechter')}. Auf den frischen Instanzen: Schablonen {tri(r['test']['median'] for r in s_)}, Bäume {tri(r['test']['median'] for r in t_)}; "
                    f"„besser“ in {labs_te.count('besser')} von 6 Zellen.")
    op = zeile("EA-S", "GP-T")
    out["operator"] = (f"**Vorlagen-Operator gegen Baumkreuzung** (EA-S → GP-T): Training {tri(r['train']['median'] for r in op)} "
                       f"({zaehl([r['train']['urteil'] for r in op])} von 3 Zellen), frische Instanzen {tri(r['test']['median'] for r in op)} ({zaehl([r['test']['urteil'] for r in op])}).")
    kmin = ev.bonferroni_k(20, 42)
    mx = max(max(r[w]["neg"], r[w]["pos"]) for fam in F for r in comp[fam] for w in ("train", "test"))
    out["bonferroni"] = (f"Mehrfachvergleich: Mit Bonferroni-Ausgleich über die 42 Zellen (21 Vergleiche, Training und Test) bräuchte der einseitige Vorzeichentest "
                         f"{kmin} von 20 Läufen auf einer Seite; die größte Zählung in den 42 Zellen ist {mx}.")
    # --- Kandidaten -----------------------------------------------------------------------------------------------------------------
    def anteil(arm, key):
        return [100.0 * sum(r["stat"][key] for r in pre["m1"][f][arm]) / sum(r["stat"]["n"] for r in pre["m1"][f][arm]) for f in F]

    out["kandidaten"] = ("**Kandidaten** (Anteil der 20 000 Auswertungen je Zelle): mit denselben Trainingsbehälterzahlen wie Best Fit "
                         + "; ".join(f"{a} {tri(anteil(a, 'eq_bf'), lambda v: de(v, 1))} %" for a in C.ARMS)
                         + ". Auf dem Training besser als Best Fit: " + "; ".join(f"{a} {tri(anteil(a, 'better'), lambda v: de(v, 1))} %" for a in C.ARMS) + ".")
    out["pool"] = ("Verschiedene Regeln im Pool, Mittel über 20 Läufe: "
                   + "; ".join(f"{a} {tri((sum(r['pool'] for r in pre['m1'][f][a]) / 20 for f in F), lambda v: de(v, 0))}" for a in C.ARMS) + ".")
    # --- Mengen ---------------------------------------------------------------------------------------------------------------------
    def sd(fam, arm, typ, k, wahl, ts="same60", ref="bf", block="m1"):
        return ev.set_deltas(pre, fam, arm, typ, k, wahl, ts, ref, block)

    for k in (3, 5):
        vb = [med(sd(f, a, "greedy", k, "orakel")) for a in C.POOL_ARMS for f in F]
        vr = [med(sd(f, a, "greedy", k, "orakel", ref="regel")) for a in C.POOL_ARMS for f in F]
        lab = [ev.urteil(sd(f, a, "greedy", k, "orakel"))[0] for a in C.POOL_ARMS for f in F]
        out[f"greedy{k}"] = (f"**Gierige Menge mit Orakelwahl, k = {k}** (Pools von EA-S und GP-T, 6 Zellen, {k} Läufe je Instanz): gegen Best Fit EA-S {tri(vb[:3])}, "
                             f"GP-T {tri(vb[3:])}; {zaehl(lab)} Zellen. Gegen die beste Einzelregel desselben Laufs: zwischen {pm(min(vr))} und {pm(max(vr))} Punkten.")
    for wahl, name in (("merkmal", "Merkmalswahl"), ("fest", "Feste Wahl")):
        lb = [ev.urteil(sd(f, a, "greedy", k, wahl))[0] for a in C.POOL_ARMS for f in F for k in (3, 5)]
        lr = [ev.urteil(sd(f, a, "greedy", k, wahl, ref="regel"))[0] for a in C.POOL_ARMS for f in F for k in (3, 5)]
        vb = [med(sd(f, a, "greedy", 3, wahl)) for a in C.POOL_ARMS for f in F]
        out[wahl] = (f"**{name} derselben Mengen** (1 Lauf je Instanz; k = 3 und 5, 12 Zellen): gegen Best Fit {zaehl(lb)} Zellen; gegen die beste Einzelregel {zaehl(lr)}. "
                     f"Mediane bei k = 3 gegen Best Fit: EA-S {tri(vb[:3])}, GP-T {tri(vb[3:])}.")
    for typ, name in (("topk", "Die k nach Trainingsfitness besten Regeln (topk)"), ("zufall", "k beliebige Regeln des Pools (Mittel über fünf Zufallsmengen je Lauf)"),
                      ("bfplus", "Best Fit plus k − 1 beliebige Regeln des Pools")):
        vb = [med(sd(f, a, typ, 3, "orakel")) for a in C.POOL_ARMS for f in F]
        lab = [ev.urteil(sd(f, a, typ, k, "orakel"))[0] for a in C.POOL_ARMS for f in F for k in (3, 5)]
        out[typ] = (f"**{name}**, Orakelwahl gegen Best Fit, k = 3: EA-S {tri(vb[:3])}, GP-T {tri(vb[3:])}; über k = 3 und 5 (12 Zellen) {zaehl(lab)}.")
    quelle = {f: {a: med(sd(f, a, "greedy", 3, "orakel")) for a in C.ARMS} for f in F}
    vorn = sum(1 for f in F if quelle[f]["Z-S"] < quelle[f]["EA-S"])
    out["poolquelle"] = ("**Woher der Pool kommt** (gierige Menge, k = 3, Orakelwahl gegen Best Fit, Median über 20 Läufe): "
                         + "; ".join(f"{a} {tri(quelle[f][a] for f in F)}" for a in C.ARMS)
                         + f". In {vorn} von 3 Familien liegt die Menge aus dem Pool des frischen Ziehens (Z-S) vor der aus dem Pool von EA-S; "
                         f"alle 18 Zellen: {zaehl([ev.urteil(sd(f, a, 'greedy', 3, 'orakel'))[0] for a in C.ARMS for f in F])}.")
    out["portfolio"] = (f"**Best Fit, First Fit, Worst Fit als feste Menge** (3 Läufe je Instanz, 3 Auswertungen im Training): Orakelwahl gegen Best Fit "
                        f"{tri(med(ev.portfolio_deltas(pre, f)) for f in F)} Punkte.")
    ratio = [100.0 * med(sd(f, a, "greedy", k, "orakel")) / med(ev.set_train_deltas(pre, f, a, "greedy", k)) for a in C.POOL_ARMS for f in F for k in (3, 5)]
    trn = [med(ev.set_train_deltas(pre, f, a, "greedy", 3)) for a in C.POOL_ARMS for f in F]
    out["training_test"] = (f"**Training gegen Test der gierigen Menge** (Orakelwahl gegen Best Fit): auf dem Training bei k = 3 EA-S {tri(trn[:3])}, GP-T {tri(trn[3:])}; "
                            f"der Testgewinn beträgt in den 12 Zellen (k = 3 und 5) zwischen {de(min(ratio), 0)} und {de(max(ratio), 0)} % des Trainingsgewinns.")
    # --- andere Instanzen -----------------------------------------------------------------------------------------------------------
    andere = [ts for ts in C.TESTSETS if ts != "same60"]
    lab = {(a, f, k, ts): ev.urteil(sd(f, a, "greedy", k, "orakel", ts))[0] for a in C.POOL_ARMS for f in F for k in (3, 5) for ts in andere}
    groesse = [v for (a, f, k, ts), v in lab.items() if ts in ("n30", "n120")]
    fremd = [v for (a, f, k, ts), v in lab.items() if ts.startswith("fremd")]
    clu = [v for (a, f, k, ts), v in lab.items() if ts.startswith("fremd") and f == "clu3"]
    out["wechsel"] = (f"**Andere Instanzen** (gierige Menge, Orakelwahl gegen Best Fit, 2 Pools, 3 Familien, k = 3 und 5): gleiche Familie mit 30 oder 120 Gegenständen "
                      f"(24 Zellen) {zaehl(groesse)}; fremde Familien (24 Zellen) {zaehl(fremd)}; davon auf Häufungen trainiert (8 Zellen) {zaehl(clu)}.")
    mk = [ev.urteil(sd(f, a, "greedy", k, "merkmal", ts))[0] for a in C.POOL_ARMS for f in F for k in (3, 5) for ts in C.TESTSETS]
    out["merkmal_alle"] = f"Merkmalswahl der gierigen Menge gegen Best Fit auf allen fünf Testmengen (60 Zellen): {zaehl(mk)}."
    for arm in C.POOL_ARMS:
        teile = []
        for f in C.BIG_FAMILIES:
            d100, d20 = sd(f, arm, "greedy", 3, "orakel", block="m3"), sd(f, arm, "greedy", 3, "orakel")
            m100 = sd(f, arm, "greedy", 3, "merkmal", block="m3")
            teile.append(f"{C.FAMILY_LABELS[f]}: Orakelwahl {pm(med(d100))} ({sum(x < 0 for x in d100)} von 10 Läufen unter null; mit 20 Trainingsinstanzen {pm(med(d20))}), "
                         f"Merkmalswahl {pm(med(m100))} ({sum(x > 0 for x in m100)} von 10 Läufen über null)")
        out[f"gross_{arm}"] = f"**100 Trainingsinstanzen, Pool {arm}** (10 Läufe, gierige Menge, k = 3, gegen Best Fit): " + "; ".join(teile) + "."
    teile = []
    for arm in C.BIG_ARMS:
        teile.append(f"{arm} " + " und ".join(f"{pm(med([r['train'] for r in pre['m3'][f][arm]]))} (mit 20: {pm(med([r['train'] for r in pre['m1'][f][arm]]))})" for f in C.BIG_FAMILIES))
    out["gross_regel"] = ("Einzelregel mit 100 Trainingsinstanzen, Median auf dem Training (gleichverteilt und breit gestreut): " + "; ".join(teile)
                          + ". Auf den frischen Instanzen: "
                          + "; ".join(f"{arm} " + " und ".join(pm(med([r['test'] for r in pre['m3'][f][arm]])) for f in C.BIG_FAMILIES) for arm in C.BIG_ARMS) + ".")
    # --- Variation ------------------------------------------------------------------------------------------------------------------
    for p in C.PARENTS:
        z = [ev.variation_cell(pre, f, p) for f in F]
        out[f"kinder_{p}"] = (f"**Elternteil {C.PARENT_LABELS[p]}** (6 000 Kinder je Familie aus 20 Trainingssätzen): auf dem Training besser "
                              f"{tri((100.0 * x['besser'] / x['n'] for x in z), lambda v: de(v, 1))} %, dieselben Behälterzahlen "
                              f"{tri((100.0 * x['gleich'] / x['n'] for x in z), lambda v: de(v, 1))} %, besser als Best Fit "
                              f"{tri((100.0 * x['better_bf'] / x['n'] for x in z), lambda v: de(v, 1))} %. Von den auf dem Training besseren Kindern sind auch auf den "
                              f"200 Testinstanzen besser als der Elternteil: {tri((ev.ganz(x['better_test']) + ' von ' + ev.ganz(x['besser']) for x in z), str)} "
                              f"({tri((100.0 * x['better_test'] / max(1, x['besser']) for x in z), lambda v: de(v, 0))} %). Der Elternteil ist in "
                              f"{tri((x['parent_worse'] for x in z), str)} von 20 Trainingssätzen schlechter als Best Fit.")
    zb = [ev.variation_cell(pre, f, "bf") for f in F]
    art = {k: (sum(x["art"][k]["besser"] for x in zb), sum(x["art"][k]["n"] for x in zb)) for k in V.KINDS}
    out["kinder_art"] = ("Nach Operatorart (Elternteil Best Fit, drei Familien zusammen): "
                         + "; ".join(f"{V.KIND_LABELS[k]} {art[k][0]} von {ev.ganz(art[k][1])} besser ({ev.pct(*art[k], 2)} %)" for k in V.KINDS) + ".")
    # --- Spielraum ------------------------------------------------------------------------------------------------------------------
    sp = {f: pre["spielraum"][f]["same60"] for f in F}
    out["spielraum"] = (f"**Spielraum** (200 Testinstanzen mit 60 Gegenständen, Überschuss über die untere Schranke): Best Fit {tri((sp[f]['BF'] for f in F), de)} %, "
                        f"First Fit {tri((sp[f]['FF'] for f in F), de)} %, Worst Fit {tri((sp[f]['WF'] for f in F), de)} %, FFD (offline) {tri((sp[f]['FFD'] for f in F), de)} %. "
                        f"First Fit und Best Fit brauchen auf {tri((sp[f]['ff_lt_bf'] + sp[f]['ff_gt_bf'] for f in F), str)} von 200 Instanzen verschieden viele Behälter.")
    m6 = pre["m6"]
    out["optimum"] = (f"Bei 10 Gegenständen (200 Instanzen je Familie, exaktes Optimum): untere Schranke im Mittel {tri((m6[f]['lb_unter_opt'] for f in F), de)} % unter dem Optimum, "
                      f"Best Fit {tri((m6[f]['bf_ueber_opt'] for f in F), de)} % darüber, Best Fit optimal auf {tri((m6[f]['bf_optimal'] for f in F), str)} von 200 Instanzen.")
    r7 = pre["m7"]["gl_20_70"]["EA-S"]
    out["zehntausend"] = ("**Stichprobe mit 10 000 Auswertungen** (EA-S, gleichverteilt, 2 Läufe, kein Befund): Einzelregel auf dem Training "
                          + " und ".join(pm(r["train"]) for r in r7) + ", auf den frischen Instanzen " + " und ".join(pm(r["test"]) for r in r7)
                          + "; gierige Menge mit k = 3 und Orakelwahl " + " und ".join(pm(r["sets"]["greedy"]["3"]["T"]["same60"][0]) for r in r7) + " Punkte.")
    return out


if __name__ == "__main__":
    for key, text in lines(ev.load_precomputed()).items():
        print(f"[{key}]\n{text}\n")
