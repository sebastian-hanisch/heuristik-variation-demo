"""Bau-Vormessung: alle Tabellen der Neumessung aus `hva_precomputed.json` (SplitMix64). Nicht Teil der Laufzeit-Importe.

Aufruf (im Demo-Ordner): python messungen/auswertung.py > messungen/ausgabe_auswertung.txt
Delta = Überschuss gegen Best Fit in Prozentpunkten (positiv = schlechter); Reihenfolge der Familien gl_20_70 / clu3 / stark."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hva_arms as AR  # noqa: E402
import hva_constants as C  # noqa: E402
import hva_evaluation as ev  # noqa: E402
import hva_variation as V  # noqa: E402

PRE = ev.load_precomputed()
med = ev.median
F = C.FAMILIES


def f2(x):
    return "%+.2f" % x


def zelle(deltas):
    lab, m, neg, pos = ev.urteil(deltas)
    return "%s (%d<0, %d>0, %s)" % (f2(m), neg, pos, lab if len(deltas) == 20 else "-")


print("== T1 Arme gegen Best Fit (Median Training / Test same60, 20 Läufe) ==")
for budget in (200, 1000):
    rows = ev.arm_table(PRE, budget)
    print("Budget %d: Urteile %s" % (budget, ev.count_labels(rows)))
    for arm in C.ARMS:
        print("  %-6s" % arm, " | ".join("%s Tr %s (%d<0) Te %s (%d<0, %d>0) %s" % (r["familie"], f2(r["train"]), r["train_neg"], f2(r["test"]), r["neg"], r["pos"], r["urteil"])
                                         for r in rows if r["arm"] == arm))
print("Spanne der Mediane bei 1000: Training %s bis %s, Test %s bis %s" % (
    f2(min(r["train"] for r in ev.arm_table(PRE))), f2(max(r["train"] for r in ev.arm_table(PRE))),
    f2(min(r["test"] for r in ev.arm_table(PRE))), f2(max(r["test"] for r in ev.arm_table(PRE)))))
alle = [(fam, arm, r) for fam in F for arm in C.ARMS for r in PRE["m1"][fam][arm]]
print("360 Läufe: Training <0 %d, =0 %d, >0 %d; Test <0 %d, =0 %d, >0 %d; Test <= -0,2: %d, >= +0,2: %d" % (
    sum(r["train"] < 0 for _, _, r in alle), sum(r["train"] == 0 for _, _, r in alle), sum(r["train"] > 0 for _, _, r in alle),
    sum(r["test"] < 0 for _, _, r in alle), sum(r["test"] == 0 for _, _, r in alle), sum(r["test"] > 0 for _, _, r in alle),
    sum(r["test"] <= -0.2 for _, _, r in alle), sum(r["test"] >= 0.2 for _, _, r in alle)))

print("\n== T2 Vergleiche B minus A, gepaart je Lauf (negativ = B besser) ==")
for budget in (200, 400, 1000):
    tot = {"train": {x: 0 for x in ev.LABELS}, "test": {x: 0 for x in ev.LABELS}}
    for fam in F:
        rows = ev.comparison_rows(PRE, fam, budget)
        for w in ("train", "test"):
            for lab, n in ev.count_labels(rows, w).items():
                tot[w][lab] += n
        if budget in (200, 1000):
            for r in rows:
                print("  %d %-8s %-6s->%-6s %-28s Training %s (%d<0, %d>0) %-10s | Test %s (%d<0, %d>0) %s" % (
                    budget, fam, r["a"], r["b"], r["art"], f2(r["train"]["median"]), r["train"]["neg"], r["train"]["pos"], r["train"]["urteil"],
                    f2(r["test"]["median"]), r["test"]["neg"], r["test"]["pos"], r["test"]["urteil"]))
    print("Budget %d, 21 Vergleiche: Training %s, Test %s" % (budget, tot["train"], tot["test"]))
print("Alle zehn Stufen, Test: besser/schlechter je Stufe:", [(b, sum(ev.count_labels(ev.comparison_rows(PRE, fam, b), "test")["besser"] for fam in F),
                                                           sum(ev.count_labels(ev.comparison_rows(PRE, fam, b), "test")["schlechter"] for fam in F)) for b in C.CHECKPOINTS])
print("Alle zehn Stufen, Training: besser/schlechter je Stufe:", [(b, sum(ev.count_labels(ev.comparison_rows(PRE, fam, b), "train")["besser"] for fam in F),
                                                               sum(ev.count_labels(ev.comparison_rows(PRE, fam, b), "train")["schlechter"] for fam in F)) for b in C.CHECKPOINTS])
print("Bonferroni: bei 42 Zellen (21 Vergleiche, Training und Test) braucht der einseitige Vorzeichentest %s von 20 Läufen; p(15 von 20) = %.4f; 42 x p = %.2f" % (
    ev.bonferroni_k(20, 42), ev.sign_test_p(15, 20), 42 * ev.sign_test_p(15, 20)))
mx = 0
for fam in F:
    for r in ev.comparison_rows(PRE, fam, 1000):
        mx = max(mx, r["train"]["neg"], r["train"]["pos"], r["test"]["neg"], r["test"]["pos"])
print("Größte Zählung auf einer Seite in den 42 Zellen bei 1000:", mx)

print("\n== T4 Kandidatenstatistik (Anteile der 20 000 Auswertungen je Zelle in %, Pool im Mittel) ==")
for arm in C.ARMS:
    out = []
    for fam in F:
        runs = PRE["m1"][fam][arm]
        n = sum(r["stat"]["n"] for r in runs)
        out.append("%s: =BF %.1f, besser %.1f, versch. %.0f, Treffer %.0f, Pool %.0f%s" % (
            fam, 100 * sum(r["stat"]["eq_bf"] for r in runs) / n, 100 * sum(r["stat"]["better"] for r in runs) / n,
            sum(r["stat"]["distinct"] for r in runs) / 20, sum(r["stat"]["dup"] for r in runs) / 20, sum(r["pool"] for r in runs) / 20,
            (", Knoten %.1f" % (sum(r["knoten"] for r in runs) / 20)) if runs[0]["knoten"] is not None else ""))
    print("  %-6s" % arm, " | ".join(out))

print("\n== T5 Mengen auf same60 (Median über 20 Läufe; gegen Best Fit / gegen die beste Einzelregel des Laufs) ==")
zaehl = {}
for arm in C.ARMS:
    for fam in F:
        for k in (3, 5):
            for typ in C.SET_TYPES_FROZEN:
                teile = []
                for w in C.CHOICES:
                    db = ev.set_deltas(PRE, fam, arm, typ, k, w)
                    dr = ev.set_deltas(PRE, fam, arm, typ, k, w, ref="regel")
                    teile.append("%s BF %s | Regel %s" % (w, zelle(db), zelle(dr)))
                    if arm in C.POOL_ARMS and typ == "greedy":
                        zaehl.setdefault((w, "bf"), []).append(ev.urteil(db)[0])
                        zaehl.setdefault((w, "regel"), []).append(ev.urteil(dr)[0])
                print("  %-6s %-8s k=%d %-7s Training %s || %s" % (arm, fam, k, typ, f2(med(ev.set_train_deltas(PRE, fam, arm, typ, k))), " || ".join(teile)))
print("greedy aus den Pools EA-S und GP-T, k = 3 und 5, same60 (12 Zellen):")
for key, labs in zaehl.items():
    print("  ", key, {x: labs.count(x) for x in ev.LABELS})
print("Spanne greedy k=3 Orakel gegen die Einzelregel (6 Zellen): %s bis %s; gegen Best Fit: %s bis %s" % (
    f2(min(med(ev.set_deltas(PRE, f, a, "greedy", 3, "orakel", ref="regel")) for f in F for a in C.POOL_ARMS)),
    f2(max(med(ev.set_deltas(PRE, f, a, "greedy", 3, "orakel", ref="regel")) for f in F for a in C.POOL_ARMS)),
    f2(min(med(ev.set_deltas(PRE, f, a, "greedy", 3, "orakel")) for f in F for a in C.POOL_ARMS)),
    f2(max(med(ev.set_deltas(PRE, f, a, "greedy", 3, "orakel")) for f in F for a in C.POOL_ARMS))))
print("Einzelregel der Pool-Arme gegen Best Fit (same60):", {(f, a): f2(med(ev.cell_deltas(PRE["m1"][f][a]))) for f in F for a in C.POOL_ARMS})
print("Anteil Zufall an greedy (Orakel gegen Best Fit, Verhältnis der Mediane):", {(f, a, k): "%.0f %%" % (
    100 * med(ev.set_deltas(PRE, f, a, "zufall", k, "orakel")) / med(ev.set_deltas(PRE, f, a, "greedy", k, "orakel"))) for f in F for a in C.POOL_ARMS for k in (3, 5)})

print("\n== T6 greedy mit Orakelwahl auf allen Testmengen (gegen Best Fit) ==")
for arm in C.POOL_ARMS:
    for fam in F:
        for k in (3, 5):
            print("  %-5s %-8s k=%d" % (arm, fam, k), " | ".join("%s %s" % (ts, zelle(ev.set_deltas(PRE, fam, arm, "greedy", k, "orakel", ts))) for ts in C.TESTSETS))
n_gl = n_all = 0
ausn = []
for arm in C.POOL_ARMS:
    for fam in F:
        for ts in C.TESTSETS:
            lab = ev.urteil(ev.set_deltas(PRE, fam, arm, "topk", 5, "orakel", ts, ref="regel"))[0]
            n_all += 1
            n_gl += lab == "gleich"
            if lab != "gleich":
                ausn.append((arm, fam, ts, lab, f2(med(ev.set_deltas(PRE, fam, arm, "topk", 5, "orakel", ts, ref="regel")))))
print("topk k=5 mit Orakelwahl gegen die Einzelregel: gleich in %d von %d Zellen; Ausnahmen %s" % (n_gl, n_all, ausn))
for w in ("merkmal", "fest"):
    labs = [ev.urteil(ev.set_deltas(PRE, f, a, "greedy", k, w, ts))[0] for a in C.POOL_ARMS for f in F for k in (3, 5) for ts in C.TESTSETS]
    print("greedy mit %s gegen Best Fit, alle 60 Zellen (2 Pools, 3 Familien, k 3/5, 5 Testmengen):" % w, {x: labs.count(x) for x in ev.LABELS})

print("\n== T7 Training gegen Test der Mengen (greedy, Orakel, gegen Best Fit; Verhältnis Test/Training) ==")
ratios = []
for arm in C.POOL_ARMS:
    for fam in F:
        for k in (3, 5):
            tr, te = med(ev.set_train_deltas(PRE, fam, arm, "greedy", k)), med(ev.set_deltas(PRE, fam, arm, "greedy", k, "orakel"))
            ratios.append(te / tr)
            print("  %-5s %-8s k=%d Training %s Test %s (%.0f %%)" % (arm, fam, k, f2(tr), f2(te), 100 * te / tr))
print("Spanne des Verhältnisses: %.0f %% bis %.0f %%" % (100 * min(ratios), 100 * max(ratios)))

print("\n== T8 100 Trainingsinstanzen (10 Läufe; Median, Läufe < 0) ==")
for fam in C.BIG_FAMILIES:
    for arm in C.BIG_ARMS:
        runs = PRE["m3"][fam][arm]
        tr, te = [r["train"] for r in runs], [r["test"] for r in runs]
        r20 = PRE["m1"][fam][arm]
        print("  %-8s %-5s Regel: Training %s (mit 20: %s) Test %s (%d<0, %d>0; mit 20: %s)" % (
            fam, arm, f2(med(tr)), f2(med([r["train"] for r in r20])), f2(med(te)), sum(x < 0 for x in te), sum(x > 0 for x in te), f2(med([r["test"] for r in r20]))))
        for k in (3, 5):
            print("       k=%d" % k, " | ".join("%s: %s" % (typ, " ".join("%s %s (%d<0, %d>0)" % (w[:2], f2(med(d)), sum(x < 0 for x in d), sum(x > 0 for x in d))
                                                               for w in C.CHOICES for d in [ev.set_deltas(PRE, fam, arm, typ, k, w, block="m3")]))
                                            for typ in ("greedy", "topk", "zufall")))

print("\n== T9 Portfolio Best Fit, First Fit, Worst Fit (gegen Best Fit) ==")
for fam in F:
    print("  %-8s" % fam, " | ".join("%s O %s M %s F %s" % (ts, f2(med(ev.portfolio_deltas(PRE, fam, "orakel", ts))), f2(med(ev.portfolio_deltas(PRE, fam, "merkmal", ts))),
                                                         f2(med(ev.portfolio_deltas(PRE, fam, "fest", ts)))) for ts in C.TESTSETS))

print("\n== T10 Kinder der Variation (20 Trainingssätze x 300 Kinder je Zelle) ==")
for fam in F:
    for p in C.PARENTS:
        z = ev.variation_cell(PRE, fam, p)
        print("  %-8s %-7s Kinder %d: besser %d (%.1f %%), gleich %d (%.1f %%), gleichwertig %d, schlechter %d (%.1f %%); auch Test besser %d von %d; besser als BF %d (%.1f %%); Eltern schlechter als BF in %d von %d Sätzen" % (
            fam, p, z["n"], z["besser"], 100 * z["besser"] / z["n"], z["gleich"], 100 * z["gleich"] / z["n"], z["gleichwertig"], z["schlechter"], 100 * z["schlechter"] / z["n"],
            z["better_test"], z["besser"], z["better_bf"], 100 * z["better_bf"] / z["n"], z["parent_worse"], z["saetze"]))
        print("       nach Art:", " | ".join("%s n %d besser %d (%.1f %%) gleich %d" % (k, z["art"][k]["n"], z["art"][k]["besser"], 100 * z["art"][k]["besser"] / max(1, z["art"][k]["n"]), z["art"][k]["gleich"]) for k in V.KINDS))

print("\n== T11 Spielraum (Überschuss über LB in %, 200 bzw. 100 Instanzen) ==")
for fam in F:
    for ts in C.TESTSETS:
        sp = PRE["spielraum"][fam][ts]
        print("  %-8s %-7s BF %.2f FF %.2f WF %.2f NF %.2f FFD %.2f | FF<BF %d, FF=BF %d, FF>BF %d von %d | LB im Mittel %.2f" % (
            fam, ts, sp["BF"], sp["FF"], sp["WF"], sp["NF"], sp["FFD"], sp["ff_lt_bf"], sp["ff_eq_bf"], sp["ff_gt_bf"], sp["n"], sp["lb_mean"]))
print("10 Gegenstände gegen das Optimum:", {f: PRE["m6"][f] for f in F})

print("\n== T12 Stichprobe 10 000 Auswertungen (EA-S, gl_20_70, 2 Läufe) ==")
for r in PRE["m7"]["gl_20_70"]["EA-S"]:
    print("  Lauf %d:" % r["lauf"], [(c["n"], f2(c["train"]), f2(c["test"])) for c in r["cps"]], "Pool", r["pool"],
          "greedy k=3 Orakel/Merkmal/fest", [f2(x) for x in r["sets"]["greedy"]["3"]["T"]["same60"]], "Training", f2(r["sets"]["greedy"]["3"]["train"]))

n_runs = sum(len(PRE["m1"][f][a]) for f in F for a in C.ARMS)
n_eval = sum(r["stat"]["n"] for f in F for a in C.ARMS for r in PRE["m1"][f][a]) + sum(r["stat"]["n"] for f in C.BIG_FAMILIES for a in C.BIG_ARMS for r in PRE["m3"][f][a]) \
    + sum(r["n"] for f in F for p in C.PARENTS for r in PRE["m4"][f][p]) + sum(r["stat"]["n"] for r in PRE["m7"]["gl_20_70"]["EA-S"])
print("\nGESAMT: %d Läufe in M1, %d gezählte Auswertungen in allen Blöcken; Vergleiche: %d" % (n_runs, n_eval, len(AR.COMPARISONS)))
