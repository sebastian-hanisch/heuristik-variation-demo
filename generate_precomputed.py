"""Erzeugt die eingefrorenen Messreihen `hva_precomputed.json` (Zufall: SplitMix64, dieselben Funktionen wie die App).

Aufruf (Blöcke einzeln, Rohdateien nach messungen/roh/, danach zusammenführen):

    python generate_precomputed.py m1:gl_20_70 m1:clu3 m1:stark     # 6 Arme x 20 Läufe je Familie, 1 000 Auswertungen, Mengen, 5 Testmengen
    python generate_precomputed.py m3:gl_20_70 m3:stark             # 100 Trainingsinstanzen, 3 Arme x 10 Läufe
    python generate_precomputed.py m4 m6 m7                         # Kinder der Variation; 10 Gegenstände gegen das Optimum; 10 000 Auswertungen
    python generate_precomputed.py merge                            # schreibt hva_precomputed.json
    python generate_precomputed.py --check                          # rechnet je Block einen Ausschnitt neu und vergleicht

Jeder Block gibt Rechenzeit und die Zahl der Auswertungen aus. Die Zeiten sind Planungshilfe, kein Befund.
"""

import json
import sys
import time
from pathlib import Path

import hva_constants as C
import hva_experiments as E
import hva_scenario as S

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "messungen" / "roh"
M7_FAMILY, M7_ARM, M7_RUNS = "gl_20_70", "EA-S", 2
M7_CHECKPOINTS = (1000, 2000, 5000, 10000)
N_SMALL = 10
DIGITS = 6      # Nachkommastellen der gespeicherten Deltas


def _round(obj):
    if isinstance(obj, float):
        return round(obj, DIGITS)
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v) for v in obj]
    return obj


def _write(name, data):
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / f"{name}.json").write_text(json.dumps(_round(data), ensure_ascii=False), encoding="utf-8")


def block_m1(fam, arms=C.ARMS, runs=C.N_RUNS_FROZEN):
    tests = E.test_sets(fam)
    out = {"spielraum": {ts: E.spielraum(sz) for ts, sz in tests.items()}, "runs": {}}
    out["portfolio"] = [E.portfolio_record(fam, lauf, tests) for lauf in range(runs)]
    n_eval = 0
    for arm in arms:
        t0 = time.time()
        out["runs"][arm] = [E.run_record(fam, arm, lauf, tests) for lauf in range(runs)]
        n_eval += sum(r["stat"]["n"] for r in out["runs"][arm])
        print(f"  m1 {fam} {arm}: {runs} Läufe in {time.time() - t0:.0f} s", flush=True)
    _write(f"m1_{fam}", out)
    return n_eval


def block_m3(fam):
    tests = E.test_sets(fam, names=("same60",))
    out = {"runs": {}, "portfolio": [E.portfolio_record(fam, lauf, tests, K=C.K_TRAIN_BIG) for lauf in range(C.N_RUNS_BIG)]}
    n_eval = 0
    for arm in C.BIG_ARMS:
        t0 = time.time()
        out["runs"][arm] = [E.run_record(fam, arm, lauf, tests, K=C.K_TRAIN_BIG) for lauf in range(C.N_RUNS_BIG)]
        n_eval += sum(r["stat"]["n"] for r in out["runs"][arm])
        print(f"  m3 {fam} {arm}: {C.N_RUNS_BIG} Läufe in {time.time() - t0:.0f} s", flush=True)
    _write(f"m3_{fam}", out)
    return n_eval


def block_m4():
    out = {}
    n_eval = 0
    for fam in C.FAMILIES:
        test = S.test_set(fam)
        out[fam] = {}
        for parent in C.PARENTS:
            t0 = time.time()
            out[fam][parent] = [E.variation_record(fam, parent, lauf, C.FROZEN_CHILDREN, test) for lauf in range(C.N_RUNS_FROZEN)]
            n_eval += sum(r["n"] for r in out[fam][parent])
            print(f"  m4 {fam} {parent}: {time.time() - t0:.0f} s", flush=True)
    _write("m4", out)
    return n_eval


def block_m6():
    out = {fam: E.spielraum_opt(S.test_set(fam, n=N_SMALL)) for fam in C.FAMILIES}
    _write("m6", out)
    return 0


def block_m7():
    tests = E.test_sets(M7_FAMILY, names=("same60",))
    recs = [E.run_record(M7_FAMILY, M7_ARM, lauf, tests, budget=M7_CHECKPOINTS[-1], checkpoints=M7_CHECKPOINTS) for lauf in range(M7_RUNS)]
    _write("m7", {M7_FAMILY: {M7_ARM: recs}})
    return sum(r["stat"]["n"] for r in recs)


def merge():
    def raw(name):
        return json.loads((RAW / f"{name}.json").read_text(encoding="utf-8"))

    m1 = {fam: raw(f"m1_{fam}") for fam in C.FAMILIES}
    m3 = {fam: raw(f"m3_{fam}") for fam in C.BIG_FAMILIES}
    data = {
        "meta": {
            "zufall": "SplitMix64 (hva_rng.py)", "erzeugt_mit": "generate_precomputed.py", "kapazitaet": C.CAPACITY, "n_gegenstaende": C.N_ITEMS,
            "k_training": C.K_TRAIN, "k_training_gross": C.K_TRAIN_BIG, "k_test": C.K_TEST, "test_seed": 0, "laeufe": C.N_RUNS_FROZEN,
            "laeufe_gross": C.N_RUNS_BIG, "stufen": list(C.CHECKPOINTS), "testmengen": {k: list(v) for k, v in C.TESTSET_SPEC.items()},
            "wahlarten": list(C.CHOICES), "mengengroessen": list(C.K_OPTIONS), "zufallsmengen_je_lauf": C.N_RANDOM_SETS,
            "kinder": C.FROZEN_CHILDREN, "einheit_delta": "Prozentpunkte Überschuss gegen Best Fit, positiv = schlechter",
        },
        "spielraum": {fam: m1[fam]["spielraum"] for fam in C.FAMILIES},
        "m1": {fam: m1[fam]["runs"] for fam in C.FAMILIES},
        "portfolio": {fam: m1[fam]["portfolio"] for fam in C.FAMILIES},
        "m3": {fam: m3[fam]["runs"] for fam in C.BIG_FAMILIES},
        "portfolio_gross": {fam: m3[fam]["portfolio"] for fam in C.BIG_FAMILIES},
        "m4": raw("m4"),
        "m6": raw("m6"),
        "m7": raw("m7"),
    }
    path = ROOT / "hva_precomputed.json"
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("hva_precomputed.json geschrieben:", path.stat().st_size, "Bytes")


def check():
    """Rechnet je Block einen Ausschnitt neu (ein Lauf) und vergleicht mit der Datei; gibt die Zahl der Prüfungen aus."""
    pre = json.loads((ROOT / "hva_precomputed.json").read_text(encoding="utf-8"))
    fam = "gl_20_70"
    tests = E.test_sets(fam)
    n = 0
    for arm in ("EA-S", "ZW-S"):
        new = _round(E.run_record(fam, arm, 0, tests))
        old = pre["m1"][fam][arm][0]
        assert new["stat"] == old["stat"] and new["pool"] == old["pool"], (arm, "Zählungen")
        assert abs(new["train"] - old["train"]) < 1e-5 and abs(new["test"] - old["test"]) < 1e-5, (arm, "Deltas")
        assert all(abs(a - b) < 1e-5 for a, b in zip(new["sets"]["greedy"]["3"]["T"]["same60"], old["sets"]["greedy"]["3"]["T"]["same60"])), (arm, "Mengen")
        n += 3
    new = _round(E.variation_record(fam, "bf", 0, C.FROZEN_CHILDREN, tests["same60"]))
    assert new == pre["m4"][fam]["bf"][0], "m4"
    n += 1
    print("GESAMT: %d Prüfungen, 0 Fehler" % n)


def main(argv):
    if not argv:
        print(__doc__)
        return
    for job in argv:
        name, _, arg = job.partition(":")
        t0 = time.time()
        if name == "merge":
            merge()
            continue
        if name == "--check":
            check()
            continue
        n_eval = {"m1": lambda: block_m1(arg), "m3": lambda: block_m3(arg), "m4": block_m4, "m6": block_m6, "m7": block_m7}[name]()
        print(f"{job}: fertig in {time.time() - t0:.0f} s, {n_eval} Auswertungen gezählt", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
