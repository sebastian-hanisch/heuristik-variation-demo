"""Presets und Permalink: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Einrasten der Adresszeilenwerte."""

import hva_arms as AR
import hva_constants as C
import hva_presets as P
import hva_sets as M
import hva_variation as V


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS) == set(P.SETTING_SPECS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_bounds_and_on_the_steps():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[key]
            if spec.options is not None:
                assert value in spec.options, (name, key)
            else:
                assert spec.lo <= value <= spec.hi and (value - spec.lo) % spec.step == 0, (name, key)


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS))
    assert first == "Eine Menge von Hand"
    assert C.PRESETS[first] == {key: spec.default for key, spec in P.SETTING_SPECS.items()} == C.DEFAULTS


def test_presets_sind_paarweise_verschieden():
    """Zwei gleiche Presets könnten nicht beide hervorgehoben werden (die Markierung ist zustandslos)."""
    seen = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(seen)) == len(seen) == 7


def test_kein_preset_liegt_nur_einen_regler_von_der_voreinstellung_am_reglerrand_entfernt():
    """Sonst würde das Verschieben eines Reglers an seinen Rand ein anderes Preset markieren (die Markierung ist zustandslos)."""
    for name, preset in list(C.PRESETS.items())[1:]:
        diff = [k for k in preset if preset[k] != C.DEFAULTS[k]]
        assert diff, name
        if len(diff) == 1:
            spec = P.SETTING_SPECS[diff[0]]
            assert preset[diff[0]] not in (spec.lo, spec.hi), (name, diff)


def test_presets_wie_beschrieben():
    p = C.PRESETS
    assert (p["Eine Menge von Hand"]["hand_schritt"], p["Eine Menge von Hand"]["budget"]) == (3, 400)
    assert (p["Variation an Best Fit"]["elternteil"], p["Variation an Best Fit"]["kinder"], p["Variation an Best Fit"]["familie"], p["Variation an Best Fit"]["hand_schritt"], p["Variation an Best Fit"]["budget"]) == ("bf", 300, "gl_20_70", 0, 200)
    assert (p["Variation und Auswahl"]["arm"], p["Variation und Auswahl"]["budget"]) == ("EA-S", 600)
    assert (p["Bäume statt Vorlagen"]["arm"], p["Bäume statt Vorlagen"]["budget"]) == ("GP-T", 600)
    assert (p["Menge mit Orakelwahl"]["arm"], p["Menge mit Orakelwahl"]["menge"], p["Menge mit Orakelwahl"]["k"], p["Menge mit Orakelwahl"]["wahl"], p["Menge mit Orakelwahl"]["budget"], p["Menge mit Orakelwahl"]["lauf_seed"]) == ("EA-S", "greedy", 3, "orakel", 1000, 1)
    assert (p["Menge ohne Orakel"]["wahl"], p["Menge ohne Orakel"]["k"], p["Menge ohne Orakel"]["menge"], p["Menge ohne Orakel"]["lauf_seed"]) == ("merkmal", 3, "greedy", 1)
    assert (p["Häufungen: wenig Vielfalt"]["familie"], p["Häufungen: wenig Vielfalt"]["k"], p["Häufungen: wenig Vielfalt"]["wahl"]) == ("clu3", 3, "orakel")
    assert "400 Auswertungen" in C.PRESET_HELP["Eine Menge von Hand"] and "300 Kinder" in C.PRESET_HELP["Variation an Best Fit"] and "200 Auswertungen" in C.PRESET_HELP["Variation an Best Fit"]
    assert "600 Auswertungen" in C.PRESET_HELP["Variation und Auswahl"] and "600 Auswertungen" in C.PRESET_HELP["Bäume statt Vorlagen"]
    assert "1 000 Auswertungen" in C.PRESET_HELP["Menge mit Orakelwahl"] and "ersten zehn Gegenständen" in C.PRESET_HELP["Menge ohne Orakel"]


def test_konstanten_passen_zum_kern():
    assert C.ARMS == AR.ARM_ORDER and set(C.ARM_LABELS) == set(AR.ARMS) and all(f"({a})" in C.ARM_LABELS[a] for a in C.ARMS)
    assert C.CHOICES == M.CHOICES and set(C.CHOICE_LABELS) == set(C.CHOICES) == set(C.CHOICE_SHORT)
    assert set(C.PARENT_LABELS) == set(C.PARENTS) and set(C.SET_LABELS) == set(C.SET_TYPES) and set(C.SET_TYPES) < set(C.SET_TYPES_FROZEN)
    assert C.CHECKPOINTS == tuple(range(100, 1001, 100)) and set(range(C.BUDGET_MIN, C.BUDGET_MAX + 1, C.BUDGET_STEP)) == set(C.CHECKPOINTS)
    assert C.FROZEN_BUDGET == C.BUDGET_MAX == 1000 and C.THREE_ARM_BUDGET_MAX == 400 and C.DEFAULT_BUDGET == 400
    assert (C.N_ITEMS, C.K_TRAIN, C.K_TRAIN_BIG, C.K_TEST, C.CAPACITY, C.N_RUNS_FROZEN, C.N_RUNS_BIG) == (60, 20, 100, 200, 100, 20, 10)
    assert C.TESTSET_SPEC == {"same60": (60, 200), "n30": (30, 200), "n120": (120, 100)} and M.N_FIRST == 10 and C.N_RANDOM_SETS == 5
    assert set(C.POOL_ARMS) <= set(C.BIG_ARMS) <= set(C.ARMS) and set(C.BIG_FAMILIES) < set(C.FAMILIES)
    assert len(AR.COMPARISONS) == 7 and [c[2] for c in AR.COMPARISONS].count("Variation allein") == 2
    assert V.KINDS == tuple(V.KIND_LABELS) and C.FROZEN_CHILDREN == C.DEFAULT_KINDER == 300


def test_clamp_begrenzt_und_rastet_ein():
    s = P.SETTING_SPECS
    assert P.clamp(s["budget"], 5) == 100 and P.clamp(s["budget"], 99999) == 1000 and P.clamp(s["budget"], 349) == 300 and P.clamp(s["budget"], 351) == 400
    assert P.clamp(s["kinder"], 1) == 100 and P.clamp(s["kinder"], 260) == 300 and P.clamp(s["kinder"], 5000) == 600
    assert P.clamp(s["k"], 2) in (1, 3) and P.clamp(s["k"], 4) in (3, 5) and P.clamp(s["k"], 99) == 5 and P.clamp(s["k"], 0) == 1
    assert P.clamp(s["familie"], "clu3") == "clu3" and P.clamp(s["familie"], "unbekannt") == "gl_20_70"
    assert P.clamp(s["arm"], "11-W") == "EA-S" and P.clamp(s["arm"], "GP0-T") == "GP0-T" and P.clamp(s["wahl"], "x") == "orakel"
    assert P.clamp(s["lauf_seed"], -3) == 0 and P.clamp(s["lauf_seed"], 5000) == 999 and P.clamp(s["hand_schritt"], 99) == 3
    assert P.bounds("budget") == (100, 1000) and P.options("familie") == C.FAMILIES and P.options("budget") is None
