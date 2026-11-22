"""Permalink (Begrenzen und Einrasten), Presets und der Knopf für neue Zufalls-Seeds."""

from dataclasses import dataclass

import streamlit as st

import hva_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: object = None           # bei Auswahlreglern: erste Option
    hi: object = None           # bei Auswahlreglern: letzte Option
    options: tuple = None       # erlaubte Werte (Auswahlfeld oder Auswahlregler), sonst None
    step: int = 1


def _choice(url_param, caster, default, options):
    return SettingSpec(url_param, caster, default, options[0], options[-1], tuple(options))


SETTING_SPECS = {
    "familie": _choice("fam", str, C.DEFAULT_FAMILY, C.FAMILIES),
    "arm": _choice("arm", str, C.DEFAULT_ARM, C.ARMS),
    "budget": SettingSpec("budget", int, C.DEFAULT_BUDGET, C.BUDGET_MIN, C.BUDGET_MAX, None, C.BUDGET_STEP),
    "k": _choice("k", int, C.DEFAULT_K, C.K_OPTIONS),
    "menge": _choice("menge", str, C.DEFAULT_SET, C.SET_TYPES),
    "wahl": _choice("wahl", str, C.DEFAULT_CHOICE, C.CHOICES),
    "testmenge": _choice("test", str, C.DEFAULT_TESTSET, C.TESTSETS),
    "elternteil": _choice("eltern", str, C.DEFAULT_PARENT, C.PARENTS),
    "kinder": SettingSpec("kinder", int, C.DEFAULT_KINDER, C.KINDER_MIN, C.KINDER_MAX, None, C.KINDER_STEP),
    "hand_schritt": SettingSpec("schritt", int, C.DEFAULT_HAND_STEP, C.HAND_STEP_MIN, C.HAND_STEP_MAX),
    "test_seed": SettingSpec("tseed", int, C.DEFAULT_TEST_SEED, C.TEST_SEED_MIN, C.TEST_SEED_MAX),
    "lauf_seed": SettingSpec("seed", int, C.DEFAULT_RUN_SEED, C.RUN_SEED_MIN, C.RUN_SEED_MAX),
}
PRESET_KEYS = {key: key for key in SETTING_SPECS}     # Schlüssel im Preset = Schlüssel im Session State


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def options(state_key):
    return SETTING_SPECS[state_key].options


def clamp(spec, value):
    """Wert aus der Adresszeile auf den Regler abbilden: Auswahl -> gültige Option (Zahlen: die nächstgelegene), Zahl -> begrenzen und
    auf den Schritt einrasten."""
    if spec.options is not None:
        if spec.caster in (int, float):
            return min(spec.options, key=lambda o: abs(o - value))
        return value if value in spec.options else spec.default
    value = max(spec.lo, min(spec.hi, value))
    return spec.lo + round((value - spec.lo) / spec.step) * spec.step


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, begrenzt und eingerastet."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            st.session_state[state_key] = clamp(spec, value)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Einstellungen."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State."""
    for name, preset in C.PRESETS.items():
        if all(st.session_state.get(state_key, spec.default) == preset.get(state_key, spec.default) for state_key, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def next_run_seed():
    """Knopf »Neue Zufalls-Seeds«: erhöht den Zufalls-Seed der Läufe um 1 (am Ende des Bereichs von vorn)."""
    nxt = int(st.session_state.get("lauf_seed", C.DEFAULT_RUN_SEED)) + 1
    st.session_state["lauf_seed"] = nxt if nxt <= C.RUN_SEED_MAX else C.RUN_SEED_MIN
