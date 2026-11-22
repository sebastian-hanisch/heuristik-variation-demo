"""Kopier-Tests: fünf Kern-Dateien sind wörtliche Kopien aus `platzierungsfunktion-demo` (Präfix `plf_` durch `hva_` ersetzt),
`hva_rng.py` ist zugleich die Vorlage aus dem Website-Repo.

Liegt die Quelle im Arbeitsbaum, wird mit ihr verglichen; sonst überspringt sich dieser Teil mit Grund. Die festgehaltenen
SHA-256-Werte (Stand des Baus, Zeilenenden auf LF) gelten immer: eine Korrektur gehört in die Quelle, nicht in die Kopie."""

import hashlib
from pathlib import Path

import pytest

import hva_rng

ROOT = Path(__file__).resolve().parent.parent
QUELLE = ROOT.parent / "platzierungsfunktion-demo"
VORLAGE = ROOT.parent / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py"
SHA256 = {
    "rng": "d69b2b2fd67c1234cc0d79249676a662a2189d02803eea7d2edfb04b7a505a3f",
    "algorithm": "838e0a21848f46927669ca52325cade74326c4dd7a4966601c9b5b6d079348be",
    "scenario": "39f895f578b528c946a81b7f6476928f403016fd37d346c0439c6516e638ff97",
    "rules": "719f97d266dae7fcdaef9759c2e5d9e03439b8c3e642bab09797d9dcad94c8d6",
    "search": "c1b766d7785aba1065392f40c43701673a557fcf84b33b7c230cc1d8ec25a441",
}


def _lf(path):
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


@pytest.mark.parametrize("name", list(SHA256))
def test_kopie_stimmt_mit_der_quelle_ueberein_falls_vorhanden(name):
    quelle = QUELLE / f"plf_{name}.py"
    if not quelle.exists():
        pytest.skip("Quelle platzierungsfunktion-demo liegt hier nicht vor")
    assert _lf(ROOT / f"hva_{name}.py") == _lf(quelle).replace("plf_", "hva_")


@pytest.mark.parametrize("name", list(SHA256))
def test_kopie_hat_den_festgehaltenen_stand(name):
    assert hashlib.sha256(_lf(ROOT / f"hva_{name}.py").encode("utf-8")).hexdigest() == SHA256[name]


def test_rng_stimmt_mit_der_vorlage_ueberein_falls_vorhanden():
    if not VORLAGE.exists():
        pytest.skip("Vorlage tools/templates/rng_splitmix64_exact.py liegt hier nicht vor")
    assert _lf(ROOT / "hva_rng.py") == _lf(VORLAGE)


def test_referenzwerte_der_ersten_ganzzahlen():
    # SplitMix64 mit Zufalls-Seed 0: bekannte erste Ausgaben des Verfahrens
    r = hva_rng.SplitMix64(0)
    assert [r.next_u64() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]


def test_random_below_und_normal_sind_fest():
    r = hva_rng.SplitMix64(60)
    xs = [r.random() for _ in range(1000)]
    assert all(0.0 <= x < 1.0 for x in xs) and 0.45 < sum(xs) / 1000 < 0.55
    assert hva_rng.SplitMix64(7).below(13) == hva_rng.SplitMix64(7).next_u64() % 13
    r = hva_rng.SplitMix64(11)
    zs = [r.normal() for _ in range(2000)]
    m = sum(zs) / len(zs)
    var = sum((z - m) ** 2 for z in zs) / len(zs)
    assert abs(m) < 0.1 and 0.9 < var < 1.1
    r2 = hva_rng.SplitMix64(11)
    assert [r2.normal() for _ in range(5)] == zs[:5]


def test_die_kopien_importieren_nur_einander():
    """Die fünf Dateien bilden eine geschlossene Menge: kein Import von Konstanten, Auswertung oder Streamlit."""
    for name in SHA256:
        text = _lf(ROOT / f"hva_{name}.py")
        for verboten in ("hva_constants", "hva_evaluation", "hva_experiments", "streamlit", "numpy.random"):
            assert verboten not in text, (name, verboten)
