"""Setzt die README aus `messungen/readme_vorlage.md` zusammen: Marken der Form {{schlüssel}} werden durch die Befundzeilen aus
`messungen/befunde.py` ersetzt, {{GRENZEN}} durch den Abschnitt der App, {{TESTZAHL}} durch die Zahl der Tests (Argument).

Aufruf (im Demo-Ordner): python messungen/readme_bauen.py <Testzahl>
Die README wird nie von Hand mit Zahlen gefüllt; `tests/test_claims.py` rechnet jede Befundzeile neu und sucht sie wörtlich in der README."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "messungen"))

import befunde  # noqa: E402
import hva_evaluation as ev  # noqa: E402
import hva_texts as T  # noqa: E402


def build(testzahl):
    text = (ROOT / "messungen" / "readme_vorlage.md").read_text(encoding="utf-8")
    werte = dict(befunde.lines(ev.load_precomputed()))
    werte["GRENZEN"] = T.GRENZEN.strip()
    werte["TESTZAHL"] = str(testzahl)
    fehlend = sorted(set(re.findall(r"\{\{([^}]+)\}\}", text)) - set(werte))
    if fehlend:
        raise SystemExit(f"Marken ohne Wert: {fehlend}")
    unbenutzt = sorted(k for k in werte if "{{" + k + "}}" not in text)
    for key, val in werte.items():
        text = text.replace("{{" + key + "}}", val)
    return text, unbenutzt


if __name__ == "__main__":
    out, unbenutzt = build(sys.argv[1] if len(sys.argv) > 1 else "0")
    (ROOT / "README.md").write_text(out, encoding="utf-8", newline="\n")
    print("README.md geschrieben (%d Zeichen); Befundzeilen ohne Marke in der Vorlage: %s" % (len(out), unbenutzt))
