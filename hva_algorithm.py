"""Kern: Online-Packen mit einer Platzierungsfunktion (nur numpy und Standardbibliothek).

Eine Platzierungsfunktion f(s, r, g, p, e) bewertet für den nächsten Gegenstand jeden offenen Behälter, in den er passt, und einen leeren
Behälter; der Behälter mit dem höchsten Wert bekommt den Gegenstand (Gleichstand: kleinster Index). Jede Regel ist eine eigene Funktion mit
expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b): `valid_mask`, `features`, `choose_bin` sind die Einheiten eines Schritts, `simulate` ruft sie
nur der Reihe nach auf, für alle Instanzen eines Satzes gleichzeitig.

Merkmale (alle auf die Kapazität normiert):
    s = Größe des Gegenstands, r = Restkapazität des Behälters, g = r - s (Lücke nach dem Einsetzen),
    p = Index / (Zahl offener Behälter + 1), e = 1 nur für den leeren Behälter.
"""

import numpy as np

CAPACITY = 100          # Kapazität jedes Behälters
CLIP = 1e18             # NaN und unendliche Bewertungen werden auf -CLIP bzw. +-CLIP gesetzt (Festlegung der Messung)
OPT_DP_MAX_N = 14       # Teilmengen-DP: 2^n Zustände mal n Übergänge
SNAP = 1e12             # Abstände werden auf 12 Nachkommastellen gerundet (siehe `snap`)


def snap(x):
    """Rundet auf 12 Nachkommastellen (Vielfache von 1e-12, Hälften zur geraden Zahl).

    Abstände wie |g - a| sind in Gleitkommazahlen nicht exakt: |0,30 - 0,20| und |0,10 - 0,20| unterscheiden sich in der letzten Stelle,
    obwohl beide Lücken gleich weit vom Ziel liegen. Ohne Rundung entschiede dieser Rest den Gleichstand, mit ihr gewinnt wie festgelegt
    der kleinste Index. Nur Multiplikation, Division und Runden: auf jeder Plattform dieselben Bits."""
    return np.rint(np.asarray(x, dtype=np.float64) * SNAP) / SNAP


# ---------------------------------------------------------------------------------------------------------------------------------------
# Schranken und Kennzahlen
# ---------------------------------------------------------------------------------------------------------------------------------------
def lower_bound(sizes, capacity=CAPACITY):
    """Untere Schranke je Instanz: Summe der Größen durch die Kapazität, aufgerundet. `sizes` ist (K, n) oder eine einzelne Folge."""
    arr = np.atleast_2d(np.asarray(sizes, dtype=np.int64))
    return -(-arr.sum(axis=1) // capacity)


def excess_pct(bins, lb):
    """Überschuss in Prozent: Mittel über die Instanzen von Behälterzahl / untere Schranke, minus 1, mal 100."""
    return 100.0 * (float(np.mean(np.asarray(bins) / np.asarray(lb))) - 1.0)


def fitness(bins, lb):
    """Trainingsfitness: Mittel von Behälterzahl / untere Schranke (kleiner ist besser)."""
    return float(np.mean(np.asarray(bins) / np.asarray(lb)))


# ---------------------------------------------------------------------------------------------------------------------------------------
# Ein Schritt des Simulators in drei Einheiten
# ---------------------------------------------------------------------------------------------------------------------------------------
def valid_mask(rem_c, nopen, s):
    """Zulässige Spalten: offene Behälter mit genug Rest und der eine leere Behälter direkt dahinter (immer zulässig).

    rem_c (K, M) Restkapazitäten der ersten M Spalten, nopen (K,) Zahl offener Behälter, s (K,) Größe des Gegenstands."""
    idx = np.arange(rem_c.shape[1], dtype=np.int64)[None, :]
    npn = nopen[:, None]
    return ((idx < npn) & (rem_c >= s[:, None])) | (idx == npn)


def features(rem_c, nopen, s, capacity=CAPACITY):
    """Merkmale (s, r, g, p, e) je Instanz und Spalte, normiert auf die Kapazität."""
    idx = np.arange(rem_c.shape[1], dtype=np.int64)[None, :]
    npn = nopen[:, None]
    sf = (s / capacity)[:, None]
    r = rem_c / capacity
    g = (rem_c - s[:, None]) / capacity
    p = idx / (npn + 1.0)
    e = (idx == npn).astype(np.float64)
    return sf, r, g, p, e


def choose_bin(scores, valid):
    """Spalte mit der höchsten Bewertung unter den zulässigen; Gleichstand: kleinster Index.

    NaN wird zu -CLIP, +-unendlich zu +-CLIP; unzulässige Spalten bekommen -unendlich und verlieren damit gegen jede zulässige."""
    sc = np.asarray(scores, dtype=np.float64)
    if sc.shape != valid.shape:
        sc = np.broadcast_to(sc, valid.shape)
    sc = np.nan_to_num(sc, nan=-CLIP, posinf=CLIP, neginf=-CLIP)
    sc = np.clip(sc, -CLIP, CLIP)
    return np.where(valid, sc, -np.inf).argmax(axis=1)


def simulate(sizes, score_fn, want_assign=False, capacity=CAPACITY):
    """Packt alle K Instanzen gleichzeitig. sizes (K, n) ganzzahlig, score_fn(s, r, g, p, e) -> Bewertung (K, M).

    Rückgabe: (Behälterzahlen (K,), Zuordnung (K, n) oder None)."""
    sizes = np.asarray(sizes, dtype=np.int64)
    K, n = sizes.shape
    rem = np.full((K, n + 1), capacity, dtype=np.int64)
    nopen = np.zeros(K, dtype=np.int64)
    rows = np.arange(K)
    assign = np.empty((K, n), dtype=np.int64) if want_assign else None
    if K == 0:
        return nopen, assign
    for t in range(n):
        s = sizes[:, t]
        m = int(nopen.max()) + 1
        rem_c = rem[:, :m]
        valid = valid_mask(rem_c, nopen, s)
        with np.errstate(all="ignore"):
            scores = score_fn(*features(rem_c, nopen, s, capacity))
            ch = choose_bin(scores, valid)
        rem[rows, ch] -= s
        nopen += ch == nopen
        if want_assign:
            assign[:, t] = ch
    return nopen, assign


def pack_trace(items, score_fn, capacity=CAPACITY):
    """Eine Instanz Schritt für Schritt, mit der Bewertungstabelle je Schritt (für die Anzeige des Handbeispiels).

    Rückgabe: dict mit `steps` (je Schritt: Größe, Zeilen (Behälter, Rest, Lücke, Bewertung, zulässig, leer), gewählter Behälter),
    `assign`, `bins` und `contents` (Liste der Gegenstände je Behälter)."""
    items = [int(x) for x in items]
    rem = []
    steps = []
    assign = []
    contents = []
    for s in items:
        nopen = len(rem)
        rem_c = np.array([rem + [capacity]], dtype=np.int64)
        nop = np.array([nopen], dtype=np.int64)
        sv = np.array([s], dtype=np.int64)
        valid = valid_mask(rem_c, nop, sv)
        with np.errstate(all="ignore"):
            sc = np.asarray(score_fn(*features(rem_c, nop, sv, capacity)), dtype=np.float64)
            sc = np.broadcast_to(sc, valid.shape)
            ch = int(choose_bin(sc, valid)[0])
        rows = [
            dict(bin=j, rest=int(rem_c[0, j]), gap=int(rem_c[0, j]) - s, score=float(sc[0, j]), valid=bool(valid[0, j]), empty=j == nopen)
            for j in range(nopen + 1)
        ]
        steps.append(dict(size=s, rows=rows, chosen=ch))
        if ch == nopen:
            rem.append(capacity)
            contents.append([])
        rem[ch] -= s
        contents[ch].append(s)
        assign.append(ch)
    return dict(steps=steps, assign=assign, bins=len(rem), contents=contents)


# ---------------------------------------------------------------------------------------------------------------------------------------
# Referenzen außerhalb der Klasse der Platzierungsfunktionen
# ---------------------------------------------------------------------------------------------------------------------------------------
def next_fit_bins(sizes, capacity=CAPACITY):
    """Next Fit für alle Instanzen: nur der zuletzt geöffnete Behälter ist offen; passt der Gegenstand nicht, wird ein neuer geöffnet."""
    sizes = np.asarray(sizes, dtype=np.int64)
    K, n = sizes.shape
    load = np.full(K, capacity + 1, dtype=np.int64)   # erzwingt das Öffnen beim ersten Gegenstand
    nb = np.zeros(K, dtype=np.int64)
    for t in range(n):
        s = sizes[:, t]
        new = load + s > capacity
        nb += new
        load = np.where(new, s, load + s)
    return nb


def next_fit_contents(items, capacity=CAPACITY):
    """Next Fit für eine Instanz: Liste der Gegenstände je Behälter."""
    contents = []
    load = capacity + 1
    for s in (int(x) for x in items):
        if load + s > capacity:
            contents.append([])
            load = 0
        contents[-1].append(s)
        load += s
    return contents


def ffd_bins(sizes, capacity=CAPACITY):
    """First Fit Decreasing (offline): je Instanz absteigend sortieren, dann First Fit. Online nicht erreichbar, nur Referenz."""
    sizes = np.asarray(sizes, dtype=np.int64)
    return simulate(-np.sort(-sizes, axis=1), score_ff, capacity=capacity)[0]


def opt_dp(items, capacity=CAPACITY):
    """Exaktes Optimum einer kleinen Instanz per Teilmengen-DP: dp[Menge] = (Behälterzahl, Füllung des letzten), lexikographisch minimal."""
    items = [int(x) for x in items]
    n = len(items)
    if n > OPT_DP_MAX_N:
        raise ValueError(f"opt_dp ist auf n <= {OPT_DP_MAX_N} begrenzt (n = {n})")
    if n == 0:
        return 0
    if max(items) > capacity or min(items) < 1:
        raise ValueError("Größen müssen in 1..Kapazität liegen")
    inf = (10 ** 9, 0)
    dp = [inf] * (1 << n)
    dp[0] = (1, 0)
    for mask in range(1 << n):
        b, load = dp[mask]
        if b >= 10 ** 9:
            continue
        for i in range(n):
            if mask >> i & 1:
                continue
            w = items[i]
            cand = (b, load + w) if load + w <= capacity else (b + 1, w)
            nm = mask | (1 << i)
            if cand < dp[nm]:
                dp[nm] = cand
    return dp[(1 << n) - 1][0]


# ---------------------------------------------------------------------------------------------------------------------------------------
# Bekannte Regeln und parametrische Regeln als Platzierungsfunktionen
# ---------------------------------------------------------------------------------------------------------------------------------------
def score_ff(s, r, g, p, e):
    """First Fit: der vorderste zulässige Behälter (der leere steht hinten)."""
    return -p


def score_bf(s, r, g, p, e):
    """Best Fit: die kleinste Lücke nach dem Einsetzen."""
    return -g


def score_wf(s, r, g, p, e):
    """Worst Fit: der größte Rest unter den offenen Behältern; der leere Behälter nur, wenn nichts passt."""
    return r - 2.0 * e


def score_pr(a):
    """Ein-Parameter-Regel PR(a) = -|g - a|: die Lücke nach dem Einsetzen möglichst nahe an a (a = 0 ist Best Fit); Abstand über `snap`."""
    def fn(s, r, g, p, e):
        return -snap(np.abs(g - a))
    return fn


def score_pr2(lam, m):
    """Zwei-Parameter-Regel PR2(lam, m) = -g - lam * [0 < g < m]: Best Fit mit Strafe für kleine, nicht leere Lücken."""
    def fn(s, r, g, p, e):
        return -g - lam * ((g > 0) & (g < m)).astype(np.float64)
    return fn


KNOWN = {"FF": score_ff, "BF": score_bf, "WF": score_wf}


def run_known(sizes, name, capacity=CAPACITY):
    """Behälterzahlen einer bekannten Regel (FF, BF, WF als Platzierungsfunktion; NF und FFD als Referenz)."""
    if name == "NF":
        return next_fit_bins(sizes, capacity)
    if name == "FFD":
        return ffd_bins(sizes, capacity)
    return simulate(sizes, KNOWN[name], capacity=capacity)[0]
