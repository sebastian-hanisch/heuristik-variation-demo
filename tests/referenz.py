"""Unabhängige Rechenwege für die Orakel-Tests: reine Python-Schleifen, exakte Brüche, Aufzählung, ganzzahliges Programm.

Nichts hiervon wird von der App benutzt. Die bekannten Regeln stehen hier als direkte Regeln (»erster passender«, »kleinster Rest«),
nicht als Bewertungsfunktion; die Regelräume werden skalar ausgewertet, Gegenstand für Gegenstand und Behälter für Behälter.
"""

from fractions import Fraction as Fr

CLIP = 1e18


def snap(x):
    """Auf 12 Nachkommastellen: Python-`round` auf die nächste ganze Zahl (Hälften zur geraden), dann zurückteilen."""
    return round(x * 1e12) / 1e12


# --- bekannte Regeln als direkte Regeln (ganzzahlig, ohne Bewertungsfunktion) ---------------------------------------------------------
def direct_pack(items, rule, C=100):
    rem, assign = [], []
    for s in items:
        ch = -1
        if rule == "FF":
            ch = next((j for j, rj in enumerate(rem) if rj >= s), -1)
        elif rule == "BF":
            fit = [(rj - s, j) for j, rj in enumerate(rem) if rj >= s]
            ch = min(fit)[1] if fit else -1          # kleinste Restlücke, bei Gleichstand kleinster Index
        elif rule == "WF":
            fit = [(-rj, j) for j, rj in enumerate(rem) if rj >= s]
            ch = min(fit)[1] if fit else -1          # größter Rest, bei Gleichstand kleinster Index
        elif rule == "NF":
            ch = len(rem) - 1 if rem and rem[-1] >= s else -1
        else:
            raise ValueError(rule)
        if ch < 0:
            rem.append(C)
            ch = len(rem) - 1
        rem[ch] -= s
        assign.append(ch)
    return len(rem), assign


def direct_ffd(items, C=100):
    return direct_pack(sorted(items, reverse=True), "FF", C)[0]


def direct_pr(items, a, C=100):
    """R(a): der zulässige Behälter, dessen Lücke nach dem Einsetzen am nächsten an a liegt (a in Einheiten der Größen);
    der leere Behälter ist ein Kandidat mit Lücke C - s; Gleichstand: kleinster Index."""
    rem, assign = [], []
    for s in items:
        cands = [(abs(rj - s - a), j) for j, rj in enumerate(rem) if rj >= s]
        cands.append((abs(C - s - a), len(rem)))
        ch = min(cands)[1]
        if ch == len(rem):
            rem.append(C)
        rem[ch] -= s
        assign.append(ch)
    return len(rem), assign


# --- Bewertungsfunktion in exakten Brüchen -------------------------------------------------------------------------------------------
def pack_score_exact(items, f, C=100, trace=None):
    rem, assign = [], []
    for t, s in enumerate(items):
        nopen = len(rem)
        best, ch, rows = None, -1, []
        for j in range(nopen + 1):
            rj = rem[j] if j < nopen else C
            if j < nopen and rj < s:
                continue
            v = f(Fr(s, C), Fr(rj, C), Fr(rj - s, C), Fr(j, nopen + 1), 1 if j == nopen else 0)
            rows.append((j, rj, v))
            if best is None or v > best:
                best, ch = v, j
        if trace is not None:
            trace.append((t, s, rows, ch))
        if ch == nopen:
            rem.append(C)
        rem[ch] -= s
        assign.append(ch)
    return len(rem), assign


# --- skalarer Simulator für beliebige Bewertungsfunktionen (Gleitkomma) --------------------------------------------------------------
def sanitize(v):
    if v != v:
        return -CLIP
    return max(-CLIP, min(CLIP, v))


def scalar_simulate(items, scalar_fn, C=100):
    rem, assign = [], []
    for s in items:
        nopen = len(rem)
        best_v, ch = None, -1
        for j in range(nopen + 1):
            rj = rem[j] if j < nopen else C
            if j < nopen and rj < s:
                continue
            v = sanitize(float(scalar_fn(s / C, rj / C, (rj - s) / C, j / (nopen + 1.0), 1.0 if j == nopen else 0.0)))
            if best_v is None or v > best_v:
                best_v, ch = v, j
        if ch == nopen:
            rem.append(C)
        rem[ch] -= s
        assign.append(ch)
    return len(rem), assign


def w_scalar(w):
    def fn(s, r, g, p, e):
        phis = [r, p, e, 1.0 if g == 0 else 0.0, 1.0 if g < 0.1 else 0.0, s * r, -snap(abs(g - s)), r * r]
        tot = 0.0
        for wj, ph in zip(w, phis):
            if wj != 0.0:
                tot = tot + float(wj) * ph
        return tot
    return fn


def _py_div(a, b):
    return a / b if abs(b) > 1e-9 else 1.0


def _py_min(a, b):
    if a != a or b != b:
        return float("nan")
    return a if a < b else b


def _py_max(a, b):
    if a != a or b != b:
        return float("nan")
    return a if a > b else b


PY_OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b, "/": _py_div, "min": _py_min, "max": _py_max}


def _eval(t, env):
    if t[0] == "v":
        return env[t[1]]
    if t[0] == "c":
        return t[1]
    return PY_OPS[t[0]](_eval(t[1], env), _eval(t[2], env))


def tree_scalar(t):
    def fn(s, r, g, p, e):
        try:
            return float(_eval(t, {"s": s, "r": r, "g": g, "p": p, "e": e}))
        except OverflowError:
            return float("nan")
    return fn


def _template_scalar(ti, th, s, r, g, p, e):
    if ti == 0:
        return -snap(abs(g - th[0]))
    if ti == 1:
        return -min(snap(abs(g - th[0])), snap(abs(g - th[1])))
    if ti == 2:
        return -g - th[0] * (1.0 if (g > 0 and g < th[1]) else 0.0)
    if ti == 3:
        return -g - th[0] * p
    if ti == 4:
        return -snap(abs(g - th[0] * s))
    raise ValueError(ti)


def s_scalar(ind):
    ti, th, t2, th2, mu = ind

    def fn(s, r, g, p, e):
        v = _template_scalar(ti, th, s, r, g, p, e)
        if t2 >= 0:
            v = v + mu * _template_scalar(t2, th2, s, r, g, p, e)
        return v
    return fn


# --- exaktes Optimum auf zwei weiteren Wegen -----------------------------------------------------------------------------------------
def opt_enum(items, C=100):
    """Aufzählung aller Mengenzerlegungen (ohne Abschneiden), n <= 8."""
    n = len(items)
    best = [n]

    def rec(i, loads):
        if i == n:
            best[0] = min(best[0], len(loads))
            return
        for j in range(len(loads)):
            if loads[j] + items[i] <= C:
                loads[j] += items[i]
                rec(i + 1, loads)
                loads[j] -= items[i]
        loads.append(items[i])
        rec(i + 1, loads)
        loads.pop()

    rec(0, [])
    return best[0]


def opt_milp(items, C=100):
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    n = len(items)
    m = direct_ffd(items, C)
    nv = n * m + m
    c = np.zeros(nv)
    c[n * m:] = 1.0
    rows, lo, hi = [], [], []
    for i in range(n):
        row = np.zeros(nv)
        row[i * m:(i + 1) * m] = 1
        rows.append(row)
        lo.append(1)
        hi.append(1)
    for j in range(m):
        row = np.zeros(nv)
        for i in range(n):
            row[i * m + j] = items[i]
        row[n * m + j] = -C
        rows.append(row)
        lo.append(-np.inf)
        hi.append(0)
    res = milp(c, constraints=LinearConstraint(np.array(rows), lo, hi), integrality=np.ones(nv), bounds=Bounds(0, 1))
    return int(round(res.fun))
