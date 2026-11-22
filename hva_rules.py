"""Regelräume: drei Arten, eine Platzierungsfunktion darzustellen, zu ziehen und zu verändern.

W  Gewichtsvektor über einem Baukasten aus acht Bausteinen (Individuum: Tupel aus 8 Gewichten in [-1, 1]).
T  Ausdrucksbaum über den Merkmalen s, r, g, p, e und Konstanten (Individuum: verschachteltes Tupel).
S  Schablonen mit Parametern, optional mit einem zweiten Term (Individuum: (Schablone, Parameter, zweite Schablone oder -1, Parameter, Gewicht)).

Zufall kommt nur über den übergebenen Generator (SplitMix64: `random()`, `below(n)`, `normal()`).
Die Konstanten der Mutationen stammen aus der Vormessung: Streuung 0,2 für Gewichte, 0,3 für Baumkonstanten, 0,1 mal Bereich für Schablonenparameter.
"""

import numpy as np

from hva_algorithm import snap

SPACES = ("W", "T", "S")


def uniform(rng, lo, hi):
    """Gleichverteilt in [lo, hi)."""
    return lo + (hi - lo) * rng.random()


def clip(v, lo, hi):
    return min(hi, max(lo, v))


# ---------------------------------------------------------------------------------------------------------------------------------------
# Raum W: Gewichtsvektor über dem Baukasten
# ---------------------------------------------------------------------------------------------------------------------------------------
BASIS_NAMES = ("r", "p", "e", "[g=0]", "[g<0,1]", "s·r", "−|g−s|", "r²")
W_SIGMA = 0.2


def basis_stack(s, r, g, p, e):
    """Die acht Bausteine des Baukastens in der Reihenfolge von BASIS_NAMES."""
    return [r, p, e, (g == 0).astype(np.float64), (g < 0.1).astype(np.float64), s * r, -snap(np.abs(g - s)), r * r]


def make_w_fn(w):
    """Platzierungsfunktion f = Summe der Gewichte mal Bausteine (Bausteine mit Gewicht 0 werden nicht berechnet)."""
    w = tuple(float(x) for x in w)

    def fn(s, r, g, p, e):
        tot = 0.0
        for wj, phi in zip(w, basis_stack(s, r, g, p, e)):
            if wj != 0.0:
                tot = tot + wj * phi
        return tot + 0.0 * r

    return fn


def w_sample(rng):
    return tuple(round(uniform(rng, -1.0, 1.0), 4) for _ in BASIS_NAMES)


def w_mutate(w, rng):
    """Jedes Gewicht mit Normalstreuung 0,2 stören und auf [-1, 1] begrenzen."""
    return tuple(round(clip(x + W_SIGMA * rng.normal(), -1.0, 1.0), 4) for x in w)


def w_str(w):
    parts = ["%s·%s" % (("%+.2f" % x).replace(".", ","), name) for x, name in zip(w, BASIS_NAMES) if x != 0.0]
    return " ".join(parts) if parts else "0"


# ---------------------------------------------------------------------------------------------------------------------------------------
# Raum T: Ausdrucksbäume
# ---------------------------------------------------------------------------------------------------------------------------------------
TERMS = ("s", "r", "g", "p", "e")
OPS = ("+", "-", "*", "/", "min", "max")
MAX_DEPTH = 6
P_CONST = 0.25          # Anteil der Konstanten unter den Blättern
P_GROW_LEAF = 0.3       # »grow«: Wahrscheinlichkeit, vor der Zieltiefe ein Blatt zu setzen
T_SIGMA = 0.3           # Streuung der Konstantenmutation
P_MUT_CONST = 0.3       # Anteil der Konstantenmutation in `tree_mutate`


def protected_div(a, b):
    """Geschützte Division: ist der Betrag des Nenners höchstens 1e-9, ist das Ergebnis 1."""
    ok = np.abs(b) > 1e-9
    return np.where(ok, a / np.where(ok, b, 1.0), 1.0)


NP_OPS = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": protected_div, "min": np.minimum, "max": np.maximum}


def eval_tree(t, env, ops=NP_OPS):
    """Baum auswerten. Knoten: ("v", Name), ("c", Konstante) oder (Operator, links, rechts)."""
    k = t[0]
    if k == "v":
        return env[t[1]]
    if k == "c":
        return t[1]
    return ops[k](eval_tree(t[1], env, ops), eval_tree(t[2], env, ops))


def make_tree_fn(t):
    def fn(s, r, g, p, e):
        return eval_tree(t, {"s": s, "r": r, "g": g, "p": p, "e": e}) + 0.0 * r

    return fn


def tree_depth(t):
    if t[0] in ("v", "c"):
        return 0
    return 1 + max(tree_depth(t[1]), tree_depth(t[2]))


def tree_size(t):
    if t[0] in ("v", "c"):
        return 1
    return 1 + tree_size(t[1]) + tree_size(t[2])


def tree_str(t):
    if t[0] == "v":
        return t[1]
    if t[0] == "c":
        return "%.3g" % t[1]
    if t[0] in ("min", "max"):
        return "%s(%s, %s)" % (t[0], tree_str(t[1]), tree_str(t[2]))
    return "(%s %s %s)" % (tree_str(t[1]), t[0], tree_str(t[2]))


def rand_leaf(rng):
    if rng.random() < P_CONST:
        return ("c", round(uniform(rng, -1.0, 1.0), 3))
    return ("v", TERMS[rng.below(len(TERMS))])


def rand_tree(rng, depth, grow):
    """Zufallsbaum bis zur Tiefe `depth`; mit `grow` darf vorher ein Blatt stehen, sonst ist der Baum voll."""
    if depth == 0 or (grow and rng.random() < P_GROW_LEAF):
        return rand_leaf(rng)
    op = OPS[rng.below(len(OPS))]
    left = rand_tree(rng, depth - 1, grow)
    right = rand_tree(rng, depth - 1, grow)
    return (op, left, right)


def rand_tree_ramped(rng):
    """Anfangsbaum (ramped half-and-half): Tiefe 2 bis 4, je zur Hälfte voll oder »grow«; die Wurzel ist immer ein Operator."""
    d = 2 + rng.below(3)
    grow = rng.random() < 0.5
    op = OPS[rng.below(len(OPS))]
    left = rand_tree(rng, d - 1, grow)
    right = rand_tree(rng, d - 1, grow)
    return (op, left, right)


def tree_paths(t, path=()):
    out = [path]
    if t[0] not in ("v", "c"):
        out += tree_paths(t[1], path + (1,))
        out += tree_paths(t[2], path + (2,))
    return out


def tree_get(t, path):
    for i in path:
        t = t[i]
    return t


def tree_put(t, path, new):
    if not path:
        return new
    if path[0] == 1:
        return (t[0], tree_put(t[1], path[1:], new), t[2])
    return (t[0], t[1], tree_put(t[2], path[1:], new))


def tree_mutate_subtree(t, rng):
    """Teilbaum-Mutation: ein zufälliger Knoten wird durch einen neuen Baum der Tiefe 0 bis 2 ersetzt; über Tiefe 6 bleibt der Elter."""
    paths = tree_paths(t)
    pth = paths[rng.below(len(paths))]
    new = rand_tree(rng, rng.below(3), True)
    child = tree_put(t, pth, new)
    return child if tree_depth(child) <= MAX_DEPTH else t


def tree_mutate_const(t, rng):
    """Eine Konstante mit Normalstreuung 0,3 stören; ohne Konstante im Baum: None."""
    paths = [p for p in tree_paths(t) if tree_get(t, p)[0] == "c"]
    if not paths:
        return None
    pth = paths[rng.below(len(paths))]
    old = tree_get(t, pth)[1]
    return tree_put(t, pth, ("c", round(clip(old + T_SIGMA * rng.normal(), -1.0, 1.0), 3)))


def tree_mutate(t, rng):
    """Mutation der (1+1)-Suche: mit 0,3 eine Konstante stören (falls es eine gibt), sonst Teilbaum-Mutation."""
    if rng.random() < P_MUT_CONST:
        c = tree_mutate_const(t, rng)
        if c is not None:
            return c
    return tree_mutate_subtree(t, rng)


def tree_crossover(a, b, rng):
    """Teilbaum-Kreuzung: ein zufälliger Knoten von a wird durch einen zufälligen Teilbaum von b ersetzt; über Tiefe 6 bleibt a."""
    pa = tree_paths(a)
    pb = tree_paths(b)
    x = pa[rng.below(len(pa))]
    y = pb[rng.below(len(pb))]
    child = tree_put(a, x, tree_get(b, y))
    return child if tree_depth(child) <= MAX_DEPTH else a


# ---------------------------------------------------------------------------------------------------------------------------------------
# Raum S: Schablonen
# ---------------------------------------------------------------------------------------------------------------------------------------
# Jede Schablone: (Kurzname, Beschreibung, Formel als Text, Bereiche der Parameter)
TEMPLATES = (
    ("S1", "Lückenziel", "−|g − a|", ((0.0, 0.4),)),
    ("S2", "Kantenanpassung", "−min(|g − a|, |g − b|)", ((0.0, 0.15), (0.1, 0.6))),
    ("S3", "Lückenstrafe", "−g − λ·[0 < g < m]", ((0.0, 1.0), (0.02, 0.4))),
    ("S4", "Positionsmischung", "−g − w·p", ((0.0, 3.0),)),
    ("S5", "Lücke proportional zur Größe", "−|g − a·s|", ((0.0, 2.0),)),
)
S_SIGMA_REL = 0.1       # Streuung der Parametermutation, relativ zur Breite des Bereichs
S_MU_SIGMA = 0.2        # Streuung der Mutation des Gewichts des zweiten Terms
P_S_PARAM, P_S_SWAP = 0.5, 0.2      # Parameter stören / Schablone wechseln; der Rest (0,3): zweiten Term anfügen, ändern, entfernen


def template_value(ti, th, s, r, g, p, e):
    """Wert der Schablone ti mit den Parametern th; Abstände über `snap` (gleiche Abstände bleiben gleich)."""
    if ti == 0:
        return -snap(np.abs(g - th[0]))
    if ti == 1:
        return -np.minimum(snap(np.abs(g - th[0])), snap(np.abs(g - th[1])))
    if ti == 2:
        return -g - th[0] * ((g > 0) & (g < th[1])).astype(np.float64)
    if ti == 3:
        return -g - th[0] * p
    if ti == 4:
        return -snap(np.abs(g - th[0] * s))
    raise ValueError(ti)


def make_s_fn(ind):
    ti, th, t2, th2, mu = ind

    def fn(s, r, g, p, e):
        v = template_value(ti, th, s, r, g, p, e)
        if t2 >= 0:
            v = v + mu * template_value(t2, th2, s, r, g, p, e)
        return v + 0.0 * r

    return fn


def s_sample_term(rng, ti=None):
    if ti is None:
        ti = rng.below(len(TEMPLATES))
    th = tuple(round(uniform(rng, lo, hi), 4) for lo, hi in TEMPLATES[ti][3])
    return ti, th


def s_sample(rng):
    """Eine Schablone mit zufälligen Parametern, ohne zweiten Term."""
    ti, th = s_sample_term(rng)
    return (ti, th, -1, (), 0.0)


def _perturb(th, ti, rng):
    return tuple(round(clip(v + S_SIGMA_REL * (hi - lo) * rng.normal(), lo, hi), 4) for v, (lo, hi) in zip(th, TEMPLATES[ti][3]))


def s_mutate(ind, rng):
    """Variation im Raum S: Parameter stören (0,5), Schablone wechseln (0,2), zweiten Term anfügen, ändern oder entfernen (0,3)."""
    ti, th, t2, th2, mu = ind
    u = rng.random()
    if u < P_S_PARAM:
        if t2 >= 0 and rng.random() < 0.5:
            return (ti, th, t2, _perturb(th2, t2, rng), mu)
        return (ti, _perturb(th, ti, rng), t2, th2, mu)
    if u < P_S_PARAM + P_S_SWAP:
        nti, nth = s_sample_term(rng)
        return (nti, nth, t2, th2, mu)
    if t2 < 0:
        nt2, nth2 = s_sample_term(rng)
        return (ti, th, nt2, nth2, round(uniform(rng, 0.0, 1.0), 3))
    v = rng.random()
    if v < 0.4:
        return (ti, th, -1, (), 0.0)
    if v < 0.7:
        return (ti, th, t2, th2, round(clip(mu + S_MU_SIGMA * rng.normal(), 0.0, 1.0), 3))
    nt2, nth2 = s_sample_term(rng)
    return (ti, th, nt2, nth2, mu)


def _term_str(ti, th):
    return "%s(%s)" % (TEMPLATES[ti][0], "; ".join(("%.4g" % v).replace(".", ",") for v in th))


def s_str(ind):
    ti, th, t2, th2, mu = ind
    out = _term_str(ti, th)
    if t2 >= 0:
        out += " + %s·%s" % (("%.3g" % mu).replace(".", ","), _term_str(t2, th2))
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------
# Gemeinsame Schnittstelle der drei Räume
# ---------------------------------------------------------------------------------------------------------------------------------------
def make_fn(space, ind):
    """Platzierungsfunktion fn(s, r, g, p, e) eines Individuums."""
    if space == "W":
        return make_w_fn(ind)
    if space == "T":
        return make_tree_fn(ind)
    if space == "S":
        return make_s_fn(ind)
    raise ValueError(space)


def sample_ind(space, rng):
    if space == "W":
        return w_sample(rng)
    if space == "T":
        return rand_tree_ramped(rng)
    if space == "S":
        return s_sample(rng)
    raise ValueError(space)


def mutate_ind(space, ind, rng):
    if space == "W":
        return w_mutate(ind, rng)
    if space == "T":
        return tree_mutate(ind, rng)
    if space == "S":
        return s_mutate(ind, rng)
    raise ValueError(space)


def ind_key(space, ind):
    """Schlüssel für den Zwischenspeicher: gleiche Individuen, gleicher Schlüssel."""
    if space == "W":
        return tuple(round(float(x), 6) for x in ind)
    return repr(ind)


def ind_str(space, ind):
    """Lesbarer Text der Regel."""
    if space == "W":
        return w_str(ind)
    if space == "T":
        return tree_str(ind)
    return s_str(ind)


def ind_to_json(space, ind):
    """Individuum als JSON-taugliche Struktur (Listen statt Tupel)."""
    def conv(x):
        return [conv(y) for y in x] if isinstance(x, (tuple, list)) else x

    return conv(ind)


def ind_from_json(space, data):
    """Umkehrung von `ind_to_json`."""
    def conv(x):
        return tuple(conv(y) for y in x) if isinstance(x, list) else x

    return conv(data)
