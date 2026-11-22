"""Plotly-Abbildungen (alle Achsen gesperrt, damit Touch-Scrollen nicht am Diagramm hängen bleibt)."""

import plotly.graph_objects as go

import hva_constants as C


def _lock(fig, height=340):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.25))
    return fig


def _k(x, stellen=2):
    v = round(float(x), stellen)
    if v == 0:
        v = 0.0
    return f"{v:.{stellen}f}".replace(".", ",").replace("-", "−")


def build_hand(names, values, bf):
    """Handbeispiel: Behälter auf den zwei Testinstanzen je Wahlart; die Linie ist Best Fit."""
    colors = [C.COLOR_BF if v == bf else (C.COLOR_LINE if v < bf else C.COLOR_RULE) for v in values]
    fig = go.Figure(go.Bar(x=names, y=values, text=[str(v) for v in values], textposition="outside", marker_color=colors,
                           hovertemplate="%{x}: %{y} Behälter<extra></extra>"))
    fig.add_hline(y=bf, line_dash="dot", line_color="gray", annotation_text=f"Best Fit: {bf}", annotation_position="top left")
    fig.update_layout(title="Behälter auf den zwei Testinstanzen", yaxis_title="Behälter (Summe)")
    fig.update_yaxes(range=[0, max(values) * 1.25])
    return _lock(fig, 320)


def build_children(counts, labels):
    """Experiment 1: Kinder je Klasse gegen den Elternteil (Zählungen)."""
    colors = [C.COLOR_LINE, C.COLOR_GREY, C.COLOR_OTHER, C.COLOR_RULE][: len(counts)]
    fig = go.Figure(go.Bar(x=labels, y=counts, text=[str(c) for c in counts], textposition="outside", marker_color=colors,
                           hovertemplate="%{x}: %{y} Kinder<extra></extra>"))
    fig.update_layout(title="Kinder gegen den Elternteil (Training)", yaxis_title="Kinder")
    fig.update_yaxes(range=[0, max(max(counts), 1) * 1.2])
    return _lock(fig, 320)


def build_children_scatter(points):
    """Experiment 1: die auf dem Training besseren Kinder; x = Trainings-Delta, y = Test-Delta, beide gegen den Elternteil."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    lim_x = max(0.5, max((abs(x) for x in xs), default=0.5) * 1.15)
    lim_y = max(0.5, max((abs(y) for y in ys), default=0.5) * 1.15)
    fig = go.Figure()
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(color=C.COLOR_LINE, size=8, opacity=0.7), name="Kind",
                             hovertemplate="Training %{x:.2f}, Test %{y:.2f} Punkte<extra></extra>"))
    fig.add_annotation(x=-lim_x * 0.5, y=-lim_y * 0.9, text="auch auf dem Test besser", showarrow=False, bgcolor="rgba(255,255,255,0.88)")
    fig.add_annotation(x=-lim_x * 0.5, y=lim_y * 0.9, text="auf dem Test schlechter", showarrow=False, bgcolor="rgba(255,255,255,0.88)")
    fig.update_layout(title="Auf dem Training bessere Kinder: Training gegen Test", xaxis_title="Trainings-Delta gegen den Elternteil (Punkte)",
                      yaxis_title="Test-Delta gegen den Elternteil (Punkte)", showlegend=False)
    fig.update_xaxes(range=[-lim_x, 0.05 * lim_x])
    fig.update_yaxes(range=[-lim_y, lim_y])
    return _lock(fig, 340)


def build_curve(points, budget, tol):
    """Experiment 2: Bestwert über die Auswertungen. `points`: (Auswertung, Trainings-Delta, Test-Delta) an den Verbesserungsstellen."""
    xs = [p[0] for p in points] + [budget]
    tr = [p[1] for p in points] + [points[-1][1]]
    te = [p[2] for p in points] + [points[-1][2]]
    fig = go.Figure()
    fig.add_hrect(y0=-tol, y1=tol, fillcolor="rgba(120,120,120,0.12)", line_width=0)
    fig.add_hline(y=0, line_color="gray", line_width=1, annotation_text="Best Fit", annotation_position="bottom right")
    fig.add_trace(go.Scatter(x=xs, y=tr, mode="lines", line=dict(shape="hv", color=C.COLOR_LINE, width=2), name="Training (Bestwert bisher)"))
    fig.add_trace(go.Scatter(x=xs, y=te, mode="lines+markers", line=dict(shape="hv", color=C.COLOR_RULE, width=2), name="dieselbe Regel auf 200 frischen Instanzen"))
    lim = max(1.0, max(abs(v) for v in tr + te) * 1.15)
    fig.update_layout(title="Bestwert über die Auswertungen (Delta gegen Best Fit, Prozentpunkte)", xaxis_title="Auswertungen", yaxis_title="Delta gegen Best Fit")
    fig.update_xaxes(range=[0, budget * 1.02])
    fig.update_yaxes(range=[-lim, lim])
    return _lock(fig, 360)


def build_choices(labels, vs_bf, vs_rule, runs, tol):
    """Experiment 3: drei Wahlarten einer Menge, Delta gegen Best Fit und gegen die beste Einzelregel desselben Laufs."""
    fig = go.Figure()
    fig.add_hrect(y0=-tol, y1=tol, fillcolor="rgba(120,120,120,0.12)", line_width=0)
    fig.add_hline(y=0, line_color="gray", line_width=1)
    x = [f"{lab}<br>{r} Läufe je Instanz" if r != 1 else f"{lab}<br>1 Lauf je Instanz" for lab, r in zip(labels, runs)]
    fig.add_trace(go.Bar(x=x, y=vs_bf, name="gegen Best Fit", marker_color=C.COLOR_BF, text=[_k(v) for v in vs_bf], textposition="outside"))
    fig.add_trace(go.Bar(x=x, y=vs_rule, name="gegen die beste Einzelregel des Laufs", marker_color=C.COLOR_OTHER, text=[_k(v) for v in vs_rule], textposition="outside"))
    lim = max(1.0, max(abs(v) for v in list(vs_bf) + list(vs_rule)) * 1.3)
    fig.update_layout(title="Wahl im Einsatz: Delta in Prozentpunkten (negativ = besser)", yaxis_title="Delta", barmode="group")
    fig.update_yaxes(range=[-lim, lim])
    return _lock(fig, 360)


def build_set_types(labels, values, highlight, tol):
    """Experiment 4: Orakelwahl über die Mengenarten gegen Best Fit; die gewählte Art ist hervorgehoben."""
    colors = [C.COLOR_RULE if i == highlight else C.COLOR_LINE for i in range(len(labels))]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, text=[_k(v) for v in values], textposition="outside",
                           hovertemplate="%{x}: %{y:.2f} Punkte<extra></extra>"))
    fig.add_hrect(y0=-tol, y1=tol, fillcolor="rgba(120,120,120,0.12)", line_width=0)
    fig.add_hline(y=0, line_color="gray", line_width=1)
    lim = max(1.0, max(abs(v) for v in values) * 1.3)
    fig.update_layout(title="Orakelwahl je Mengenart: Delta gegen Best Fit (Prozentpunkte)", yaxis_title="Delta gegen Best Fit")
    fig.update_yaxes(range=[-lim, lim])
    return _lock(fig, 340)


def build_frozen_cells(labels, medians, verdicts, tol, title):
    """Mediane eingefrorener Zellen als Balken; die Farbe folgt dem Urteil."""
    cmap = {"besser": C.COLOR_LINE, "gleich": C.COLOR_GREY, "schlechter": C.COLOR_RULE, None: C.COLOR_OTHER}
    fig = go.Figure(go.Bar(x=labels, y=medians, marker_color=[cmap[v] for v in verdicts], text=[_k(v) for v in medians], textposition="outside",
                           hovertemplate="%{x}: Median %{y:.2f} Punkte<extra></extra>"))
    fig.add_hrect(y0=-tol, y1=tol, fillcolor="rgba(120,120,120,0.12)", line_width=0)
    fig.add_hline(y=0, line_color="gray", line_width=1)
    lim = max(1.0, max(abs(v) for v in medians) * 1.3)
    fig.update_layout(title=title, yaxis_title="Median Delta gegen Best Fit")
    fig.update_yaxes(range=[-lim, lim])
    return _lock(fig, 340)
