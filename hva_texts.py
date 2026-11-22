"""Feste Texte der App. Der Abschnitt »Wo die Annahmen enden« steht wortgleich in der README (ein Test vergleicht beide)."""

TITEL = "🧬 Heuristiken variieren und zu Mengen bündeln: ein Sprachmodell-Muster, ohne Sprachmodell gemessen"

EINLEITUNG = """
Verfahren wie FunSearch, EoH und ReEvo lassen ein Sprachmodell neue Regeln für das Packen schreiben, behalten die besten und wiederholen das.
Diese Demo stellt den Kreislauf **ohne Sprachmodell** nach: ein Vorlagen-Operator verändert Regeln (Parameter stören, Vorlage wechseln, einen
zweiten Term anfügen), und sie misst, ob ein Gewinn aus der **Variation**, aus der **Auswahl der Besten** oder aus einer **Menge von Regeln**
kommt, bei der je Instanz die beste Packung zählt. Die Aufgabe ist das Online-Packen in Behälter (Online-Bin-Packing) mit 60 Gegenständen und
Kapazität 100; der feste Gegner ist **Best Fit**. Gezählt wird in **Auswertungen** (eine Regel einmal auf allen 20 Trainingsinstanzen) und,
getrennt davon, in **Läufen je Instanz** im Einsatz.

Weiter unten: der Expander **🔬 Alle Arme und Mengen, 20 Läufe (eingefroren)**, der Abschnitt **🚧 Wo die Annahmen enden** und der Expander
**📐 Mathematische Formulierung**.
"""

EINFACH = """
Regeln für das Packen muss man nicht von Hand schreiben, man kann sie erzeugen lassen: Eine Regel wird leicht verändert, die Packung wird
geprüft, die besten Regeln bleiben und werden wieder verändert. In der Literatur übernimmt ein Sprachmodell das Verändern. Hier erledigt es
eine einfache Vorlage, damit sichtbar wird, wie viel vom Verändern kommt und wie viel vom Behalten der Besten. Das Ergebnis: Auf den
Instanzen, mit denen gesucht wurde, sehen die gefundenen Regeln besser aus als auf neuen Instanzen. Mehrere Regeln gemeinsam helfen, aber
nur, wenn man für jede Instanz alle Regeln laufen lässt und die beste Packung nimmt (Orakelwahl); das kostet mehrfachen Aufwand. Eine einfache
Wahl nach Merkmalen der ersten Gegenstände war in der Messung schlechter als Best Fit, eine feste Wahl nicht besser.
"""

SO_FUNKTIONIERT = """
1. **Eine Regel ist eine Platzierungsfunktion.** Für jeden ankommenden Gegenstand bewertet sie jeden offenen Behälter, in den er passt, und
   einen leeren Behälter; der Behälter mit dem höchsten Wert bekommt den Gegenstand, bei Gleichstand der mit dem kleinsten Index.
   Best Fit ist die Funktion »kleinste Lücke nach dem Einsetzen«.
2. **Variieren.** Der Vorlagen-Operator verändert eine Regel in einem von drei Schritten: die Parameter eines Terms stören (Wahrscheinlichkeit 0,5),
   die Vorlage wechseln (0,2), einen zweiten Term anfügen, ändern oder entfernen (0,3).
3. **Bewerten.** Jede neue Regel packt die 20 Trainingsinstanzen; das zählt als eine Auswertung. Alle bewerteten Regeln landen im **Pool**.
4. **Auswählen und wiederholen.** Mit Auswahl bleiben die Besten und werden wieder variiert; ohne Auswahl werden die Eltern zufällig gezogen.
5. **Bündeln.** Aus dem Pool lässt sich statt einer Regel eine **Menge** von Regeln nehmen. Im Einsatz laufen dann entweder alle Regeln der
   Menge auf jeder Instanz (Orakelwahl), oder eine Wahlregel bestimmt vorab eine Regel je Instanz.
"""

KREISLAUF_DOT = """
digraph kreislauf {
  rankdir=LR;
  node [shape=box, style="rounded,filled", fillcolor="#eef3f7", fontname="Helvetica", fontsize=11];
  edge [fontname="Helvetica", fontsize=10];
  pool [label="Pool\\nalle bewerteten Regeln"];
  wahl [label="Eltern wählen\\n(die Besten oder zufällig)"];
  var [label="Variieren\\n(Vorlagen-Operator)"];
  bew [label="Bewerten\\n20 Trainingsinstanzen\\n= 1 Auswertung"];
  menge [label="Menge bilden\\nk Regeln aus dem Pool"];
  einsatz [label="Einsatz auf frischen Instanzen\\nOrakel-, Merkmals- oder feste Wahl"];
  pool -> wahl -> var -> bew -> pool;
  pool -> menge -> einsatz;
}
"""

VORLAGEN_TABELLE = """
| Vorlage | Formel der Bewertung | Parameter (Anteile der Kapazität) | Idee |
|---|---|---|---|
| S1 Lückenziel | −abs(g − a) | a von 0 bis 0,4 | die Lücke nach dem Einsetzen soll nahe a liegen; a = 0 ist Best Fit |
| S2 Kantenanpassung | −min(abs(g − a), abs(g − b)) | a von 0 bis 0,15, b von 0,1 bis 0,6 | zwei Lückenziele, das nähere zählt |
| S3 Lückenstrafe | −g − λ·[0 < g < m] | λ von 0 bis 1, m von 0,02 bis 0,4 | Best Fit, aber kleine, nicht leere Lücken kosten λ |
| S4 Positionsmischung | −g − w·p | w von 0 bis 3 | Best Fit mit Vorliebe für früh geöffnete Behälter |
| S5 Lücke proportional zur Größe | −abs(g − a·s) | a von 0 bis 2 | das Lückenziel wächst mit dem Gegenstand |

g ist die Lücke nach dem Einsetzen, s die Größe des Gegenstands, p die relative Position des Behälters. Vier der fünf Vorlagen enthalten Best Fit
als Randfall (S1 mit a = 0, S3 mit λ = 0, S4 mit w = 0, S5 mit a = 0); S2 nicht.
"""

ZUORDNUNG = """
**Was aus der Literatur nachgestellt ist und was nicht.** Die Zuordnung ist ein Analogon; keine Zeile wiederholt eine Messung der genannten Arbeit.

| Muster der Literatur | Baustein dieser Demo | nicht nachgestellt |
|---|---|---|
| FunSearch (Romera-Paredes u. a., Nature 2024): ein Sprachmodell schreibt Programme, ein Auswerter bewertet sie, die guten bleiben in einem Pool | Platzierungsfunktion als Skelett, Auswerter = Simulator, Pool = alle bewerteten Regeln | Sprachmodell, Inseln, die dort sehr viel größere Zahl von Proben |
| EoH (Liu u. a., ICML 2024): »Gedanke« und Code werden gemeinsam weiterentwickelt | Vorlage plus Parameter als Individuum | der Gedanke als Text, die Operatoren auf Prompts |
| ReEvo (Ye u. a., NeurIPS 2024): Evolution plus Reflexion | Auswahl der Besten und Variation | die Reflexion |
| EoH-S (Liu u. a., Vorabdruck 2025): Menge von Heuristiken, Güte = Mittel über Instanzen des Minimums über die Menge | Orakelwahl, gierig gebildete Menge | das Sprachmodell als Erzeuger der Mitglieder |
| Zhang u. a. (PPSN 2024): Vergleich von vier Verfahren; die Basislinien dort benutzen selbst ein Sprachmodell | frisches Ziehen und Variation ohne Auswahl, beide **ohne** Sprachmodell | alle Aussagen über Sprachmodelle |
| Sim, Renau, Hart (EvoApplications 2025): Best Fit als fester Gegner über viele Instanzmengen | Trainings- gegen Testinstanzen, Best Fit als Gegner | die Instanzmengen und Heuristiken jener Arbeit |

Quellenstand: Die Arbeiten wurden für die Planung dieser Reihe im Volltext gelesen (FunSearch über einen Abrufdienst zusammengefasst); beim Bau wurden sie nicht erneut geprüft.
"""

KERNFRAGEN = (
    "Hilft die Variation, oder hilft die Auswahl der Besten?",
    "Gewinnt eine Menge von Regeln, und was kostet das?",
    "Hält das auf anderen Instanzen?",
)

E0_CAPTION = ("Vier Regeln (Best Fit, First Fit, Worst Fit und das Lückenziel S1 mit a = 0,10), vier Trainings- und zwei Testinstanzen mit je acht "
              "Gegenständen, Kapazität 100. Die Tabelle zeigt Behälterzahlen; der Regler »Schritt des Handbeispiels« baut die Menge auf.")
E0_SCHRITTE = (
    "**Schritt 0: die Tabelle.** Best Fit und S1(0,10) haben auf den vier Trainingsinstanzen dieselben Behälterzahlen. Auf dem Training sind sie nicht zu unterscheiden.",
    "**Schritt 1: erste Wahl.** Die gierige Menge beginnt mit der Regel mit der kleinsten Trainingssumme: Best Fit (bei Gleichstand mit S1 der kleinere Index).",
    "**Schritt 2: zweite Wahl.** Aufgenommen wird die Regel, mit der die Summe der Instanz-Minima am kleinsten wird. Das ist Worst Fit, die Regel mit der schlechtesten Trainingssumme.",
    "**Schritt 3: Einsatz.** Auf den zwei Testinstanzen stehen Orakelwahl, Merkmalswahl und feste Wahl nebeneinander.",
)
E0_ZAEHLUNG = ("Zählung in zwei Einheiten, die nie addiert werden: Im Training sind es {pool} Auswertungen (vier Regeln, je einmal auf allen vier "
               "Trainingsinstanzen, also {laeufe} Platzierungsläufe). Im Einsatz braucht die Orakelwahl {orakel} Läufe je Instanz, die Merkmalswahl und die "
               "feste Wahl je 1, die feste Menge Best Fit, First Fit, Worst Fit {portfolio}.")

E1_INTRO = ("Ein Elternteil, viele Kinder: jedes Kind entsteht durch einen Schritt des Vorlagen-Operators und wird einmal auf den 20 Trainingsinstanzen "
            "bewertet (eine Auswertung je Kind). Die auf dem Training besseren Kinder werden zusätzlich auf den 200 Testinstanzen geprüft. "
            "Die Regler »Elternteil« und »Kinder« wirken nur auf dieses Experiment.")
E2_INTRO = ("Sechs Arme mit gleichem Budget an Auswertungen. »Variation allein« ist der Schritt vom frischen Ziehen zur Variation ohne Auswahl, "
            "»Auswahl der Besten« der Schritt von dort zur Variation mit Auswahl. Der Verlauf zeigt einen Lauf des gewählten Arms; die Regel wird an jeder "
            "Stelle, an der der Bestwert sich verbessert, auch auf den 200 frischen Instanzen geprüft.")
E2_ZAEHLEINHEIT = ("**Zähleinheit.** Alle Arme bekommen gleich viele Auswertungen; ein Treffer im Zwischenspeicher zählt mit. Branke, Nguyen, Pickardt und Zhang "
                   "(2016) weisen darauf hin, dass gleiche Auswertungszahlen nur begrenzt vergleichbar sind, wenn die Regeln verschieden teuer sind. "
                   "Hier ist jede Auswertung ein Simulatorlauf gleicher Größe; die Bäume haben aber mehr Knoten als die Schablonen Terme.")
E3_INTRO = ("Die Menge entsteht aus dem Pool des Laufs von Experiment 2. Drei Wahlarten im Einsatz: **Orakelwahl** (alle k Regeln laufen auf jeder "
            "Instanz, die beste Packung zählt), **Merkmalswahl** (nach Mittel und Streuung der ersten zehn Gegenstände nimmt jede Instanz die Regel, die "
            "auf der ähnlichsten Trainingsinstanz am besten war; ein Lauf je Instanz) und **feste Wahl** (die Regel der Menge mit der kleinsten "
            "Trainingssumme; ein Lauf je Instanz).")
E4_INTRO = ("Vier Mengen nebeneinander, alle mit Orakelwahl: gierig nach Vielfalt gewählt, die k nach Trainingsfitness besten, k beliebige Regeln des Pools "
            "(Mittel über fünf Zufallsmengen) und die feste Menge Best Fit, First Fit, Worst Fit. Der Regler »Menge« hebt eine Zeile hervor.")
E5_INTRO = ("Nur eingefrorene Läufe, kein eigener Lauf: die gierige Menge auf Instanzen anderer Größe und anderer Familien sowie mit 100 statt 20 "
            "Trainingsinstanzen. Was andere Instanzen mit **einzelnen** erzeugten Regeln machen, ist nicht Gegenstand dieser Demo.")

ARME_KOPF = "| Arm | Raum | So läuft er |\n|---|---|---|\n"

GRENZEN = """
* **Kein Sprachmodell.** Der Operator ist eine Vorlage mit Parametern. Über die Wirkung eines Sprachmodells als Variationsoperator sagt die
  Messung nichts; »Gedanke und Code«, Reflexion und die Frage, ob ein Modell bekannte Heuristiken aus seinen Trainingsdaten wiedergibt, sind
  nicht gemessen. Die Arme ohne Auswahl sind ein Analogon ohne Sprachmodell, nicht die Basislinien von Zhang u. a.
* **Kleiner Maßstab, eigene Umsetzungen.** 60 Gegenstände, Kapazität 100, drei erzeugte Familien, 20 Trainingsinstanzen (für zwei Familien
  auch 100), 100 bis 1 000 Auswertungen, 20 Läufe je Zelle. Keine echten Daten, keine anderen Kapazitäten, keine sortierte oder gegnerische
  Reihenfolge, nicht irace, nicht HyFlex. Ein Nutzen für die Praxis ist nicht belegt.
* **Vorwissen in den Vorlagen.** Vier der fünf Vorlagen enthalten Best Fit als Randfall. Unterschiede zwischen Schablonen und Bäumen sind
  deshalb kein reines Maß für die Güte der Suche.
* **Die Orakelwahl ist eine obere Schranke, keine Methode.** Sie setzt voraus, dass im Einsatz alle Regeln der Menge auf jeder Instanz laufen.
  Gemessen sind zwei Wahlen mit einem Lauf je Instanz: ein 1-nächster Nachbar auf zwei Merkmalen der ersten zehn Gegenstände und eine feste Wahl.
  Andere Merkmale, andere Lerner und eine Wahl nach einer Probepackung sind nicht gemessen; die Aussagen gelten für diese beiden Wahlregeln.
* **Entscheidungsregel ohne Ausgleich.** »Besser« heißt: Median höchstens −0,2 Prozentpunkte und mindestens 15 von 20 Läufen besser. Über
  die vielen Zellen wird nicht ausgeglichen; »gleich« heißt »nicht nach dieser Regel unterscheidbar«, nicht »kein Unterschied«.
* **Bezug ist die untere Schranke, nicht das Optimum.** Der Überschuss über die untere Schranke überschätzt den Abstand zum Optimum; bei
  60 Gegenständen ist das Optimum nicht gemessen.
* **Suchverfahren nicht abgestimmt.** Populationsgrößen, Turniergrößen und Mutationsstärken sind fest. Budgets über 1 000 Auswertungen sind
  nur in zwei Läufen mit 10 000 Auswertungen angesehen worden, das ist kein Befund.
* **Zählung in Auswertungen.** Sie zählt alle Regeln gleich; bei verschieden teuren Regeln ist das nur begrenzt gültig (Branke u. a. 2016).
* **Vielfalt der Instanzen.** Der Gewinn der Menge fällt je Familie verschieden aus; drei Familien reichen nicht, um daraus eine Ursache
  abzulesen.
"""

MATHE = r"""
**Lesehilfe.** Gegeben sind ganzzahlige Gegenstandsgrößen in Ankunftsreihenfolge und Behälter der Kapazität $C$. Jeder Gegenstand muss sofort
und endgültig in einen Behälter. Gesucht ist eine Platzierungsfunktion $f$, die auf frischen Instanzen weniger Behälter braucht als Best Fit,
oder eine Menge $S$ solcher Funktionen.

| Symbol | Bedeutung |
|---|---|
| $C = 100$ | Kapazität jedes Behälters |
| $n$ | Zahl der Gegenstände einer Instanz (hier 60) |
| $s_t \in \{1, \dots, C\}$ | Größe des $t$-ten Gegenstands |
| $m_t$ | Zahl der offenen Behälter vor dem $t$-ten Gegenstand |
| $\rho_j$ | Restkapazität des offenen Behälters $j$ ($j = 0, \dots, m_t - 1$); $\rho_{m_t} = C$ für den leeren Behälter |
| $s, r, g, p, e$ | Merkmale eines Kandidaten: $s = s_t / C$, $r = \rho_j / C$, $g = r - s$, $p = j / (m_t + 1)$, $e = 1$ nur für $j = m_t$ |
| $f$ | Platzierungsfunktion $f(s, r, g, p, e) \in \mathbb{R}$ |
| $B(f, I)$ | Zahl der Behälter, die $f$ auf der Instanz $I$ öffnet |
| $\mathrm{LB}(I)$ | untere Schranke |
| $D$, $T$ | Trainingsinstanzen ($K = 20$ oder $100$) und Testinstanzen |
| $F_K(f)$ | Trainingsfitness (kleiner ist besser); ihre Berechnung ist eine Auswertung |
| $P$ | Pool: alle in einem Lauf bewerteten Regeln |
| $S \subseteq P$, $k = \lvert S \rvert$ | Menge von Regeln und ihre Größe |
| $G(S)$ | Güte einer Menge auf dem Training |
| $\Delta$ | Überschuss der Regel oder Menge minus Überschuss von Best Fit, in Prozentpunkten (positiv = schlechter) |

**Zulässige Kandidaten** beim $t$-ten Gegenstand (der leere Behälter ist immer dabei):

$$J_t = \{\, j < m_t : \rho_j \ge s_t \,\} \cup \{ m_t \}$$

**Wahl** (höchster Wert, bei Gleichstand der kleinste Index):

$$j_t^* = \min \operatorname*{arg\,max}_{j \in J_t} f\!\left(\tfrac{s_t}{C},\ \tfrac{\rho_j}{C},\ \tfrac{\rho_j - s_t}{C},\ \tfrac{j}{m_t + 1},\ [\,j = m_t\,]\right)$$

Ist $j_t^* = m_t$, wird ein neuer Behälter geöffnet. Danach gilt $\rho_{j_t^*} \leftarrow \rho_{j_t^*} - s_t$. Die Behälterzahl ist $B(f, I) = m_{n+1}$.

**Untere Schranke und Überschuss:**

$$\mathrm{LB}(I) = \left\lceil \frac{1}{C} \sum_{t=1}^{n} s_t \right\rceil$$

$$\mathrm{ex}(f, I) = \frac{B(f, I)}{\mathrm{LB}(I)} - 1$$

**Trainingsfitness** und **Delta gegen Best Fit** auf einem Satz $\mathcal{I}$ von Instanzen:

$$F_K(f) = \frac{1}{K} \sum_{I \in D} \frac{B(f, I)}{\mathrm{LB}(I)}$$

$$\Delta(f) = 100 \cdot \left( \frac{1}{|\mathcal{I}|} \sum_{I \in \mathcal{I}} \mathrm{ex}(f, I) - \frac{1}{|\mathcal{I}|} \sum_{I \in \mathcal{I}} \mathrm{ex}(f_{\text{BF}}, I) \right)$$

**Bekannte Regeln:** First Fit $f = -p$, Best Fit $f = -g$, Worst Fit $f = r - 2e$.

**Vorlagen (Schablonen).** S1 $-\lvert g - a \rvert$ mit $a \in [0;\, 0{,}4]$; S2 $-\min(\lvert g - a \rvert, \lvert g - b \rvert)$ mit $a \in [0;\, 0{,}15]$, $b \in [0{,}1;\, 0{,}6]$; S3 $-g - \lambda\,[0 < g < m]$ mit $\lambda \in [0;\, 1]$, $m \in [0{,}02;\, 0{,}4]$; S4 $-g - w p$ mit $w \in [0;\, 3]$; S5 $-\lvert g - a s \rvert$ mit $a \in [0;\, 2]$. Ein Individuum ist eine Vorlage mit Parametern, optional plus eine zweite Vorlage mit Gewicht $\mu \in [0, 1]$:

$$f = f_1 + \mu f_2$$

**Vorlagen-Operator** (Wahrscheinlichkeiten 0,5, 0,2 und 0,3): die Parameter $\theta_i \in [\mathrm{lo}_i, \mathrm{hi}_i]$ des ersten oder zweiten Terms stören (alle Parameter des gewählten Terms, je Parameter unabhängig),

$$\theta_i' = \min\bigl(\mathrm{hi}_i,\ \max\bigl(\mathrm{lo}_i,\ \theta_i + 0{,}1\,(\mathrm{hi}_i - \mathrm{lo}_i)\,z_i\bigr)\bigr), \qquad z_i \sim \mathcal{N}(0, 1)$$

oder die Vorlage des ersten Terms wechseln (Parameter gleichverteilt neu gezogen), oder den zweiten Term anfügen (mit $\mu$ gleichverteilt in $[0, 1]$); ist schon ein zweiter Term da, wird er mit 0,4 entfernt, mit 0,3 sein Gewicht gestört (Streuung 0,2) und mit 0,3 durch einen neuen ersetzt.

**Bäume.** Ausdrücke über $s, r, g, p, e$ und Konstanten in $[-1, 1]$ mit den Operatoren $+$, $-$, $\times$, geschützte Division (ist der Betrag des Nenners höchstens $10^{-9}$, ist das Ergebnis 1), min, max; Anfangstiefe 2 bis 4, Höchsttiefe 6; Teilbaum-Kreuzung mit Wahrscheinlichkeit 0,8, sonst Teilbaum-Mutation.

**Arme.** Z-S und Z-T: unabhängige Ziehungen. ZW-S: zehn Schablonen, je Runde zehn Kinder gleichverteilt gezogener Eltern, die Kinder ersetzen die Eltern. EA-S: wie ZW-S, Eltern durch Turnier der Größe 2, aus Eltern und Kindern bleiben die zehn Besten. GP0-T: 30 Bäume, Eltern gleichverteilt, keine Elite. GP-T: Turnier der Größe 3, Elite 1. Berichtet wird der Bestwert nach $N$ Auswertungen; jede bewertete Regel zählt, auch wenn ihr Wert aus dem Zwischenspeicher kommt.

**Güte einer Menge** (Summe der Instanz-Minima auf dem Training) und **gierige Menge**:

$$G(S) = \sum_{I \in D} \min_{f \in S} B(f, I)$$

$$f_{i+1} \in \operatorname*{arg\,min}_{f \in P \setminus S_i} G(S_i \cup \{f\}), \qquad S_{i+1} = S_i \cup \{f_{i+1}\}$$

Bei Gleichstand gewinnt die Regel mit der kleineren Trainingsfitness, danach die früher bewertete. Die Menge »topk« besteht aus den $k$ Regeln mit der kleinsten Trainingsfitness; die Zufallsmenge aus $k$ gleichverteilt gezogenen Regeln des Pools.

**Einsatz auf einer Instanz $I$.** Orakelwahl ($k$ Läufe je Instanz):

$$B_{\text{Orakel}}(S, I) = \min_{f \in S} B(f, I)$$

Merkmalswahl (ein Lauf je Instanz): $x(I)$ = Mittel und Streuung der ersten zehn Gegenstandsgrößen, beide mit Mittel und Streuung der Trainingsinstanzen normiert; $I'$ ist die Trainingsinstanz mit dem kleinsten quadrierten Abstand zu $x(I)$, und es läuft die Regel der Menge, die auf $I'$ die wenigsten Behälter braucht. Feste Wahl (ein Lauf je Instanz): die Regel der Menge mit der kleinsten Summe der Behälterzahlen auf dem Training.

**Monotonie und abnehmender Zusatznutzen.** Für eine Instanz sei $b_S = \min_{f \in S} B(f, I)$. Eine weitere Regel $h$ senkt den Wert um $\max(0,\ b_S - B(h, I)) \ge 0$. Aus $S \subseteq S'$ folgt $b_S \ge b_{S'}$ und damit

$$\max(0,\ b_S - B(h, I)) \ \ge\ \max(0,\ b_{S'} - B(h, I))$$

Die Summe über die Instanzen erhält beides: eine Regel mehr verschlechtert $G$ nie, und ihr Nutzen ist in der kleineren Menge mindestens so groß wie in der größeren. Eine Gütegarantie für die gierige Menge wird hier nicht behauptet.

**Entscheidungsregel** über $L = 20$ Läufe mit den Deltas $\Delta_1, \dots, \Delta_L$: »besser«, wenn

$$\operatorname{median}(\Delta_1, \dots, \Delta_L) \le -0{,}2 \quad \text{und} \quad \#\{\, l : \Delta_l < 0 \,\} \ge 15$$

»schlechter« spiegelbildlich, sonst »gleich«. Die Wahrscheinlichkeit, dass bei gleich wahrscheinlichen Vorzeichen mindestens 15 von 20 Läufen auf einer Seite liegen, ist

$$\sum_{k=15}^{20} \binom{20}{k} \, 2^{-20} \approx 0{,}0207$$

**Festlegungen der Messung.** NaN in einer Bewertung wird zu $-10^{18}$, $\pm\infty$ zu $\pm 10^{18}$. Abstände der Form $\lvert g - a \rvert$ werden auf 12 Nachkommastellen gerundet, damit gleiche Abstände gleich bleiben. Das Modell der Platzierung, der Zielgröße, der Mengen und der Entscheidungsregel ist vollständig; die Arme der Suche sind als Skizze beschrieben (ohne Pseudocode).

**Quellen.** Romera-Paredes u. a. (2024), FunSearch, Nature 625. Liu, Tong, Yuan, Lin, Luo, Wang, Lu, Zhang (2024), EoH, ICML 2024 (arXiv 2401.02051). Ye, Wang, Cao, Berto, Hua, Kim, Park, Song (2024), ReEvo, NeurIPS 2024 (arXiv 2402.01145). Liu, Liu, Zhang, Tong, Yuan (2025), EoH-S, Vorabdruck (arXiv 2508.03082). Zhang, Liu, Lin, Wang, Lu, Zhang (2024), Understanding the Importance of Evolutionary Search in AHD with LLMs, PPSN 2024 (arXiv 2407.10873). Sim, Renau, Hart (2025), Beyond the Hype, EvoApplications 2025 (arXiv 2501.11411). Branke, Nguyen, Pickardt, Zhang (2016), Automated design of production scheduling heuristics: a review, IEEE Transactions on Evolutionary Computation 20(1), 110–124. Die Angaben stammen aus der Literaturrecherche zu dieser Reihe und wurden beim Bau nicht erneut geprüft.
"""
