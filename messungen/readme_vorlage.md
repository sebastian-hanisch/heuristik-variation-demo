# Heuristiken variieren und zu Mengen bündeln: ein Sprachmodell-Muster, ohne Sprachmodell gemessen – Streamlit-Demo

<!-- Nach dem Deploy direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-heuristik-variation-demo.streamlit.app/)** -->

Teil der Reihe **„Hyperheuristiken erklärt“** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Die Reihe hat zwei Äste: Heuristiken **auswählen** und Heuristiken **erzeugen**. Diese Demo gehört zum zweiten Ast. Sie baut auf der Demo über erzeugte Platzierungsfunktionen auf (gleiche Instanzen, gleicher Simulator, gleiche Regelräume) und steht als **Kontrast** neben der Demo über erzeugte Prioritätsregeln für die Einplanung: dort verändert eine Teilbaum-Kreuzung die Regeln, hier ein Operator, der nur aus Vorlagen und Parametern besteht.

Die Aufgabe ist eindimensionales **Online-Packen** (Online-Bin-Packing): Gegenstände kommen nacheinander, jeder muss sofort in einen Behälter der Kapazität 100. Eine **Platzierungsfunktion** bewertet für den nächsten Gegenstand jeden offenen Behälter und einen leeren; der Behälter mit dem höchsten Wert bekommt ihn. Verfahren wie FunSearch, EoH und ReEvo lassen ein Sprachmodell solche Funktionen schreiben, behalten die besten und wiederholen das. Diese Demo stellt den Kreislauf **ohne Sprachmodell** nach und misst an Best Fit. Alle Instanzen sind erzeugt, keine echten Daten.

## Kernfrage

**Woher kommt der Gewinn eines Kreislaufs aus Variation und Auswahl beim Packen: aus der Variation, aus der Auswahl der Besten oder aus einer Menge von Regeln, und was kostet das?**

Voraussetzungen der Messung: Kapazität 100, 60 Gegenstände je Instanz, drei erzeugte Instanzfamilien, 20 Trainingsinstanzen (für zwei Familien auch 100), 100 bis 1 000 Auswertungen je Lauf, 200 frische Testinstanzen derselben Verteilung, sechs Arme (eigene Umsetzungen), 20 Läufe je Zelle, kein Sprachmodell. Gezählt wird in **Auswertungen** (eine Regel einmal auf allen Trainingsinstanzen) und getrennt davon in **Platzierungsläufen je Instanz** im Einsatz. Dann gilt:

* **Ein Gewinn einzelner erzeugter Regeln auf frischen Instanzen ist nicht nachweisbar.** {{arme}}
* **Auf frischen Instanzen ist weder ein Vorteil der Variation noch einer der Auswahl der Besten nachweisbar.** {{vergleiche}}
* **Eine gierig gebildete Menge gewinnt mit Orakelwahl (k Läufe je Instanz); mit den zwei gemessenen Wahlregeln mit einem Lauf je Instanz gewinnt sie nicht.** {{greedy3}}
* **Die Merkmalswahl mit einem Lauf je Instanz verliert.** {{merkmal}}
* **Eine solche Menge entsteht auch aus Pools ohne Auswahl der Besten (etwa dem frischen Ziehen); die Menge selbst wird gierig gewählt.** {{poolquelle}}

Das sind Aussagen über diese Räume, dieses Budget, diese Familien und diese Messgröße. Sie sagen nichts über Hyperheuristiken im Allgemeinen, nichts über Sprachmodelle als Variationsoperator, nichts über Budgets oberhalb des Gemessenen und nichts über andere Wahlregeln als die zwei gemessenen (Merkmalswahl und feste Wahl).

## Befunde und Korrekturen gegenüber der Vormessung

Die Vormessung (Planung, `numpy.default_rng`, Abstände in gewöhnlichen Gleitkommazahlen) wurde im Bau mit dem Zufallsgenerator der Demo (SplitMix64) und dem korrigierten Kern der Kopierquelle vollständig neu gerechnet. In der Vormessung brachen Rundungsreste in Abständen der Form |g − a| exakte Gleichstände; der kopierte Kern rundet solche Abstände auf 12 Nachkommastellen. Davon sind alle Zahlen zu Schablonen und zu Mengen aus Schablonen berührt. Wie viel einer Verschiebung an dieser Korrektur und wie viel am Zufallsgenerator liegt, ist nicht getrennt gemessen.

| Aussage der Vormessung | Neumessung dieser Demo | Folge |
|---|---|---|
| Kein Arm ist auf frischen Instanzen „besser“ als Best Fit; „schlechter“ sind die drei Schablonen-Arme auf der gleichverteilten Familie | 0 von 18 Zellen „besser“; „schlechter“ ist eine Zelle, und eine andere: EA-S auf der breit gestreuten Familie (+0,24) | Aussage hält; welche Zelle „schlechter“ ist, hängt an den Zufalls-Seeds |
| 21 Vergleiche auf dem Test: alle „gleich“ | 21 von 21 „gleich“, auch an jeder der zehn Budgetstufen | hält |
| Dieselben 21 Vergleiche auf dem Training: 0 „besser“, 1 „schlechter“ (EA-S gegen GP-T, gleichverteilt) | 3 „besser“, 1 „schlechter“ (EA-S gegen GP-T, breit gestreut) | „auf dem Training ist nichts entschieden“ gestrichen |
| Die Auswahl der Besten bringt auf dem Training etwa 0,2 Punkte im Baumraum (−0,19 / −0,22 / −0,21), bei Schablonen nur auf der gleichverteilten Familie (−0,20) | Schablonen −0,26 / 0,00 / −0,22 (zwei Zellen „besser“), Bäume −0,19 / 0,00 / −0,01 (keine Zelle „besser“) | umgekehrt: der Trainingsgewinn der Auswahl steht bei den Schablonen, und nur in zwei von drei Familien |
| Die Variation allein ist im Baumraum auf dem Training schlechter als frisches Ziehen (+0,19 / +0,04 / +0,02) | Bäume +0,18 / +0,03 / 0,00, Schablonen +0,10 / 0,00 / 0,00; in keiner von 6 Zellen „besser“, in keiner „schlechter“ | hält als „nicht besser“; „schlechter“ ist nach der Entscheidungsregel nicht belegt |
| Der Trainingsvorsprung von EA-S bleibt mit 100 Trainingsinstanzen (−0,65 und −0,25) | −0,41 und −0,29 (mit 20: −0,72 und −0,57): er schrumpft | gestrichen; dieselbe Korrektur steht in der Demo über Platzierungsfunktionen |
| Gierige Menge, k = 3, Orakelwahl gegen Best Fit: EA-S −0,75 / −0,23 / −0,47, GP-T −0,55 / −0,25 / −0,23 | EA-S −0,92 / −0,35 / −0,51, GP-T −0,47 / −0,27 / −0,32; alle 6 Zellen „besser“ | hält, Zahlen neu |
| Dieselbe Menge gegen die beste Einzelregel desselben Laufs: −0,29 bis −0,97 | −0,34 bis −1,18 | Zahlen neu |
| Merkmalswahl gegen Best Fit „schlechter“ in 11 von 12 Zellen, gegen die Einzelregel in 8 von 12 | 12 von 12 und 9 von 12 | hält, ohne Ausnahme |
| topk mit Orakelwahl, k = 5, gegen die Einzelregel: „gleich“ in 28 von 30 Zellen | 25 von 30; die fünf Ausnahmen („besser“) liegen auf anderen Testmengen als der gleichen Familie mit 60 Gegenständen | Aussage auf die Testmenge der gleichen Familie begrenzt: dort 12 von 12 Zellen „gleich“ gegen Best Fit |
| Der Testgewinn der gierigen Menge liegt bei 21 bis 46 % ihres Trainingsgewinns | 29 bis 59 % | Zahl neu |
| Auf Häufungen trainiert gewinnt die Menge auf fremden Familien nichts (GP-T, k = 3: +0,32 und +0,46) | 0,00 und +0,02; in keiner der 8 Zellen „besser“ | hält, die Zahlen liegen näher an null |
| Mit 100 Trainingsinstanzen: Orakelwahl −1,23 (EA-S) und −0,80 (GP-T) auf der gleichverteilten Familie, Merkmalswahl +0,25 bis +0,60 | −1,11 und −0,98, je 10 von 10 Läufen unter null; Merkmalswahl +0,43 und +0,40 | hält |
| Best Fit, First Fit, Worst Fit mit Orakelwahl: −0,08 / −0,05 / −0,02 | −0,09 / −0,09 / −0,04 | hält |
| Drei beliebige Regeln des Pools gewinnen −0,21 gegen Best Fit (18 von 20 Läufen), etwa 28 % des Gewinns der gierigen Menge (EA-S, gleichverteilt) | −0,16 (17 von 20 Läufen, Urteil „gleich“), 18 % | schwächer; bei fünf beliebigen Regeln in derselben Zelle −0,46 und „besser“ |
| (in der Vormessung nicht gemessen) Menge aus dem Pool des frischen Ziehens | gierige Menge, k = 3: −1,20 / −0,54 / −0,43, in 2 von 3 Familien vor dem Pool von EA-S | neuer Befund: die Menge gewinnt auch aus Pools ohne Auswahl der Besten (alle 18 Zellen aus 6 Pools und 3 Familien „besser“) |
| (nicht gemessen) Best Fit plus zwei beliebige Regeln des Pools | EA-S −0,29 / −0,11 / −0,20, GP-T −0,09 / −0,08 / −0,06; über k = 3 und 5 in 4 von 12 Zellen „besser“ | neuer Befund |
| Elternteil Best Fit: 1,4 % der Kinder besser, 56,9 % mit denselben Behälterzahlen; 13 von 83 besseren halten auf dem Test (eine Familie) | 2,0 / 2,6 / 2,2 % besser, 57,2 / 78,5 / 56,7 % mit denselben Behälterzahlen; es halten 10 von 123, 115 von 159 und 6 von 135 | „die Mehrheit hält nicht“ gilt für zwei Familien, **nicht** für die Familie mit Häufungen |
| Elternteil frische Schablone: 50,2 % der Kinder besser | 35,8 / 33,6 / 40,0 % | Zahl neu (die frische Schablone wird je Trainingssatz neu gezogen) |
| First Fit und Best Fit unterscheiden sich auf 43 / 4 / 24 von 100 Instanzen | 71 / 8 / 57 von 200 | gleiche Reihenfolge der Familien |

Nicht aus der Vormessung übernommen: die Zufalls-Seeds der Arme. Die vier Arme, die es auch in der Demo über Platzierungsfunktionen gibt, benutzen deren Zufalls-Seeds; ihre 240 Läufe sind dieselben wie dort (ein Test vergleicht sie, wenn beide Ordner vorliegen).

## Modell

Jede Instanz ist eine Folge von 60 ganzzahligen Größen; Kapazität 100. Drei Familien: gleichverteilt auf 20 bis 70; drei Häufungen um 20, 35 und 55 (Normalstreuung 2); breit gestreut (Normal mit Mittel 40 und Streuung 22, gerundet, auf 1 bis 100 begrenzt). Eine Platzierungsfunktion f(s, r, g, p, e) bekommt die Größe s des Gegenstands, die Restkapazität r des Behälters, die Lücke g = r − s nach dem Einsetzen (alle als Anteil der Kapazität), die relative Position p des Behälters und das Kennzeichen e für den leeren Behälter. Der zulässige Behälter mit dem höchsten Wert bekommt den Gegenstand, bei Gleichstand der mit dem kleinsten Index. Best Fit ist f = −g, First Fit f = −p, Worst Fit f = r − 2e.

Messgröße ist der **Überschuss** über die untere Schranke (Summe der Größen durch die Kapazität, aufgerundet): Behälterzahl durch Schranke, minus 1, gemittelt über die Instanzen. **Delta** heißt überall: Überschuss der Regel oder Menge minus Überschuss von Best Fit, in Prozentpunkten; negativ = besser als Best Fit. Der Überschuss über die untere Schranke überschätzt den Abstand zum Optimum; bei 60 Gegenständen ist das Optimum nicht gemessen. {{optimum}}

**Entscheidungsregel** (dieselbe wie in der Demo über Platzierungsfunktionen): „besser“, wenn der Median des Deltas über 20 Läufe höchstens −0,2 Punkte beträgt und mindestens 15 von 20 Läufen ein Delta unter null haben; „schlechter“ spiegelbildlich; sonst „gleich“. Der einseitige Vorzeichentest für 15 von 20 hat p = 0,0207. Es gibt keinen Ausgleich über die Zellen: bei 42 Zellen wären unter lauter Nullunterschieden höchstens 42 · 0,0207 = 0,87 Zufallstreffer „besser“ zu erwarten (Obergrenze, weil die Regel zusätzlich einen Median von höchstens −0,2 Punkten verlangt; die Zellen sind nicht unabhängig). „Gleich“ heißt „nicht nach dieser Regel unterscheidbar“. Für Zellen mit 10 Läufen (100 Trainingsinstanzen) nennt die Demo Mediane und Zählungen, kein Urteil. Die Formeln stehen vollständig im Expander „📐 Mathematische Formulierung“ der App.

## Methodik

**Regelräume.** Schablonen: fünf Vorlagen mit ein oder zwei Parametern (Lückenziel, Kantenanpassung, Lückenstrafe, Positionsmischung, Lücke proportional zur Größe), optional plus eine zweite Vorlage mit Gewicht. Vier der fünf Vorlagen enthalten Best Fit als Randfall, die Kantenanpassung nicht. Bäume: Ausdrücke über s, r, g, p, e und Konstanten mit +, −, ×, geschützter Division, min, max; Höchsttiefe 6.

**Vorlagen-Operator.** Ein Schritt verändert eine Schablone: mit Wahrscheinlichkeit 0,5 die Parameter des ersten oder zweiten Terms stören (alle Parameter des gewählten Terms, je Parameter eine Normalstreuung von einem Zehntel der Breite seines Bereichs), mit 0,2 die Vorlage wechseln, mit 0,3 einen zweiten Term anfügen, ändern oder entfernen.

**Sechs Arme, gleiches Budget in Auswertungen.**

| Arm | Raum | Variation | Auswahl der Besten |
|---|---|---|---|
| Z-S | Schablonen | keine, jede Auswertung zieht frisch | keine |
| ZW-S | Schablonen | Vorlagen-Operator, zehn Kinder je Runde | keine: Eltern gleichverteilt, Kinder ersetzen die Eltern |
| EA-S | Schablonen | Vorlagen-Operator, zehn Kinder je Runde | Turnier der Größe 2, aus Eltern und Kindern bleiben die zehn Besten |
| Z-T | Bäume | keine, jede Auswertung zieht frisch | keine |
| GP0-T | Bäume | Teilbaum-Kreuzung (0,8) oder Teilbaum-Mutation (0,2), Population 30 | keine: Eltern gleichverteilt, keine Elite |
| GP-T | Bäume | wie GP0-T | Turnier der Größe 3, Elite 1 |

„Variation allein“ ist der Vergleich Z-S gegen ZW-S (Z-T gegen GP0-T), „Auswahl der Besten“ der Vergleich ZW-S gegen EA-S (GP0-T gegen GP-T), jeweils gepaart je Lauf auf denselben Trainingsinstanzen. Berichtet wird der Bestwert nach N Auswertungen; jede bewertete Regel zählt, auch wenn ihr Wert aus dem Zwischenspeicher kommt.

**Mengen.** Der Pool eines Laufs sind alle verschiedenen bewerteten Regeln. Daraus entstehen Mengen der Größe k: **gierig** (wiederholt die Regel aufnehmen, die die Summe der Instanz-Minima auf dem Training am stärksten senkt; Gleichstand: bessere Einzelfitness, dann die früher bewertete), **topk** (die k Regeln mit der besten Trainingsfitness), **beliebig** (k gleichverteilt gezogene Regeln, je Lauf fünf Zufallsmengen, das Mittel zählt), **Best Fit plus k − 1 beliebige**. Daneben steht die feste Menge Best Fit, First Fit, Worst Fit.

**Wahl im Einsatz.** Orakelwahl: alle k Regeln laufen auf jeder Instanz, die beste Packung zählt (k Läufe je Instanz). Merkmalswahl: Mittel und Streuung der ersten zehn Gegenstandsgrößen, 1-nächster Nachbar unter den Trainingsinstanzen, es läuft die Regel, die auf dieser Trainingsinstanz am besten war (1 Lauf je Instanz). Feste Wahl: die Regel der Menge mit der kleinsten Summe der Behälterzahlen auf dem Training (1 Lauf je Instanz).

**Testmengen.** Gleiche Familie mit 60 Gegenständen (200 Instanzen), mit 30 (200) und mit 120 Gegenständen (100 Instanzen), dazu die beiden anderen Familien mit 60 Gegenständen (je 200).

**Handbeispiel.** Vier Regeln (Best Fit, First Fit, Worst Fit, Lückenziel mit a = 0,10), vier Trainings- und zwei Testinstanzen mit je acht Gegenständen: Best Fit und das Lückenziel haben beide die Trainingssumme 15; die gierige Menge nimmt Best Fit und dann Worst Fit (Summe der Instanz-Minima 14). Auf den zwei Testinstanzen braucht Best Fit 8 Behälter, die Orakelwahl über {Best Fit, Worst Fit} 7, die Merkmalswahl 9, die feste Wahl 8.

**Zufall und Größe der Messung.** Alle Zufallszahlen kommen aus SplitMix64 (`hva_rng.py`), kein `numpy.random`. Eingefroren sind 360 Läufe mit je 1 000 Auswertungen (6 Arme, 3 Familien, 20 Läufe; Bestwert an zehn Stufen), 60 Läufe mit 100 Trainingsinstanzen (3 Arme, 2 Familien, 10 Läufe), 54 000 Kinder (3 Familien, 3 Elternteile, 20 Trainingssätze, je 300 Kinder) und 2 Läufe mit 10 000 Auswertungen: zusammen 494 000 gezählte Auswertungen. Die Trainingsinstanzen und die Läufe der Arme Z-S, EA-S, Z-T und GP-T sind dieselben wie in der Demo über Platzierungsfunktionen (gleiche Zufalls-Seeds).

## Befunde (gemessen, keine Behauptungen)

Reihenfolge in Dreiergruppen: gleichverteilt / Häufungen / breit gestreut. Zahlen sind Mediane über 20 Läufe in Prozentpunkten gegen Best Fit, wenn nichts anderes dabeisteht.

**Einzelne Regeln**

* {{arme}}
* {{laeufe}}
* Median je Arm (Training und Test, je drei Familien): {{arm_Z-S}}; {{arm_ZW-S}}; {{arm_EA-S}}; {{arm_Z-T}}; {{arm_GP0-T}}; {{arm_GP-T}}.
* {{kandidaten}}
* {{pool}}

**Variation gegen Auswahl**

* {{vergleiche}}
* {{stufen}}
* {{variation}}
* {{auswahl}}
* {{operator}}
* {{bonferroni}}

**Ein Schritt der Variation**

* {{kinder_bf}}
* {{kinder_s1_20}}
* {{kinder_frisch}}
* {{kinder_art}}

**Mengen von Regeln**

* {{greedy3}}
* {{greedy5}}
* {{merkmal}}
* {{fest}}
* {{topk}}
* {{zufall}}
* {{bfplus}}
* {{poolquelle}}
* {{portfolio}}
* {{training_test}}

**Andere Instanzen, mehr Trainingsinstanzen, mehr Budget**

* {{wechsel}}
* {{merkmal_alle}}
* {{gross_EA-S}}
* {{gross_GP-T}}
* {{gross_regel}}
* {{zehntausend}}
* {{spielraum}}

**Zahl → Test**

| Zahl dieser README | Test (Band oder exakt) |
|---|---|
| Jede Zeile der Befundliste oben | `tests/test_claims.py`: aus `hva_precomputed.json` neu gerechnet und wörtlich mit der README verglichen (Zählungen exakt, Mediane auf zwei Stellen) |
| Die Datei `hva_precomputed.json` | `tests/test_neu_rechnen.py`: Schema und Vollständigkeit; ein voller Lauf mit allen Mengen, drei Läufe bis 200 Auswertungen, ein Lauf mit 100 Trainingsinstanzen bis 100 Auswertungen, drei Zellen der Kinder, Spielraum und feste Menge neu gerechnet (Zählungen exakt, Deltas mit Toleranz 1e-5) |
| Kein Arm „besser“, kein Vergleich auf dem Test „besser“, Variation allein auf dem Training nie „besser“ (alle zehn Stufen) | `tests/test_claims.py`, Zählung über das ganze Raster |
| Gierige Menge mit Orakelwahl „besser“ in 12 von 12 Zellen (Pools von EA-S und GP-T, k = 3 und 5), Merkmalswahl nie „besser“, topk in 12 von 12 Zellen „gleich“ | `tests/test_claims.py`, exakt |
| Handbeispiel (15, 14, 8, 7, 9, 8) | `tests/test_handbeispiel.py`, exakt, von Hand gerechnet |
| p = 0,0207; 18 von 20 mit Bonferroni; 0,87 | `tests/test_verdicts.py`, `tests/test_claims.py`, exakt |
| Wahrscheinlichkeiten des Operators 0,5 / 0,2 / 0,3 | `tests/test_oracle_heuristik_variation.py`, Band (4,5 Standardfehler über 20 000 Züge) |
| 494 000 gezählte Auswertungen und die Größen der Blöcke | `tests/test_claims.py`, exakt |
| Die Läufe mit 10 000 Auswertungen | nur Schema geprüft, nicht neu gerechnet: Messung dieser README, nicht im Test |
| Rechenzeiten in den Abschnitten „Verifikation“ und „Tests ausführen“ | Messung dieser README, nicht im Test |

**Lesehilfen zu den Befunden.** Die Zeile „Woher der Pool kommt“ beschreibt eine Zählung, keine Ursache: der Pool des frischen Ziehens enthält viele verschiedene, meist schlechte Regeln, und die gierige Wahl sucht daraus Regeln, die sich auf den Trainingsinstanzen ergänzen. Ob der Gewinn mehr ist als ein Minimum über verschieden streuende Packungen, ist eine Deutung. Die Zeile zur Merkmalswahl gilt für die eine gemessene Wahlregel (1-nächster Nachbar auf zwei Merkmalen), nicht für Wahlregeln allgemein. Die Reihenfolge der Familien beim Gewinn der Menge und bei der Zahl der Instanzen, auf denen First Fit und Best Fit sich unterscheiden, ist bei drei Familien keine Statistik.

**Literatur (laut Quelle, mit Belegstufe der Planung).** Romera-Paredes u. a., FunSearch, Nature 625 (2024): Volltext über einen Abrufdienst zusammengefasst. Liu u. a., EoH, ICML 2024 (arXiv 2401.02051); Ye u. a., ReEvo, NeurIPS 2024 (arXiv 2402.01145); Liu u. a., EoH-S, Vorabdruck (arXiv 2508.03082): Volltext gelesen. EoH-S bewertet eine Menge je Instanz als Minimum über die Menge; das entspricht der Orakelwahl dieser Demo und setzt k-fachen Aufwand je Instanz oder eine fehlerfreie Wahl voraus. Zhang u. a., PPSN 2024 (arXiv 2407.10873, Volltext gelesen): die Basislinien dort benutzen selbst ein Sprachmodell; das frische Ziehen und die Variation ohne Auswahl dieser Demo sind dazu ein Analogon ohne Sprachmodell, keine Wiederholung. Sim, Renau, Hart, EvoApplications 2025 (arXiv 2501.11411, Volltext gelesen): Best Fit als Gesamtsieger über 12 Mengen mit 6 064 Instanzen gegen fünf Heuristiken, die Sprachmodelle erzeugt haben; anderer Maßstab, hier nur Kontext. Branke, Nguyen, Pickardt, Zhang (2016), IEEE Transactions on Evolutionary Computation 20(1), 110–124 (Volltext gelesen): gleiche Auswertungszahlen sind nur begrenzt vergleichbar, wenn Regeln verschieden teuer sind. Alle Angaben stammen aus der Literaturrecherche zu dieser Reihe und wurden beim Bau nicht erneut geprüft; es gibt keinen Literaturtest.

## Ehrliche Grenzen

{{GRENZEN}}

Weitere Punkte, die nur die Messung betreffen:

* **Nicht gemessen:** eine zweite Wahlregel mit einem Lauf je Instanz; Mengen aus Pools einer fremden Familie; Zufallsmengen nur aus den nach Trainingsfitness besten Regeln; 100 Trainingsinstanzen für die Familie mit Häufungen und für andere Testmengen als die der gleichen Familie; wie stark das Delta einer festen Regel von der Wahl der 200 Testinstanzen abhängt.
* **Plattform:** Die Messreihen wurden unter Windows gerechnet. Der Zufall ist plattformfest, die Trainingsfitness ist ein Mittel von Gleitkommazahlen; ein Lauf unter Linux (Docker) steht aus.

## Verifikation

{{TESTZAHL}} Tests (`pytest --collect-only -q tests`).

* **Handbeispiel** (`test_handbeispiel.py`): drei Regeln auf einer Instanz mit Zuordnung, eine Variation über einen Gleichstand, die Tabelle aus vier Regeln und sechs Instanzen, gierige Menge, topk, Orakel-, Merkmals- und feste Wahl, Merkmalswahl in exakten Brüchen, Monotonie und abnehmender Zusatznutzen über alle 15 Teilmengen, Zählung in zwei Einheiten.
* **Orakel** (`test_oracle_heuristik_variation.py`, unabhängige Rechenwege in `tests/referenz.py`): Stapel-Simulator gegen direkte Regeln, Schablonen und Bäume gegen einen skalaren Interpreter, Lückenziel gegen die ganzzahlige direkte Regel; Teilmengen-DP gegen Aufzählung und ganzzahliges Programm; Güte einer Menge gegen naive Schleifen in Brüchen; gierige Menge gegen die Aufzählung aller k-Mengen und Schritt für Schritt gegen die Definition; 1-nächster Nachbar gegen Aufzählung in Brüchen; Operator gegen `s_mutate` und die Anteile der drei Arten; Zählung und Folgentreue aller sechs Arme; Kennzahlen von Hand. Große Fassung beim Bau: 6 135 Prüfungen, 0 Abweichungen (`messungen/orakel_gross.py`). Dabei verfehlte die gierige Menge das Optimum über alle k-Mengen in 62, 58 und 36 von je 400 kleinen Zufallstabellen (k = 2, 3, 4), größte Lücke 4 Behälter: sie ist eine Heuristik, kein Optimum.
* **Eigenschaften und Randfälle** (`test_eigenschaften_mengen.py`, `test_edge_cases.py`): gleicher Zufalls-Seed gleiche Ausgabe, Umnummerieren der Instanzen, Orakelwahl nie schlechter als ein Mitglied, Skalierung, n = 1 und n = 2, k = 1, Pool kleiner als k, alle Regeln gleich, eine Trainingsinstanz, Nenner null, inklusive Schwellen.
* **Kopien** (`test_copies.py`): fünf Kern-Dateien gegen die Quelle (falls vorhanden) und gegen festgehaltene SHA-256-Werte; `hva_rng.py` gegen die Vorlage.
* **Verdikte** (`test_verdicts.py`): jede Schwelle, jeder Zweig, jede Zahl im Text.
* **Eingefrorene Messreihen** (`test_neu_rechnen.py`): Schema, Vollständigkeit, keine Nullspalte ohne Erklärung, Neu-Rechnen eines Ausschnitts als Band, Abgleich der vier gemeinsamen Arme mit den Läufen der Kopierquelle (falls vorhanden).
* **Aussagen** (`test_claims.py`): jede Befundzeile gegen die Datei und gegen diese README; der Text der Grenzen ist in App und README derselbe.
* **Oberfläche** (`test_app.py`, `test_layout.py`, `test_presets.py`, `test_footer.py`, `test_runtime_imports.py`): jeder Button, jeder Regler an beiden Grenzen, jede Auswahl, Worst Case am Slider-Maximum, Permalink, die Schritte des Handbeispiels, Verdikt-Zeilen je Preset, Hervorhebung der Presets, Footer, Laufzeit-Importe.

**Nicht geprüft:** die Linux-CI (`demo_linux_check.py`) steht aus. Im Browser wurde die Seite einmal geladen und jedes Beispielszenario der Seitenleiste einmal angeklickt (kein Skriptfehler, kein Formelfehler bei ausgeklappten Expandern); Regler, dunkles Design und Kopfhöhe wurden im Browser nicht geprüft. Ein unabhängiger Aussagenprüfer hat die Texte nicht gelesen. Kein Mutationstest.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Handbeispiel, Kernfragen, fünf Experimente, eingefrorene Tabellen, Grenzen, Formeln |
| `hva_rng.py`, `hva_algorithm.py`, `hva_scenario.py`, `hva_rules.py`, `hva_search.py` | wörtliche Kopien aus der Demo über Platzierungsfunktionen (Präfix ersetzt): Zufallsgenerator, Simulator, Instanzen, Regelräume, Suche mit Zählung und Pool |
| `hva_arms.py` | die sechs Arme, ihre Zufalls-Seeds, die sieben Vergleiche |
| `hva_variation.py` | ein Schritt des Vorlagen-Operators mit Angabe der Art, Einordnung der Kinder |
| `hva_sets.py` | Güte einer Menge, gierige Menge, topk, Zufallsmengen, Orakel-, Merkmals- und feste Wahl |
| `hva_experiments.py` | Rechnungen der Experimente: Lauf mit Pool, Mengen, Einsatz auf Testmengen, Kinder, Spielraum, Handbeispiel |
| `hva_evaluation.py` | Zahlenformat, Entscheidungsregel, Tabellen aus den eingefrorenen Läufen, Verdikte als reine Funktionen |
| `hva_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `hva_texts.py` | feste Texte der App |
| `hva_constants.py`, `hva_presets.py`, `hva_layout.py` | Grenzen, Schwellen, Presets; Permalink, Hervorhebung, Zufalls-Seed-Knopf; fester Kopf |
| `hva_precomputed.json` | eingefrorene Messreihen |
| `generate_precomputed.py` | erzeugt die Messreihen blockweise; `--check` rechnet einen Ausschnitt neu |
| `messungen/` | Rohdateien je Block, Ausgaben der Läufe, `auswertung.py` (alle Tabellen der Neumessung), `befunde.py` und `readme_bauen.py` (Befundzeilen dieser README), `orakel_gross.py` |
| `tests/` | siehe „Verifikation“ |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit (streamlit, numpy, pandas, plotly) und zusätzlich pytest und scipy für die Tests |
| `.github/workflows/tests.yml`, `.streamlit/config.toml`, `.gitignore` | CI, Streamlit-Einstellungen |

## Bewusst nicht enthalten

* Ein echtes Sprachmodell als Variationsoperator, „Gedanke und Code“, Reflexion, die Frage nach bekannten Heuristiken in den Trainingsdaten eines Modells.
* Der Gewichtsraum und die (1+1)-Suchen, die parametrischen Gegner mit ein und zwei Parametern, das Lesen der erzeugten Regel, Overfitting mit 100 Trainingsinstanzen als eigenes Experiment: das ist Gegenstand der Demo über Platzierungsfunktionen.
* Verteilungswechsel für einzelne erzeugte Regeln; eine Kostenbilanz über die ganze Reihe.
* irace, HyFlex, FunSearch oder andere fremde Umsetzungen; Abstimmung der Suchparameter.
* Budgets über 1 000 Auswertungen als Befund; andere Kapazitäten; sortierte oder gegnerische Reihenfolgen; mehrdimensionales Packen.
* Weitere Wahlregeln für Mengen (andere Merkmale, andere Lerner, Wahl nach einer Probepackung).
* PDF-Export.

## Verwandte Demos

* [Online-Packen](https://sebastianhanisch-online-packen-demo.streamlit.app/): feste Regeln (Next Fit, First Fit, Best Fit, Harmonic) bei großen Instanzen, Bezug FFD; die Zahlen dort sind mit den Zahlen hier nicht vergleichbar (andere Instanzgröße, andere Kapazität, anderer Bezug).
* [Packen: Verfahrenswahl](https://sebastianhanisch-packen-verfahrenswahl-demo.streamlit.app/): Handregeln für Offline-Packen; gelernte Regeln fehlen dort.
* [Reihenfolge suchen](https://sebastianhanisch-reihenfolge-suchen-demo.streamlit.app/): Suche im Raum der Reihenfolgen, Zähleinheit Auswertungen.
* [Genetischer Algorithmus](https://sebastianhanisch-genetic-algorithm-demo.streamlit.app/) für die Begriffe Population, Turnier und Mutation.
* [MAP-Elites](https://sebastianhanisch-map-elites-demo.streamlit.app/): Vielfalt als Archiv über einem Verhaltensraum. Die Vielfalt einer Menge ist hier etwas anderes: die Summe der Instanz-Minima auf dem Training.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Die Messreihen neu erzeugen (beim Bau rund 45 Minuten mit zwei Prozessen nebeneinander, zusammen rund 81 Minuten Rechenzeit, auf einem zugleich anders belasteten Rechner):

```bash
python generate_precomputed.py m1:gl_20_70 m1:clu3 m1:stark m3:gl_20_70 m3:stark m4 m6 m7
python generate_precomputed.py merge
python messungen/readme_bauen.py <Testzahl>
```

Gebaut mit Streamlit, Plotly, pandas und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
