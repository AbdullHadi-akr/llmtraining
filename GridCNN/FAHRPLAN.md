# GridCNN — Fahrplan

> **Lies zuerst [`README.md`](README.md).** Dort steht *warum*; hier steht *was,
> in welcher Reihenfolge, und woran es scheitert*.
>
> Heißt `FAHRPLAN.md` nach der Konvention des Repos — `PINNmodulusTwo` hat
> seinen eigenen, und die `.gitignore` führt genau diesen Namen auf ihrer
> Whitelist.

> ## ⚠ 09.09. — Tor 0 ist ROT. Es wird kein CNN gebaut.
>
> Der Rangtest ist gelaufen. **Die gepoolte Ortsstruktur braucht 4 Moden für
> 99.9 % der Energie** (1 / 2 / 4 / 6 für 90 / 99 / 99.9 / 99.99 %).
>
> Das ist genau der Fall, für den das Tor gebaut war: *„≤ ~5 Moden → 🔴 Umbau.
> Statt Stufe 4–5 wird ein ROM gebaut."* Ein Faltungsstapel über 11 × 11 lernt
> dann einen Raum, der sich mit **vier Zahlen** beschreiben lässt.
>
> Die Stufen 1–3 bleiben **unverändert gültig** — sie sind Physik, nicht
> Architektur. Was sich ändert, steht ab Stufe 4.
>
> Der Ordner heißt weiter `GridCNN`, damit die Verweise aus PR #31 halten. Ein
> `git mv` nach `GridROM` ist ein eigener, mechanischer Commit — Vorschlag, kein
> Alleingang.

**Sortierung:** von oben nach unten — was zu tun ist, steht oben; was erledigt
ist, wandert nach unten in „Erledigt".

Der Plan ist eine **Leiter mit Toren**, keine gerade Linie. **Ein rotes Tor
ändert den Plan, nicht nur den Haken.**

---

## ▶ Das Nächste: `balance_check.py` ein zweites Mal

Der erste Lauf hat drei Dinge geliefert, von denen **zwei einen zweiten Lauf
brauchen** — an beiden war das Werkzeug schuld, nicht die Daten.

```bash
cd /mnt/c/Users/M0245635/batterysurrogatemodell
git checkout main && git pull

# Erst einmal die Spaltennamen sehen -- "dT gemessen" kam als nan:
python3 GridCNN/tools/balance_check.py --ops OP04 --list-columns 2>&1 | tee 09_spalten.txt

# Dann der eigentliche Lauf, jetzt mit DREI Flusslevels statt einem:
python3 GridCNN/tools/balance_check.py 2>&1 | tee 10_bilanz.txt
```

**Was am Werkzeug korrigiert ist:**

| | war | ist |
|---|---|---|
| Default-OPs | OP04, OP05, OP07, OP14 — **beide Fluss-OPs fahren V̇ = 30**, also nur zwei Level | alle sieben Konstant-Treiber-**Trainings**-OPs → Level 0 / 15 / 30, jedes mehrfach |
| `T_in` | ein einziger geratener Spaltenname → `nan` | Rückfallkette bis `Input Signale.csv`, und das Skript sagt, welche Quelle es benutzt hat |
| Abschnitt 1 | nur `Q_ht/jr1` | zusätzlich **`Q_ht/tot`** — die Zahl, die die 2.5 auflöst |
| Erwartung bei ṁ = 0 | „Anteil ≈ 0, sonst 🔴" | **falsch, korrigiert.** Begründung unter Stufe 1 |

> **OP16 fehlt in den Defaults absichtlich.** Es fährt V̇ = 90 und wäre der
> vierte Stützpunkt — aber es ist ein **Test**-OP. `U` daran zu kalibrieren wäre
> eine Auswahl auf dem Extrapolationstier. OP16 ist die Gegenprobe für `U(V̇)`,
> nie die Stütze.

---

## Die Leiter, neu ab Stufe 4

| Stufe | was | Dauer | Tor |
|---|---|---|---|
| **0** | Rangtest | ✅ **erledigt — ROT** | 4 Moden → ROM statt CNN |
| **1** | Bilanz-Gegenprobe | 🟡 **teilweise**, zweiter Lauf offen | geht die Wärmebilanz auf? |
| **2** | vier Größen in den Cache | 30 min | Reports weiter grün? |
| **3** | **Physik ohne Netz** — jetzt als Galerkin-System | Stunden Bauzeit | schlägt reine Physik die trivialen Vorhersager? |
| **3b** | **NEU: trägt die Basis auf den ausgehaltenen OPs?** | Minuten | Projektionsrest auf OP06/09/13/15/16 |
| **4** | das ROM: `g` auf den Modalkoeffizienten | Tage | schlägt es Stufe 3? |
| **5** | truncated BPTT | Tage | fällt der Spätfehler (O13)? |
| **6** | Vergleich gegen PINNmodulusTwo | 1 Lauf | derselbe Split, dieselben Metriken |

> **Der Unterbau für Stufe 3 liegt schon im Repo.** `grid.py`, `physics.py` und
> `solve.py` sind gebaut und geprüft (45 Tests, CI grün) — sie entstanden für
> den CNN, tragen das Galerkin-System aber genauso, weil `Φᵀ L Φ` aus genau
> diesem `L` gebaut wird. Was im Repo steht und was fehlt, steht unten unter
> „Der Code — Stand 14.09."; das Protokoll der Läufe in
> [`BENCHMARK.md`](BENCHMARK.md).

---

# Stufe 1 — Geht die Bilanz auf? 🟡

**Gebaut:** [`tools/balance_check.py`](tools/balance_check.py). Erster Lauf am
09.09., Zahlen in der Stand-Tabelle. Drei Befunde:

## 1a. `Q_ht/JR1 ≈ 2.5` — weder 1 noch 2

Meine Verdikt-Bänder sagten „beide Platten" (1.6 … 2.6), aber 2.5 ist kein
Faktor 2. **Hypothese, die der zweite Lauf prüft:**

> `Q_ht ≈ total_w`, und `total_w / JR1 ≈ 2.5`.

Dokument 030 hat festgestellt, dass `Heat Source Monitor (total)` **nicht**
`JR1 + JR2` ist, sondern größer. Wenn der Monitor die *gesamte* Erzeugung
draint — was im quasistationären Spätfenster zu erwarten ist — dann ist 2.5
**kein Konventionsfehler**, sondern die Aussage:

**Die Zelle erzeugt mehr Wärme als 2 × JR1, und die Modellquelle `q_dot` deckt
nur JR1 ab.** Bei `jr2/jr1 = 1` (gemessen) und `tot/jr1 ≈ 2.5` wären das grob
**20 % der Gesamterzeugung**, die nirgends im Modell vorkommen — Ableiter,
Stromschienen, Kontaktwiderstände.

Das wäre ein neuer offener Punkt für `PINNmodulusTwo` (**O17**), kein
GridCNN-Thema. Aber erst, wenn `Q_ht/tot` es belegt — die Spalte gibt es jetzt.

## 1b. Bei ṁ = 0 bleiben ~0.27 — **mein Torkriterium war falsch**

Ich hatte geschrieben: *„bei ṁ = 0 fließt Energie ab → 🔴 der Entwurf hat einen
Pfad übersehen."* Das war falsch gedacht.

**`ṁ = 0` heißt kein *Fluss*, nicht kein *Fluid*.** Das Kühlmittel steht im
Kanal und nimmt Wärme in seine **eigene Wärmekapazität** auf. Ein
Solid-to-Fluid-Wärmestrom bei stehendem Fluid ist physikalisch richtig — die
Energie wird **gespeichert**, nicht abtransportiert.

Und die Zahlen bestätigen es:

| | `U` [W/m²K] |
|---|---|
| mit Fluss (V̇ = 30) | ~1130 |
| ohne Fluss | ~50 |

Faktor **~23**. Stehendes Flüssigkühlmittel bei O(50) und Zwangskonvektion in
einer Kühlplatte bei O(1000) sind beide lehrbuchplausibel. **Das ist eine
Bestätigung der Messkette, kein Fehler.**

### Was daraus folgt, ist ein Modellbefund

`ΔT_fluid = Q̇ / (ṁ · Cp)` **divergiert für ṁ → 0**. Die Form ist nur für einen
Durchfluss richtig. Richtig ist beides zusammen:

```
C_fluid · dT_fluid/dt  =  Q̇  −  ṁ · Cp · (T_fluid − T_in)
                            ^Quelle      ^Advektion
```

Bei ṁ = 0 bleibt reines Aufladen; bei großem ṁ fällt die alte stationäre Bilanz
heraus. **Eine Zustandsvariable mehr, und die Formel hat keine Singularität.**

> ### Und das trifft genau O14
>
> V̇ = 0 ist das Regime mit dem schlechtesten ausgehaltenen Wert (5.374 gegen
> 2.928 °C im Mittel, OP06 mit 6.270 °C). Bisher war die Erklärung **Abdeckung**
> (O14: nur zwei Trainings-OPs, beide Kälteextreme). Jetzt gibt es daneben einen
> **Mechanismus**: es ist das Regime, in dem die Fluidbeschreibung als reine
> Advektion zusammenbricht.
>
> **Die Abdeckungserklärung bleibt gültig** — die zwei Befunde ersetzen sich
> nicht. Sie sind aber **trennbar**, und das ist die interessante Messung:
> *verbessert ein Kapazitätsterm die V̇ = 0-OPs, ohne dass neue Daten dazukommen?*
> Wenn ja, war O14 nicht nur Abdeckung.

## 1c. `U(V̇)` hat zwei Punkte, nicht eine Kurve

Weil OP04 **und** OP05 beide V̇ = 30 fahren. Behoben, siehe „Das Nächste".

## Das Tor, neu formuliert

| Ergebnis | Folge |
|---|---|
| `Q_ht/tot ≈ 1` | 🟢 kein Konventionsfehler. Stattdessen **O17**: die Modellquelle ist unvollständig |
| `Q_ht/jr1 ≈ 2`, `Q_ht/tot ≉ 1` | 🟡 beide Platten. **Entweder** `Q` halbieren **oder** `A` verdoppeln — nie beides |
| Fluidbilanz-Verhältnis ≈ 1.0 | 🟢 `ghost_hi` steht |
| `U` über drei Flusslevel auf einer Kurve | 🟢 `U(V̇)` wird feste Funktion, null freie Parameter |
| ~~bei ṁ = 0 fließt Energie ab~~ | ~~🔴~~ **gestrichen** — war falsch, siehe 1b |

**Blockiert nichts mehr:** `data_raw/` liegt vor.

---

# Stufe 2 — Die vier Größen in den Cache

Unverändert: `q_solid_to_fluid`, `fluid_out_temp`, `cp_fluid`, `mdot` in
`generate_cache.py` und `opbundle_contract.md`, `schema_version` hoch, alle
sechzehn OPs neu bauen (10–30 min).

**Dazu neu:** `total_w` gehört mit hinein, falls 1a sich bestätigt — sonst ist
O17 nie messbar.

**Tor:** `profile_report`, `coverage_report` und `energy_balance_report` müssen
**exakt dieselben Zahlen** liefern wie vorher.

---

# Stufe 3 — Physik ohne Netz, jetzt als Galerkin-System

Der Rangtest macht diese Stufe **billiger, nicht überflüssig**. Statt eines
expliziten Lösers auf 3 × 11 × 11 wird die Wärmeleitung auf die POD-Basis
projiziert:

```
T(t) = m(t)·1  +  Φ · a(t)          Φ aus Stufe 0, r = 4 … 6

ȧ = (Φᵀ L Φ) · a  +  Φᵀ (Quelle + Wandfluss)
    ^ eine r×r-Matrix, EINMAL berechnet
```

Das ist ein **lineares ODE-System mit sechs Unbekannten** statt eines
Gitterlösers. Sekunden pro Trajektorie statt Stunden.

### Was diese Stufe weiterhin leistet

* **Prüft Wandterm und Randbedingungen unabhängig vom Lernen.** Ist das
  Vorzeichen falsch oder `U` um Faktor 2 daneben, sieht man es hier — und nicht
  drei Wochen später als „das Netz konvergiert schlecht".
* **Liefert die Physik-Latte.** Ernsthafter als `persistence` und `train-mean`.
* **Definiert den Rest.** Was der Galerkin-Löser nicht trifft, ist genau das,
  was `g` lernen muss.
* **Testet den Kapazitätsterm aus 1b** — ohne ein einziges Gewicht.

### Was wegfällt — und was ausdrücklich nicht

> **Präzisiert am 15.09.** Hier stand ursprünglich, Padding, Geisterschichten,
> der x-Stencil und der Kreuzterm fielen weg. Das war zu grob formuliert, und
> mit dem Code aus #37 im Repo würde es zu einem falschen Schluss verleiten:
> dass `grid.py` und `physics.py` mit der Absage des CNN erledigt seien.

**Aus dem Rollout fällt der Gitterlöser heraus** — gerollt wird auf `r + 1`
Zahlen, nicht auf 363 Knoten. Das ist der Gewinn.

**Aus dem Aufbau fällt gar nichts.** `L` in `Φᵀ L Φ` ist genau der Operator aus
`physics.anisotropic_laplacian`, angewandt auf den gepaddeten Stapel aus
`grid.pad_all`. Um die `r × r`-Matrix **einmal** zu bilden, wird `L` auf jede
der `r` Basisfunktionen angewandt — mit Padding, mit Geisterschichten, mit dem
nicht-äquidistanten x-Stern und mit dem Kreuzterm. Danach nie wieder.

Der Unterschied ist also `r + 1` Operatoranwendungen **insgesamt** statt einer
je Zeitschritt und Trajektorie. Deshalb ist die Stufe billiger — nicht, weil
weniger Physik gerechnet würde.

> **Die Symmetrie ist gratis und exakt.** `dT/dx = 0` an der Zellmitte ist eine
> *homogene lineare* Bedingung. Jeder Trainings-Schnappschuss erfüllt sie, also
> erfüllt sie jede Linearkombination — und `Φ` spannt nichts anderes auf. Kein
> `w_bc`, kein Strafterm.
>
> Das **Padding** bleibt trotzdem nötig, und zwar beim Aufbau von `Φᵀ L Φ`:
> `L` braucht Werte außerhalb des Gebiets, egal worauf es angewandt wird. Was
> gratis wird, ist die *Einhaltung* der Bedingung durch den Zustand — nicht
> ihre *Umsetzung* im Operator.

⚠ **Und R6 überlebt die Projektion.** `Φᵀ L Φ` erbt die Bilanzuntreue von `L`
(3.6 % Drift, sättigend). Der Umbau aufs ROM erledigt den Befund nicht, er
macht ihn nur billiger messbar.

### Tor

| Ergebnis | Folge |
|---|---|
| schlägt `train-mean` auf den ausgehaltenen OPs | 🟢 weiter zu Stufe 4 |
| stabil, aber schlechter | 🟡 normal, der Rest ist groß |
| divergiert | 🔴 Vorzeichen im Wandterm oder `U` um Faktor 2 — **nicht mit dem Netz übertünchen** |
| schlägt schon `PINNmodulusTwo` (6.270 / 3.585 °C) | 🔴🟢 dann lief das Lernen bisher gegen fehlende Physik, und der Plan wird ein anderer |

---

# Stufe 3b — NEU: trägt die Basis überhaupt? *(Minuten, kein Training)*

Der Rangtest hat `Φ` auf den **Trainings**-OPs gemessen. Die eigentliche Frage
ist eine andere:

> Erfasst eine Basis aus elf Trainings-OPs auch **OP06, OP09, OP13, OP15,
> OP16** auf 99.9 %?

Gerechnet wird der **Projektionsrest** `‖T − ΦΦᵀT‖ / ‖T‖` je ausgehaltenem OP.
Kein Training, keine Gewichte, Sekunden.

**Das ist eine harte Obergrenze.** Wenn die Basis OP06 nur auf 98 % erfasst,
kann kein `g` der Welt darunter kommen — der Fehler steckt dann in `Φ`, nicht im
Lernen. Und es ist die erste Messung des Projekts, die eine Grenze **vor** dem
Training benennt statt danach.

| Ergebnis | Folge |
|---|---|
| Rest ≲ 0.1 % auf allen fünf | 🟢 die Basis verallgemeinert, `r` bleibt klein |
| Rest groß auf OP06 | 🟡 `r` erhöhen und erneut prüfen — oder es ist O14, dann sagt es das sauber |
| Rest groß überall | 🔴 POD auf Trainings-OPs ist der falsche Ansatz; zurück zu einem Feldmodell |

---

# Stufe 4 — Das ROM

```
Zustand:  [m(t), a(t)]        r+1 = 5 … 7 Zahlen statt 363
Schritt:  [m,a]_{t+1} = [m,a]_t + Δt · g(m, a, Historie, Treiber, q_wall)
g:        MLP, 2-3 Schichten x 32-64  ->  ~5 k Parameter
```

**`g` korrigiert den Galerkin-Löser aus Stufe 3, es ersetzt ihn nicht.**

Der Wandterm bleibt exakt: `T₂` ist eine **lineare Funktion** von `[m, a]` (die
Zeilen von `Φ` auf der Gehäusewand-Ebene), also ist `q_wall` im Modalraum
berechenbar, ohne das Feld zu rekonstruieren.

Verluste: `L = w_data·L_data + w_phys·L_phys + w_wall·L_wall`. **Kein `L_bc`** —
die Basis erfüllt die Symmetrie exakt.

> **Der eigentliche Gewinn steht in der Parameterzahl.** ~5 k gegen ~70–100 k
> beim heutigen MLP, bei elf Trajektorien. Das ist der Punkt, an dem das
> Verhältnis Parameter zu unabhängigen Beispielen erstmals vernünftig aussieht —
> und es ist die Grenze, die ich in README §11.4 als bindend bezeichnet habe.

**Tor:** schlägt Stufe 3 auf den fünf ausgehaltenen OPs.

---

# Stufe 5 — Truncated BPTT

Unverändert der Hebel auf **O13** (OP06: 6.270 °C im Mittel, 13.248 °C spät).
Fenster von `k` Schritten, Gradient durch alle `k`, `detach` am Fensterende.

**Und beim ROM ist es fast gratis:** der Zustand sind sieben Zahlen statt 363,
die Aktivierungen je Schritt sind ein Bruchteil. `k = 500` oder mehr ist
denkbar, wo beim CNN 200 die Grenze war.

**Tor:** `late_mae` fällt, bei mindestens gleichem `mae`.

---

# Stufe 6 — Der Vergleich

Derselbe Split (train 11 / val OP06+OP09 / test OP13, OP15, OP16), dieselben
`op_metrics`, dieselben trivialen Vorhersager, **plus** die Physik-Latte aus
Stufe 3 **plus** die Projektionsgrenze aus Stufe 3b. Über mehrere Seeds — *ein
Seed ist keine Streuung.*

OP19 wird mitgerollt: berichten, nie trainieren, nie selektieren (O11).

---

## Was übernommen wird — importiert, nicht kopiert

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "PINNmodulusTwo"))
import data, op_registry, op_metrics
```

**Kein Modulus, und jetzt auch kein Conv.** Reines PyTorch, und `g` ist so
klein, dass die CPU reicht.

---

## Was NICHT gebaut wird

* **Der Faltungsstapel.** Tor 0 hat ihn abgesagt. Nicht „später vielleicht" —
  abgesagt, bis eine Messung ihn zurückholt.
* **ConvGRU / FNO / DeepONet.** Erledigt sich mit dem CNN.
* **Eine gelernte Basis (Autoencoder).** Erst wenn Stufe 3b sagt, dass POD nicht
  reicht. Ein linearer Unterraum, der auf vier Moden passt, braucht keinen
  nichtlinearen Encoder.
* **Plots, Resume, Checkpoint-Merge.** Zuletzt.

---

## Abbruchkriterien

* **Divergenz in Stufe 3 wird nicht mit dem Netz übertünchert.**
* **Kein `L_bc` durch die Hintertür.** Braucht das Modell einen Strafterm für
  die Symmetrie, ist die Basis falsch aufgebaut.
* **`Q̇` und `T_fluid_out` gehen nie als Modelleingang rein.** Aufsicht und
  Gegenprobe, nichts anderes.
* **`U` wird nie auf OP13/OP15/OP16 kalibriert.** Trainierte Flusslevel sind
  0 / 15 / 30; die 90 aus OP16 sind die Gegenprobe.
* **Ein Seed ist keine Streuung.**

---

# Erledigt

| Datum | was |
|---|---|
| 02.09. | Gitter verifiziert: 3 × 11 × 11, äquidistant, über alle OPs identisch, Randring **auf** der Flächenkante |
| 02.09. | Randbedingungen geklärt; Quellenseite als Halbmodell bestätigt (`V_JR1 = 4.394793e-04 m³`) |
| 02.09. | `A = 0.0206 m²`, Koeffizient als `U` (Gesamtdurchgang) etikettiert |
| 02.09. | `tools/spatial_rank.py` und `tools/balance_check.py` gebaut und getestet |
| 02.09. | `L_phys` bleibt — ein Seed ist keine Streuung (README §12.1) |
| 02.09. | F4–F8 entschieden (README §12.3) |
| 02.09. | Kritik am Versuchsplan: [`README_OPS.md`](README_OPS.md). Schwerster Befund: `T0 = T_fluid` in 11 von 11 |
| 02.09. | O16 im PINN-Fahrplan eingetragen; PR #31 gemergt |
| **09.09.** | **Stufe 0 gelaufen — ROT.** 4 Moden bei 99.9 %. Der CNN ist abgesagt, das ROM ist der Plan |
| **09.09.** | **Stufe 1, erster Lauf.** `jr2/jr1 = 1`, `U`-Verhältnis 1130 : 50 plausibel. Mein ṁ = 0-Torkriterium war falsch und ist gestrichen |
| **09.09.** | Werkzeug korrigiert: drei Flusslevel statt einem, `T_in`-Rückfallkette, `Q_ht/tot` |
| **14.09.** | **Der Gitter-Unterbau steht.** `grid.py`, `physics.py`, `solve.py`, `benchmark.py`, 45 Tests, CI grün |
| **14.09.** | Stencil-Ordnungen gemessen, beide Randzusagen **exakt** null, Route R6 beziffert |

## Stand

| Stufe | Kriterium | gemessen | Datum |
|---|---|---|---|
| 0 | Gitterprobe 3×11×11 | **bestätigt** | 09.09. |
| 0 | **gepoolte Ortsstruktur** (90/99/99.9/99.99 %) | **1 / 2 / 4 / 6 Moden** | **09.09.** |
| 0 | Tor | 🔴 **≤ 5 → ROM statt CNN** | 09.09. |
| 1 | `jr2/jr1` | **1.000** — zwei gleiche Wickel | 09.09. |
| 1 | `Q_ht/jr1` spät, mit Fluss | **≈ 2.5** — weder 1 noch 2, siehe 1a | 09.09. |
| 1 | `Q_ht/tot` | *offen* — Spalte erst jetzt im Werkzeug | |
| 1 | Fluidbilanz-Verhältnis | **nan** — `T_in`-Spalte nicht gefunden, behoben | 09.09. |
| 1 | Wandanteil bei ṁ = 0 | **≈ 0.27** — physikalisch richtig (Fluidkapazität), Kriterium war falsch | 09.09. |
| 1 | `U` mit Fluss / ohne | **~1130 / ~50 W/m²K**, Faktor ~23 — vorläufig, Faktor 2 aus 1a offen | 09.09. |
| 1 | `U(V̇)` auf einer Kurve? | *offen* — erster Lauf hatte nur zwei Level | |
| 2 | Reports unverändert | | |
| — | Reshape aus Koordinaten ableitbar und umkehrbar | **ja**, 0.198089 × 0.104441 m | 14.09. |
| — | `dT/dx` an der Symmetrieebene | **exakt `0.0`**, ohne Toleranz | 14.09. |
| — | zentrale Differenz am y/z-Rand | **exakt `0.0`** | 14.09. |
| — | Stencil-Ordnung d²/dy², d²/dz² | **1.98 / 1.98** | 14.09. |
| — | Stencil-Ordnung d²/dx² (gestreckt) | **1.28** — richtig für den nicht-äquidistanten Dreipunktstern | 14.09. |
| — | Bilanztreue des Sterns | **3.6 % Drift, sättigt** (R6) | 14.09. |
| 3 | Galerkin-Löser stabil | | |
| 3 | Physik-Latte, val OP06 / OP09 | | |
| 3b | **Projektionsrest je ausgehaltenem OP** | | |
| 4 | ROM schlägt Physik-Latte | | |
| 5 | `late_mae` gefallen | | |
| 6 | val / test gegen PINNmodulusTwo | | |

---

# Der Code — Stand 14.09.

> Dieser Abschnitt beschreibt, **was im Repo liegt**. Der Plan oben sagt, was
> gemessen werden soll; hier steht, womit. Das fortlaufende Protokoll der
> Läufe steht in [`BENCHMARK.md`](BENCHMARK.md), zusammen mit den offenen
> Routen R1–R6.

> ### ⚠ Gebaut wurde er als Unterbau für den CNN — er trägt das ROM genauso
>
> Der Code stammt aus PR #37 und ist **vor** der Einarbeitung von Tor 0
> entstanden. Die Frage liegt also nahe, ob er mit der Absage des
> Faltungsstapels hinfällig ist. **Er ist es nicht, und zwar aus einem
> konkreten Grund:** das Galerkin-System aus Stufe 3 ist
>
> ```
> ȧ = (Φᵀ L Φ) · a  +  Φᵀ (Quelle + Wandfluss)
> ```
>
> und `L` **ist** `physics.anisotropic_laplacian`. Der Stern, das Padding, der
> Kreuzterm und der Wandterm werden nicht ersetzt — sie werden **projiziert**.
> Was der CNN als Faltung über (y, z) gebraucht hätte, braucht das ROM als
> Matrix, die einmal aus demselben Operator entsteht.
>
> **Hinfällig ist genau eine Datei, und die gibt es noch nicht:** `model.py`
> als Faltungsstapel. Was daran im Fahrplan stand, ist unten korrigiert.

## Was steht

| Datei | was sie tut | geprüft durch |
|---|---|---|
| [`grid.py`](grid.py) | Reshape 363 → (3,11,11), **aus den Koordinaten abgeleitet**, plus die drei Paddings | `tests/test_grid.py`, 12 Tests |
| [`physics.py`](physics.py) | nicht-äquidistanter x-Stern, Kreuzterm `λ_xy`, Quelle, Wandterm, `U(V̇)` | `tests/test_physics.py`, 19 Tests |
| [`solve.py`](solve.py) | explizites Euler, entdimensioniert wie `data.py` | `tests/test_solve.py`, 14 Tests |
| [`benchmark.py`](benchmark.py) | Stufenläufer, schreibt ins lebende Dokument | CI, Stufe 0 und 2 |
| [`BENCHMARK.md`](BENCHMARK.md) | das Protokoll und die offenen Routen | von Hand |

45 Tests, Sekunden, **ohne Modulus und ohne `data_cache`**. Die CI fährt sie in
einer eigenen Invokation — beide Projekte haben ein Modul namens `physics`, und
in einem gemeinsamen Lauf gewänne das erste.

### Die drei Zusagen, die jetzt bewiesen sind statt behauptet

1. Der Reshape wird **abgeleitet**, nicht geraten, und trifft die dokumentierte
   Geometrie: Spannweite 0.198089 × 0.104441 m, `dx = 10.786 / 11.114 mm`.
2. `dT/dx` an der Symmetrieebene ist **exakt `0.0`** — geprüft ohne Toleranz und
   über Feldskalen von 1e-8 bis 1e8. Im Zähler steht `T₁ − T₁`.
3. Die zentrale Differenz am y/z-Rand ist **exakt `0.0`**. `reflect`, nicht
   `replicate`.

> Zusage 2 wird im ROM **anders erreicht, nicht aufgegeben**: die POD-Basis
> erfüllt die Symmetrie exakt, weil `dT/dx = 0` homogen linear ist und jeder
> Schnappschuss sie erfüllt (Stufe 3). Das Padding bleibt trotzdem geprüft —
> der Löser aus Stufe 3 baut `Φᵀ L Φ` damit auf.

### Ein Befund, der beim Bauen abgefallen ist

**Der Stern ist nicht bilanztreu** (Route R6). Rollt man adiabat aus, driftet
das Mittel um **3.6 % der Feldstreuung** und **sättigt** dann; das Feld läuft
sauber gegen eine Konstante (`std` → 4e-5). Ursache sind zwei bewusste
Entscheidungen: die nicht-konservative Form `Fo : ∇²T` und knotenzentriertes
`reflect` ohne Halbzellgewichte.

Dass die Drift sättigt, ist der Beleg, dass kein Rand *echte* Energie leckt. Sie
gehört aber **neben** die Physik-Latte geschrieben, nicht darunter versteckt.

⚠ **Und sie überlebt die Projektion.** `Φᵀ L Φ` erbt die Eigenschaften von `L`,
also trägt das Galerkin-System denselben Bias. R6 wird durch den Umbau auf das
ROM also **nicht** erledigt — er wird nur billiger zu messen.

## Was fehlt

| | was | blockiert durch |
|---|---|---|
| **`rom.py`** | POD-Basis `Φ` aus den Trainings-OPs, `Φᵀ L Φ` einmal aufgebaut, Projektion und Rekonstruktion | nichts — kann gebaut werden |
| **Stufe 3b** | Projektionsrest je ausgehaltenem OP — die harte Obergrenze **vor** dem Training | `rom.py` |
| **`model.py`** | `g` auf den Modalkoeffizienten: MLP, 2–3 Schichten × 32–64, **~5 k Parameter** | `rom.py` |
| **`train.py`** | Trainingsschleife, Ein-Schritt zuerst (parallel zu `PINNmodulusTwo`) — **kein Teacher Forcing**, siehe Kasten | `model.py` |
| **Wandterm benutzbar** | `U(V̇)` kalibrieren | **Stufe 2**: `q_solid_to_fluid`, `mdot`, `cp_fluid`, `fluid_out_temp` fehlen im Bündel |
| **Physik-Latte** | Stufe 3 mit echter Wand | Wandterm |
| **`C_fluid`** | Wärmekapazität des Kühlmittels im Kanal, für den Kapazitätsmodus | liegt nicht vor — **nicht raten**, siehe R3 |

⚠ **Kein Teacher Forcing — und das wird oft falsch erzählt.**
`train.py:911` rollt **einmal je Epoche je OP** die *eigene* Trajektorie unter
`torch.no_grad()` aus, gesät nur von der gemessenen Anfangsbedingung, und friert
sie ein. Erst darauf laufen `inner_steps` Ein-Schritt-Updates gegen die Labels.
Der Kommentar dort sagt es wörtlich: *„No teacher forcing: this is the
free-running rollout"*, und `evaluate()` wiederholt es.

Der Unterschied zwischen Training und Auswertung ist also **nicht** der
Eingangszustand — beide rollen frei. Er ist:

| | Training | Auswertung |
|---|---|---|
| Trajektorie | eingefroren, je Epoche erneuert | live |
| Labels | ja, als Ziel | nein |
| Gradient | nur **ein** Schritt (Historie detached) | keiner |

**Daraus folgt der ganze Hebel von Stufe 5.** Weil der Gradient die Historie nie
überquert, kann er den Spätfehler (O13) strukturell nicht erreichen — egal wie
lange man trainiert. Truncated BPTT ist die einzige Änderung, die daran etwas
ändert. Wer `train.py` baut, darf diese Eigenschaft nicht versehentlich
wegoptimieren. **Das gilt für das ROM unverändert** — es ist eine Eigenschaft
der Trainingsschleife, nicht der Architektur.

⚠ **Der Löser läuft heute adiabat.** Der Wandterm ist gebaut und getestet, aber
ohne die vier Cache-Größen nicht kalibrierbar. `solve.rollout` trägt das als
Notiz mit und `SolveResult.summary()` schreibt „Wandterm AUS (adiabat)" — das
ist eine **Ablation, keine Latte**, und darf auch nicht als eine zitiert werden.

## Die Architektur, wie sie gebaut wird

**Korrigiert am 15.09.** Der Kasten unten stand in PR #37 noch als
Faltungsstapel über 44 Kanäle. Tor 0 hat ihn abgesagt; hier steht, was an seine
Stelle tritt.

```
Zustand:   [m(t), a(t)]                    r+1 = 5 … 7 Zahlen statt 363
Schritt:   [m,a]_{t+1} = [m,a]_t + Δt · f( m, a, Historie, Treiber, q_wall )

f  =  (Φᵀ L Φ)·a + Φᵀ(Q + q_wall)  +  g_θ(m, a, u_t)
      ^feste Physik, eine r×r-Matrix    ^gelernte Korrektur, ~5 k Parameter
```

| | |
|---|---|
| Eingang | `m`, `a` (5–7 Zahlen), die Historie derselben, **18 Treiber** — keine Karten, keine Kanäle |
| Ausgang | `r+1` Änderungsraten |
| Rekonstruktion | `T = m·1 + Φ·a`, eine Matrixmultiplikation |
| Verlust | `L = w_data·L_data + w_phys·L_phys + w_wall·L_wall` — **kein `L_bc`** |

Die 17 statischen Karten fallen weg: sie waren dafür da, dem Faltungskern die
Translationsäquivarianz zu brechen. Eine Basis, die aus genau diesen Daten
gewonnen wurde, trägt die Ortsinformation schon in `Φ`.

**Die Größe ist eine Konsequenz der Messung.** Der Entwurf sah 64 Kanäle × 4
Blöcke vor (~100 k Parameter), PR #37 hatte auf 16 × 3 ≈ 11 400 verkleinert.
Bei vier gemessenen Moden und elf Trajektorien sind **~5 k** vorgesehen — und
das ist der Punkt, an dem das Verhältnis Parameter zu unabhängigen Beispielen
erstmals vernünftig aussieht (README §11.4).

> **Der Wandterm bleibt exakt.** `T₂` ist eine **lineare Funktion** von `[m, a]`
> — die Zeilen von `Φ` auf der Gehäusewand-Ebene. `q_wall` ist damit im
> Modalraum berechenbar, ohne das Feld zu rekonstruieren.

### Die Ablation steht als Erstes an

Drei Konfigurationen, die sich in genau einer Sache unterscheiden. **Sie gilt
für das ROM unverändert** — sie trennt Architektur von Verlust, und das ist von
der Wahl zwischen Faltung und Basis unabhängig:

| | die Rate `f` | `w_phys` | misst |
|---|---|---|---|
| **A** | `g_θ` allein | 0 | reine Blackbox auf den Moden — die Latte |
| **B** | Galerkin + `g_θ` | 0 | was die Physik **in der Architektur** bringt |
| **C** | Galerkin + `g_θ` | > 0 | was der Strafterm **obendrauf** bringt |

A → B ändert die Architektur bei gleichem Verlust, B → C den Verlust bei
gleicher Architektur. Jede Differenz ist damit einem einzigen Eingriff
zuzuordnen.

⚠ Die Seed-Streuung liegt bei 0.518 / 0.882 °C. Ein Unterschied unter ~1 °C ist
mit drei Seeds **nicht lesbar** — jede Konfiguration braucht eine Seed-Schleife.

## Die nächsten drei Schritte

1. **`balance_check.py` ein zweites Mal** — Minuten, nur numpy. Klärt `Q_ht/tot`
   und `U(V̇)` über drei Flusslevel.
2. **Stufe 2, der Cache-Umbau** — danach ist der Wandterm kalibrierbar und die
   Physik-Latte messbar.
3. **`rom.py` und Stufe 3b** — die Basis bauen und den Projektionsrest auf den
   ausgehaltenen OPs messen. Das ist die billigste Messung im ganzen Plan und
   die einzige, die eine Obergrenze **vor** dem Training nennt: liegt der Rest
   auf OP06 bei 2 %, kommt kein `g` darunter.

> Schritt 1 und 2 brauchen die Rechenmaschine mit `data_raw/` und `data_cache/`.
> Schritt 3 braucht nur den Cache — aber nicht die vier neuen Größen, also kann
> er **parallel** zu Schritt 2 laufen.
