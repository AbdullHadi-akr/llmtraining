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

### Was wegfällt

Padding, Geisterschichten, der nicht-äquidistante x-Stencil, der Kreuzterm für
`λ_xy` als FD. Die Basis trägt das alles implizit:

> **Die Symmetrie ist gratis und exakt.** `dT/dx = 0` an der Zellmitte ist eine
> *homogene lineare* Bedingung. Jeder Trainings-Schnappschuss erfüllt sie, also
> erfüllt sie jede Linearkombination — und `Φ` spannt nichts anderes auf. Kein
> Padding, kein `w_bc`, kein Strafterm. Das ist sauberer als die
> Geisterschicht-Variante, die dafür gebaut worden wäre.

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
| 02.09. | `L_phys` bleibt — ein Seed ist keine Streuung |
| 02.09. | Kritik am Versuchsplan: [`README_OPS.md`](README_OPS.md). Schwerster Befund: `T0 = T_fluid` in 11 von 11 |
| 02.09. | O16 im PINN-Fahrplan eingetragen; PR #31 gemergt |
| **09.09.** | **Stufe 0 gelaufen — ROT.** 4 Moden bei 99.9 %. Der CNN ist abgesagt, das ROM ist der Plan |
| **09.09.** | **Stufe 1, erster Lauf.** `jr2/jr1 = 1`, `U`-Verhältnis 1130 : 50 plausibel. Mein ṁ = 0-Torkriterium war falsch und ist gestrichen |
| **09.09.** | Werkzeug korrigiert: drei Flusslevel statt einem, `T_in`-Rückfallkette, `Q_ht/tot` |

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
| 3 | Galerkin-Löser stabil | | |
| 3 | Physik-Latte, val OP06 / OP09 | | |
| 3b | **Projektionsrest je ausgehaltenem OP** | | |
| 4 | ROM schlägt Physik-Latte | | |
| 5 | `late_mae` gefallen | | |
| 6 | val / test gegen PINNmodulusTwo | | |
