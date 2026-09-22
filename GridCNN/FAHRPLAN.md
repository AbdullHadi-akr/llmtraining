# GridCNN — Fahrplan

> **Lies zuerst [`README.md`](README.md).** Dort steht *warum*; hier steht *was,
> in welcher Reihenfolge, und woran es scheitert*.
>
> Heißt `FAHRPLAN.md` nach der Konvention des Repos — `PINNmodulusTwo` hat
> seinen eigenen, und die `.gitignore` führt genau diesen Namen auf ihrer
> Whitelist.

> ## ⚠ 15.09. — Der CNN wird gebaut, gegen das rote Tor 0
>
> Tor 0 ist am 09.09. rot gefallen (**4 Moden bei 99.9 %**) und hätte ein ROM
> verlangt. Am 15.09. ist entschieden worden, den CNN trotzdem zu bauen; der
> ROM-Plan ist gestrichen.
>
> **Die vollständige Begründung steht im [`README.md`](README.md)** unter
> „Die Entscheidung, die man kennen muss" — hier nur die Folgen für den Plan:
>
> * Gebaut werden **16 × 3 = 11 427 Parameter**, nicht die 64 × 4 aus der
>   Präsentation (die **137 923** kosten, nicht „~100 k" wie §11.4 sagt).
>   Die Größe bleibt als Sweep-Achse erreichbar.
> * Stufe 3 ist wieder der explizite Löser, Stufe 4 der CNN. Stufe 3b entfällt.
> * Der Rangtest und sein Ergebnis bleiben in „Erledigt" und in der
>   Stand-Tabelle. **Eine Messung wird nicht dadurch ungültig, dass man sich
>   anders entscheidet** — sie ist die erste Stelle, an der man nachsieht,
>   wenn Stufe 4 nicht trägt.

**Sortierung:** von oben nach unten — was zu tun ist, steht oben; was erledigt
ist, wandert nach unten in „Erledigt".

Der Plan ist eine **Leiter mit Toren**, keine gerade Linie. **Ein rotes Tor
ändert den Plan, nicht nur den Haken.**

---

## ▶ Das Nächste: Abschnitt 4 nachholen, dann Stufe 2

**Der zweite Lauf ist am 22.09. gefahren** (`10_bilanz.txt`, sieben
Konstant-Treiber-Trainings-OPs, drei Flusslevel). Abschnitt 1 bis 3 sind
durch, **Abschnitt 4 ist abgestürzt** — also genau `U(V̇)`, die Frage, für die
der Lauf gemacht wurde. Wieder das Werkzeug, wieder nicht die Daten; der Fehler
ist repariert (siehe unten). Es fehlt nur der Wiederholungslauf:

```bash
cd ~/llmtraining            # nicht der Windows-Pfad; python3 aus modulus_env
git checkout main && git pull
python3 GridCNN/tools/balance_check.py 2>&1 | tee 11_bilanz.txt
```

Abschnitt 1–3 kommen dabei unverändert wieder — nachgerechnet, der reparierte
Pfad gibt in Abschnitt 2 denselben Wert wie der alte. Neu ist Abschnitt 4 und
eine Spalte `T_fluid`, die mitschreibt, **welche** Fluidtemperatur in `dT`
steckt.

> ### Was der Lauf vom 22.09. ergeben hat
>
> | Abschnitt | | |
> |---|---|---|
> | 2 — Fluidbilanz | **1.030 … 1.062** | 🟢 **`ghost_hi` steht.** Kein Halbmodell-Faktor auf dem Fluidpfad. `T_in` kam aus `Input Signale.csv`, wie gebaut |
> | 1 — `Q_ht/tot` | **0.700 … 0.772** mit Fluss | 🔴 **Die Hypothese aus 1a ist widerlegt** — der Monitor draint **nicht** die ganze Erzeugung |
> | 1 — `Q_ht/jr1` | **2.286 … 2.566** | 🟡 nahe „beide Platten", aber **nicht konstant** — siehe unten |
> | 1 — `tot/jr1` | **2.99 … 3.44** | die Zelle erzeugt **gut das Dreifache** einer Rolle |
> | 4 — `U(V̇)` | **abgestürzt** | offen, der Wiederholungslauf holt es |
>
> **Der Befund, der den Plan ändert:** `Q_ht/jr1` **hängt vom Flusslevel ab** —
> 2.286 / 2.302 / 2.304 bei V̇ = 0.0013 gegen 2.488 / 2.566 bei 0.0026, und
> 0.334 / 0.336 ohne Fluss. **Ein Konventionsfaktor kann das nicht.** Eine
> Bezugsfläche ist Geometrie; sie weiß nicht, wie schnell das Kühlmittel
> fließt. Das Verhältnis ist also **keine Konstante, die man wegdividiert**,
> sondern zu einem Teil Physik: mehr Kühlung ⇒ mehr abgeführter Anteil.
>
> Dasselbe an `Q_ht/tot`: 0.11 ohne Fluss → 0.72 → 0.76. Es **sättigt gegen
> ~0.8, nicht gegen 1**. Im späten Fenster wird also immer noch gespeichert;
> die Zelle ist nicht quasistationär, und der Wandmonitor kann die Erzeugung
> gar nicht ausgleichen.
>
> **Was daraus für die Quelle folgt** — und das bleibt stehen, unabhängig vom
> Faktor: bei `jr2/jr1 = 1.000` und `tot/jr1 ≈ 3.2` liegen rund
> **ein Drittel bis 40 % der Gesamterzeugung außerhalb der beiden Wickel**
> (Ableiter, Stromschienen, Kontaktwiderstände). Die Modellquelle `q_dot` deckt
> nur JR1 ab. Das ist **O17 für `PINNmodulusTwo`** und es ist jetzt belegt,
> nicht vermutet. Damit `energy_balance_report` es je messen kann, muss
> `total_w` in **Stufe 2** mit in den Cache.
>
> ⚠ **Das Tor „Q halbieren oder A verdoppeln" greift so nicht mehr.** Es setzt
> voraus, dass `Q_ht/jr1` eine Konstante ist. Ist es nicht. Die Trennung
> zwischen Konvention und Physik hängt jetzt an Abschnitt 4: **liegt `U` über
> die drei Flusslevel auf einer Kurve, ist der flussabhängige Teil erklärt**,
> und was dann an konstantem Faktor übrig bleibt, ist die Konvention.

### Was am Werkzeug diesmal kaputt war

`np.interp(tw, r["t"], r["t_in"])` → `fp and xp are not of the same length`.
Das Skript liest fünf CSVs je OP und hat die Zeitachse von **zwei** mitgelesen.
`t_out` (aus `*_Temperaturen.csv`) und `t_in` (aus `*_Input Signale.csv`)
wurden behandelt, als lägen sie auf der Achse von `*_Heat Source.csv`. `t_in`
ist dort ein **Skalar** — die Datei führt Sollwerte und hat gar keine
Zeitspalte.

> **In Abschnitt 2 ist genau das nicht aufgefallen**, weil dort `t_out - t_in`
> steht und numpy Länge 1 still auf Länge N broadcastet. Die Zahl war richtig,
> aber aus dem falschen Grund — bei einer *nicht* konstanten Reihe wäre sie
> still falsch gewesen. **Dieselbe Klasse wie `xyz = raw[0]` in `data.py`:**
> eine Zusage, die stimmt, die aber niemand prüft.
>
> Repariert: `auf_achse()` verlangt, dass man die Achse einer Reihe nennt;
> ein Skalar trägt keine und gilt überall; eine Reihe **ohne** Achse wird
> abgelehnt statt geraten. `als_konstante()` prüft den Rückfallwert und bricht
> ab, wenn er sich doch ändert.

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

## Die Leiter

| Stufe | was | Dauer | Tor |
|---|---|---|---|
| **0** | Rangtest | ✅ **erledigt — ROT, überstimmt** | 4 Moden. Folge: 16 × 3 statt 64 × 4 |
| **1** | Bilanz-Gegenprobe | 🟡 **teilweise**, zweiter Lauf offen | geht die Wärmebilanz auf? |
| **2** | vier Größen in den Cache | 30 min | Reports weiter grün? |
| **3** | **Physik ohne Netz** — der explizite Löser | ✅ **gebaut**, läuft adiabat | schlägt reine Physik die trivialen Vorhersager? |
| **4** | **der CNN**, Ein-Schritt-Training | ✅ **gebaut**, Ladepfad offen | schlägt er Stufe 3? |
| **5** | truncated BPTT | Tage | fällt der Spätfehler (O13)? |
| **6** | Vergleich gegen PINNmodulusTwo | 1 Lauf | derselbe Split, dieselben Metriken |

> **Stufe 3 und 4 sind gebaut, aber nicht gemessen.** Beide hängen an
> derselben Sache: `solve.py` läuft adiabat und `train.py` hat keinen
> Ladepfad, weil `U(V̇)` nicht kalibrierbar ist und der `data_cache` die vier
> Größen aus Stufe 2 nicht enthält. **Stufe 2 ist der Engpass, nicht die
> Bauzeit.**

> **Was im Repo steht und was fehlt**, steht unten unter „Der Code"; das
> Protokoll der Läufe in [`BENCHMARK.md`](BENCHMARK.md).

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

| Ergebnis | Folge | gemessen 22.09. |
|---|---|---|
| `Q_ht/tot ≈ 1` | 🟢 kein Konventionsfehler. Stattdessen **O17**: die Modellquelle ist unvollständig | **nein** — 0.700 … 0.772 |
| `Q_ht/jr1 ≈ 2`, `Q_ht/tot ≉ 1` | 🟡 beide Platten. **Entweder** `Q` halbieren **oder** `A` verdoppeln — nie beides | **2.286 … 2.566, flussabhängig** — siehe 1d |
| Fluidbilanz-Verhältnis ≈ 1.0 | 🟢 `ghost_hi` steht | ✅ **1.030 … 1.062** |
| `U` über drei Flusslevel auf einer Kurve | 🟢 `U(V̇)` wird feste Funktion, null freie Parameter | *offen* — Abschnitt 4 abgestürzt |
| ~~bei ṁ = 0 fließt Energie ab~~ | ~~🔴~~ **gestrichen** — war falsch, siehe 1b | — |

## 1d. `Q_ht/jr1` ist keine Konstante — und damit ist das Tor falsch gestellt

Das war am 09.09. nicht sichtbar, weil beide Fluss-OPs dasselbe Level fuhren.
Mit drei Leveln steht es da:

| V̇ | OPs | `Q_ht/jr1` | `Q_ht/tot` |
|---|---|---|---|
| 0 | OP07, OP14 | 0.334 / 0.336 | 0.109 / 0.112 |
| 0.0013 | OP01–OP03 | 2.286 / 2.302 / 2.304 | 0.709 / 0.739 / 0.700 |
| 0.0026 | OP04, OP05 | 2.488 / 2.566 | 0.772 / 0.746 |

**Eine Bezugsfläche weiß nicht, wie schnell das Kühlmittel fließt.** Ein
Konventionsfaktor ist Geometrie und damit konstant; dieses Verhältnis ist es
nicht. Also steckt Physik darin, und „Q halbieren oder A verdoppeln" —
beides *konstante* Eingriffe — kann nicht die ganze Antwort sein.

`Q_ht/tot` sättigt sichtbar gegen **~0.8, nicht gegen 1**: auch im späten
Fenster wird noch gespeichert. Das Fenster ist quasistationär genug für einen
Vergleich, aber die Zelle ist es nicht — der Wandmonitor *kann* die Erzeugung
dort gar nicht ausgleichen.

**Was trotzdem feststeht, unabhängig vom Faktor:** `jr2/jr1 = 1.000` und
`tot/jr1 = 2.99 … 3.44`. Rund **ein Drittel bis 40 % der Gesamterzeugung liegt
außerhalb der beiden Wickel** — und die Modellquelle `q_dot` deckt nur JR1 ab.
Das ist **O17** für `PINNmodulusTwo`, jetzt belegt statt vermutet. Messbar wird
es erst, wenn `total_w` in **Stufe 2** mit in den Cache geht.

**Die Trennung hängt jetzt an Abschnitt 4.** Liegt `U` über die drei
Flusslevel auf einer Kurve, ist der flussabhängige Teil erklärt; was dann an
konstantem Faktor übrig bleibt, ist die Konvention. Vorher ist jede Korrektur
an `Q` oder `A` geraten.

**Blockiert nichts mehr:** `data_raw/` liegt vor.

---

# Stufe 2 — Die vier Größen in den Cache

Unverändert: `q_solid_to_fluid`, `fluid_out_temp`, `cp_fluid`, `mdot` in
`generate_cache.py` und `opbundle_contract.md`, `schema_version` hoch, alle
siebzehn OPs neu bauen (10–30 min).

**Dazu neu:** `total_w` gehört mit hinein, falls 1a sich bestätigt — sonst ist
O17 nie messbar.

⚠ **Es sind siebzehn OPs, nicht sechzehn.** Im Cache liegen OP01–OP16 **plus
OP19** (der Report-only-OP aus O11). Wer sechzehn neu baut, lässt OP19 auf dem
alten `schema_version` zurück — und merkt es erst, wenn Stufe 6 ihn mitrollen
will. Gemessen am 22.09.

⚠ **Der Cache-Umbau macht die Gittergleichheit wieder ungeprüft.** Am 22.09. ist
gemessen, dass `xyz` über alle siebzehn OPs bitgleich ist; `load_ops` nimmt sie
aber weiterhin unbesehen aus `raw[0]`. Nach dem Rebuild gehört der Vergleich
als Prüfung in `load_ops` — drei Zeilen, und sie fangen genau den Fehler, den
`grid.derive_layout` nicht sehen kann.

**Tor:** `profile_report`, `coverage_report` und `energy_balance_report` müssen
**exakt dieselben Zahlen** liefern wie vorher.

---

# Stufe 3 — Physik ohne Netz  ✅ gebaut

**Gebaut:** [`solve.py`](solve.py), [`physics.py`](physics.py),
[`grid.py`](grid.py) — ein klassischer expliziter Euler auf dem 3 × 11 × 11:

```
T_{t+1} = T_t + dt * ( Fo : grad² T + Qsrc )
```

Kein gelerntes Gewicht. Der Wert liegt in vier Dingen, und alle vier fielen
weg, wenn man diese Stufe überspringt und gleich das Netz baut:

* **Es prüft Padding, Stencil und Wandterm unabhängig vom Lernen.** Ist der
  Kreuzterm falsch oder das Padding verdreht, sieht man es hier — und nicht
  drei Wochen später als „das Netz konvergiert schlecht".
* **Es liefert die Physik-Latte.** Verglichen wird sonst gegen `persistence`
  und `train-mean`; beide sind trivial. „Reine Wärmeleitung mit kalibrierter
  Kühlwand" ist eine *ernsthafte* Latte.
* **Es beantwortet die CFL-Frage empirisch.** Läuft der Löser bei
  `subsample_time: 2` (dt = 0.2 s gegen Δt_max 0.241 s) stabil?
* **Es ist der Rest-Definitionspunkt.** Was der Löser *nicht* trifft, ist genau
  das, was `g_θ` lernen muss. Damit ist die Aufgabe des Netzes definiert statt
  geraten — und weil `model.py` denselben Operator benutzt, ist es dieselbe
  Zahl und nicht eine ähnliche.

### ⚠ Er läuft heute adiabat, und das ist keine Latte

Der Wandterm ist gebaut und getestet, aber ohne `U(V̇)` nicht kalibrierbar.
`solve.rollout` trägt das als Notiz mit und `SolveResult.summary()` schreibt
„Wandterm AUS (adiabat)". **Das ist eine Ablation und darf nicht als Latte
zitiert werden.** Blockiert durch Stufe 2.

### Was der Rangtest an dieser Stufe geändert hat

Nichts. Sie ist Physik, nicht Architektur — das galt schon, als der ROM der
Plan war, und es gilt jetzt genauso.

### Das Tor

| Ergebnis | Folge |
|---|---|
| schlägt `train-mean` auf den ausgehaltenen OPs | 🟢 der Unterbau stimmt, weiter zu Stufe 4 |
| stabil, aber schlechter als `train-mean` | 🟡 normal — der Löser kennt die Materialdaten nur genähert. Weiter, aber der Rest ist groß |
| divergiert | 🔴 entweder CFL (dann `subsample_time: 1`) oder ein Vorzeichenfehler im Padding. **Nicht mit dem Netz übertünchen** |
| schlägt schon `PINNmodulusTwo` (6.270 / 3.585 °C) | 🔴🟢 dann lief das Lernen bisher gegen fehlende Physik, und der Plan wird ein anderer |

---

# Stufe 4 — Der CNN  ✅ gebaut, nicht gemessen

**Gebaut:** [`model.py`](model.py), [`train.py`](train.py), 46 Tests.

```
f  =  L(T_t)  +  Qsrc_t  +  g_θ(X_t)
      ^feste Physik         ^gelernte Korrektur, 11 427 Parameter
```

`L` **ist** `physics.anisotropic_laplacian` — derselbe Operator wie in Stufe 3,
keine zweite Fassung davon.

| | |
|---|---|
| Eingang `X_t` | **44 Kanäle** à 11 × 11: 9 Zustand/Historie + 17 statische Karten + 18 gebroadcastete Treiber |
| Ausgang | 3 × 11 × 11 — eine Änderungsrate je x-Ebene |
| Stapel | 44 → 16 → 16 → 16 → 3, alles 3×3, `reflect`-gepaddet |
| Verlust | `L = w_data·L_data + w_phys·L_phys + w_wall·L_wall` — **kein `L_bc`** |

### Die zwei Rollen der x-Achse

Der Punkt, an dem der Entwurf am leichtesten misszuverstehen ist:

* **im Physik-Term `L`** ist x eine **echte Achse** mit Geisterschichten —
  Symmetrie bei x = 0, Kühlwand bei x = 0.0219, nicht-äquidistanter
  Dreipunktstern dazwischen.
* **in der Korrektur `g_θ`** ist x in die **Kanäle** gefaltet. Drei Ebenen sind
  zu wenig, um darüber zu falten.

Kein Widerspruch: der Stencil braucht die Nachbarn in x, der Faltungskern nicht.

### Zwei Entscheidungen, die nicht kosmetisch sind

**`head` startet auf exakt null.** Damit ist `g_θ` beim ersten Schritt null und
das Modell rechnet **Bit für Bit** den Löser aus Stufe 3. Das löst den Satz
*„f korrigiert den Löser, es ersetzt ihn nicht"* ein, statt ihn zu behaupten —
`test_bei_init_rechnet_das_modell_exakt_die_physik` prüft es ohne Toleranz.

**Jede Faltung padded `reflect`, nicht `zeros`.** Mit dem Default sähe der Kern
am Rand eine erfundene Null, im z-Score also eine Temperatur von `T_mu` — eine
Randbedingung, die niemand beschlossen hat.

### Die Ablation steht als Erstes an

Drei Konfigurationen, die sich in genau einer Sache unterscheiden:

| | die Rate `f` | Eingang | `w_phys` | CLI | misst |
|---|---|---|---|---|---|
| **A** | `g_θ` allein | 44 Kanäle | 0 | `--no-physics` | reine Blackbox — die Latte |
| **B** | `L+Q+g_θ` | 44 Kanäle | 0 | (Vorgabe) | was die Physik **in der Architektur** bringt |
| **C** | `L+Q+g_θ` | 44 Kanäle | > 0 | `--w-phys 0.1` | was der Strafterm **obendrauf** bringt |
| **D** | `L+Q+g_θ` | **42** — ohne `y_map`/`z_map` | 0 | `--no-coord-maps` | ob der Ortsprior aus §2b **echt** ist |

A → B ändert die Architektur bei gleichem Verlust, B → C den Verlust bei
gleicher Architektur. Jede Differenz ist einem einzigen Eingriff zuzuordnen.

> ### Arm D — neu am 22.09., ✅ gebaut
>
> **Die Frage:** README §2b führt als Vorteil des CNN an, dass er Position
> *nicht* auswendig lernen **kann** und räumliche Struktur deshalb über die
> Materialkarten begründen **muss**. [`model.py:193`](model.py) gibt ihm die
> Position trotzdem — zwei z-gescorte Koordinatenkarten, mit dem Kommentar,
> ohne sie wäre *„jede Randzelle von jeder Mittelzelle ununterscheidbar"*.
> Beide Texte gehen von derselben Prämisse aus und ziehen den entgegengesetzten
> Schluss. **A/B/C berühren das nicht — alle drei tragen die Karten.**
>
> **Was D ändert:** genau zwei Kanäle von 44. 11 139 Parameter statt 11 427
> (2 × 16 × 9 = 288 weniger in der ersten Faltung). Sonst identisch — ein Test
> hält fest, dass die fünfzehn Materialkarten **bitgleich** bleiben.
>
> | Ergebnis | Folge |
> |---|---|
> | **D ≈ B** | 🟢 der Ortsprior ist echt, §2b stimmt, und der CNN steht dort wirklich anders als das MLP |
> | **D deutlich schlechter** | 🔴 der CNN hat dieselbe Positionskrücke wie das MLP. §2b beschreibt dann einen Entwurf, den der Code nicht umsetzt — und §11.1 ist beantwortet, nur anders als erhofft |
>
> ⚠ **Und D ist ein enger Test.** Am 22.09. gemessen: `region`/`rho`/`Cp` sind
> je x-Ebene konstant, `lam` variiert nur in **zwei Zeilen am unteren y-Rand**
> (22 von 121 Punkten), und der Kontrast in der Ebene ist nach dem z-Score rund
> **zwölfmal schwächer** als der zwischen den Ebenen. D fragt also, ob dieses
> schmale Band reicht. Fällt D durch, ist damit **nicht** gezeigt, dass ein
> Ortsprior unmöglich wäre — nur, dass dieser Datensatz ihn nicht hergibt.

> **Beide starten auf etwas Sinnvollem.** Weil `head` null ist, startet **A**
> bei `persistence` (Rate null) und **B** bei der Physik. Keine der beiden
> startet bei Rauschen, und das hält den Vergleich fair.

⚠ Die Seed-Streuung liegt bei 0.518 / 0.882 °C, auf OP06 bis 1.63 °C. Ein
Unterschied unter ~1 °C ist mit drei Seeds **nicht lesbar** — jede
Konfiguration braucht eine Seed-Schleife. `train.py` meckert bei `--seeds < 3`.

### Die Größenachse

| | Parameter | |
|---|---|---|
| `--width 16 --blocks 3` | **11 427** | Vorgabe, nach Tor 0 |
| `--width 24 --blocks 3` | 20 595 | Gegenprobe |
| `--width 64 --blocks 4` | **137 923** | der Entwurf aus der Präsentation |

Die Achse prüft, ob größer überhaupt etwas bringt. Bringt sie nichts, ist das
kein Nullergebnis, sondern eine Bestätigung des Rangtests.

### Das Tor

Schlägt Stufe 3 auf den fünf ausgehaltenen OPs. Wenn nicht, hat das Netz nichts
beigetragen — und dann ist der Rangtest die erste Stelle, an der man nachsieht.

---

# Stufe 5 — Truncated BPTT

Der Hebel auf **O13** (OP06: 6.270 °C im Mittel, 13.248 °C spät). Fenster von
`k` Schritten, Gradient durch alle `k`, `detach` am Fensterende.

`k` = 50 zuerst, dann 200. Aktivierungsspeicher ist grob 31k Floats je Schritt
(README §4), also ist 200 machbar.

> ### ⚠ Warum das die *einzige* Änderung ist, die O13 erreicht
>
> `train.py` rollt je Epoche und OP einmal frei aus, friert die Trajektorie ein
> und trainiert darauf Ein-Schritt-Paare. Der Gradient überquert die Historie
> also **nie** — er sieht genau einen Schritt. Damit kann kein noch so langes
> Training den Spätfehler strukturell erreichen: er entsteht aus der
> Akkumulation über hunderte Schritte, und über die wird nicht optimiert.
>
> `test_der_gradient_ueberquert_die_historie_nicht` hält das fest, am Verhalten
> statt an der Absicht. Wer diese Schleife anfasst, muss ihn brechen sehen,
> bevor er Stufe 5 unmöglich macht.

**Tor:** `late_mae` fällt, bei mindestens gleichem `mae`. Fällt nur `mae` und
`late_mae` nicht, hat BPTT nicht getan, wofür es da ist.

---

# Stufe 6 — Der Vergleich

Derselbe Split (train 11 / val OP06+OP09 / test OP13, OP15, OP16), dieselben
`op_metrics`, dieselben trivialen Vorhersager, **plus** die Physik-Latte aus
Stufe 3. Über mehrere Seeds — *ein
Seed ist keine Streuung.*

OP19 wird mitgerollt: berichten, nie trainieren, nie selektieren (O11).

---

## Was übernommen wird — importiert, nicht kopiert

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "PINNmodulusTwo"))
import data, op_registry, op_metrics
```

**Kein Modulus.** GridCNN braucht `FCLayer` nicht; reines PyTorch. Läuft
trotzdem in `modulus_env`, weil Torch dort schon liegt.

---

## Was NICHT gebaut wird

Damit der Umfang nicht wandert:

* **ConvGRU / ConvLSTM.** Erst wenn Stufe 5 steht. Ein unbeschränkter
  versteckter Zustand bei elf Trajektorien ist ein echtes Risiko.
* **FiLM-Konditionierung.** Die Treiber werden erst gebroadcastet. FiLM ist eine
  eigene Sweep-Achse, keine Architekturentscheidung.
* **FNO / DeepONet / Neural ODE.** Bei 11 × 11 Overkill.
* **Das ROM.** Am 15.09. gestrichen — der CNN ist die Entscheidung. Der
  Rangtest, der dafür gesprochen hätte, bleibt gemessen und in der
  Stand-Tabelle stehen; er ist die erste Stelle, an der man nachsieht, wenn
  Stufe 4 nicht trägt.
* **Plots, Resume, Checkpoint-Merge.** Zuletzt, und nur für Achsen, die
  wirklich Stunden laufen.

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
| **09.09.** | **Stufe 0 gelaufen — ROT.** 4 Moden bei 99.9 %. Folge: der Stapel wird auf 16 × 3 verkleinert |
| **09.09.** | **Stufe 1, erster Lauf.** `jr2/jr1 = 1`, `U`-Verhältnis 1130 : 50 plausibel. Mein ṁ = 0-Torkriterium war falsch und ist gestrichen |
| **09.09.** | Werkzeug korrigiert: drei Flusslevel statt einem, `T_in`-Rückfallkette, `Q_ht/tot` |
| **14.09.** | **Der Gitter-Unterbau steht.** `grid.py`, `physics.py`, `solve.py`, `benchmark.py`, 45 Tests, CI grün |
| **14.09.** | Stencil-Ordnungen gemessen, beide Randzusagen **exakt** null, Route R6 beziffert |
| **15.09.** | **Entscheidung: CNN statt ROM**, 16 × 3 = 11 427 Parameter. Der ROM-Plan ist gestrichen |
| **15.09.** | `model.py` und `train.py` gebaut, 46 Tests. Bei Init rechnet das Modell **exakt** den Löser |
| **15.09.** | Zwei Funde aus Tests: 64 × 4 sind **137 923** statt „~100 k"; ein konstanter Materialkanal wurde zu ±1 statt 0 |
| **22.09.** | **Gittergleichheit gemessen statt angenommen.** `xyz` ist über alle **siebzehn** Cache-OPs bitgleich, gleiche Reihenfolge, `layer` ebenso — `data.py:653` nahm das bisher unbesehen aus `raw[0]` |
| **22.09.** | **Die dokumentierte Spannweite war falsch.** 0.198089 × 0.104441 an sieben Stellen; gemessen sind 0.198094368 × 0.104431991 m. Cache und Legacy-CSVs stimmen exakt überein |
| **22.09.** | **Der Geometrie-Test war zirkulär** und hätte gegen die echten Koordinaten in allen vier Vergleichen gefallen. `test_layout_aus_den_echten_koordinaten` schließt den Kreis |
| **22.09.** | **Materialverteilung gemessen:** `region`/`rho`/`Cp` je x-Ebene konstant; `lam` variiert, aber nur in zwei Zeilen am unteren y-Rand (22 von 121 Punkten) |
| **22.09.** | **§2b widerspricht `model.py:193`** — die zwei Koordinatenkarten geben dem Kern die Position, die §2b ihm abspricht. Offen, Route R8 |
| **22.09.** | **`main` war seit dem 17.09. rot** und niemand hat es bemerkt: `pytest (GridCNN)` meldete auf `163b21c` `2 failed, 97 passed`, und der Benchmark-Schritt dahinter wurde seither übersprungen. PR #40 erbt dieses Rot — seine vier Dateien liegen alle in `PINNmodulusTwo/`, dessen Suite grün ist |
| **22.09.** | **Ablationsarm D gebaut** — `--no-coord-maps`, 42 statt 44 Kanäle, 11 139 Parameter. Sieben Tests, darunter einer, der die fünfzehn Materialkarten bitgleich festhält |
| **22.09.** | **Die Gittergleichheit ist jetzt geprüft statt angenommen**: `data._assert_shared_geometry` fällt, wenn ein OP eine andere Punktreihenfolge hat. Test in beide Richtungen, mit reiner Zeilenvertauschung |
| **22.09.** | PR #40 (Achse 1, O18, MPS-Prüfung) in diesen Branch hereingeholt — ein PR statt zwei |
| **22.09.** | **Stufe 1, zweiter Lauf.** Fluidbilanz **1.030 … 1.062** → `ghost_hi` steht. `Q_ht/tot = 0.700 … 0.772`, **nicht ≈ 1** — die Hypothese aus 1a ist widerlegt. `tot/jr1 = 2.99 … 3.44` belegt **O17** |
| **22.09.** | **`Q_ht/jr1` ist keine Konstante** — 2.29 bei V̇ = 0.0013 gegen 2.53 bei 0.0026. Eine Bezugsfläche kann nicht flussabhängig sein, also ist das Tor „Q halbieren oder A verdoppeln" falsch gestellt. Siehe 1d |
| **22.09.** | **`balance_check.py` Abschnitt 4 abgestürzt und repariert:** das Skript las die Zeitachse von zwei der fünf CSVs und legte den Rest stillschweigend darauf. In Abschnitt 2 hat numpy das still gebroadcastet — richtige Zahl, falscher Grund |
| **22.09.** | Beide fallenden Tests repariert: die float32-Schranke im gepaddeten Rollout (1.2e-6 gegen 1e-6, hielt zufällig) und **`torch.equal` im float64-Vergleich** — Bitgleichheit zwischen zwei Batchgrößen ist durch nichts garantiert und war nicht portabel (lokal grün, auf dem Runner rot). Jetzt beide gegen eine relative Schranke `1e-12` |

## Stand

| Stufe | Kriterium | gemessen | Datum |
|---|---|---|---|
| 0 | Gitterprobe 3×11×11 | **bestätigt** | 09.09. |
| 0 | **gepoolte Ortsstruktur** (90/99/99.9/99.99 %) | **1 / 2 / 4 / 6 Moden** | **09.09.** |
| 0 | Tor | 🔴 **≤ 5 Moden** — überstimmt, Folge ist 16 × 3 | 09.09. |
| 1 | `jr2/jr1` | **1.000** — zwei gleiche Wickel | 09.09. |
| 1 | `Q_ht/jr1` spät, mit Fluss | **2.286 … 2.566** — und **flussabhängig**, also kein Konventionsfaktor. Siehe 1d | **22.09.** |
| 1 | `Q_ht/tot` | **0.700 … 0.772** mit Fluss, 0.11 ohne — **nicht ≈ 1**. Sättigt gegen ~0.8: es wird noch gespeichert | **22.09.** |
| 1 | `tot/jr1` | **2.99 … 3.44** — ⅓ bis 40 % der Erzeugung liegt außerhalb der Wickel. **O17 belegt** | **22.09.** |
| 1 | Fluidbilanz-Verhältnis | ✅ **1.030 … 1.062** — 🟢 `ghost_hi` steht. `T_in` aus `Input Signale.csv` | **22.09.** |
| 1 | Wandanteil bei ṁ = 0 | **≈ 0.27** — physikalisch richtig (Fluidkapazität), Kriterium war falsch | 09.09. |
| 1 | `U` mit Fluss / ohne | **~1130 / ~50 W/m²K**, Faktor ~23 — vorläufig, Faktor 2 aus 1a offen | 09.09. |
| 1 | `U(V̇)` auf einer Kurve? | *offen* — Abschnitt 4 stürzte am 22.09. ab (Zeitachsen), Werkzeug repariert, Lauf steht aus | |
| 2 | Reports unverändert | | |
| — | Reshape aus Koordinaten ableitbar und umkehrbar | **ja**, 0.198094368 × 0.104431991 m | 14.09. / korrigiert 22.09. |
| — | `xyz` über alle siebzehn Cache-OPs identisch | **ja, bitgleich**, gleiche Reihenfolge | 22.09. |
| — | Cache-Koordinaten = Legacy-CSVs | **ja, exakt** | 22.09. |
| — | Äquidistanz in y/z im Cache | **bis float32**: 4e-7 relativ gegen `_uniform`-Toleranz 1e-6, Reserve 3.7× | 22.09. |
| — | `region`/`rho`/`Cp` in der Ebene | **konstant** je x-Ebene | 22.09. |
| — | `lam` in der Ebene | **variabel**, aber nur zwei Zeilen am unteren y-Rand; 99/121 Punkte ein Wert | 22.09. |
| — | Kontrast in der Ebene gegen zwischen den Ebenen (nach z-Score) | **Faktor ~12 schwächer** | 22.09. |
| — | Kanäle, die in der Ebene variieren | **17 von 44** (9 Zustand + 6 `lam` + 2 Koordinaten) | 22.09. |
| — | davon strukturell flache Gewichte in Schicht 1 | **3 456 von 11 427 = 30 %**, effektiv ≈ 7 971 | 22.09. |
| — | `dT/dx` an der Symmetrieebene | **exakt `0.0`**, ohne Toleranz | 14.09. |
| — | zentrale Differenz am y/z-Rand | **exakt `0.0`** | 14.09. |
| — | Stencil-Ordnung d²/dy², d²/dz² | **1.98 / 1.98** | 14.09. |
| — | Stencil-Ordnung d²/dx² (gestreckt) | **1.28** — richtig für den nicht-äquidistanten Dreipunktstern | 14.09. |
| — | Bilanztreue des Sterns | **3.6 % Drift, sättigt** (R6) | 14.09. |
| 3 | Löser stabil bei dt = 0.2 s | | |
| 3 | Physik-Latte, val OP06 / OP09 | | |
| 4 | CNN schlägt Physik-Latte | | |
| 5 | `late_mae` gefallen | | |
| 6 | val / test gegen PINNmodulusTwo | | |

---

# Der Code — Stand 15.09.

> Dieser Abschnitt beschreibt, **was im Repo liegt**. Der Plan oben sagt, was
> gemessen werden soll; hier steht, womit. Das fortlaufende Protokoll der Läufe
> steht in [`BENCHMARK.md`](BENCHMARK.md), zusammen mit den offenen Routen.

## Was steht

| Datei | was sie tut | geprüft durch |
|---|---|---|
| [`grid.py`](grid.py) | Reshape 363 → (3,11,11), **aus den Koordinaten abgeleitet**, plus die drei Paddings | `tests/test_grid.py`, 12 Tests |
| [`physics.py`](physics.py) | nicht-äquidistanter x-Stern, Kreuzterm `λ_xy`, Quelle, Wandterm, `U(V̇)` | `tests/test_physics.py`, 19 Tests |
| [`solve.py`](solve.py) | explizites Euler, entdimensioniert wie `data.py` | `tests/test_solve.py`, 14 Tests |
| [`model.py`](model.py) | **der Faltungsstapel in Δ-Form**, 11 427 Parameter | `tests/test_model.py`, 23 Tests |
| [`train.py`](train.py) | **die Trainingsschleife**, Ein-Schritt gegen eine eingefrorene Trajektorie | `tests/test_train.py`, 23 Tests |
| [`benchmark.py`](benchmark.py) | Stufenläufer, schreibt ins lebende Dokument | CI, Stufe 0 und 2 |

**91 Tests, Sekunden, ohne Modulus und ohne `data_cache`.** Die CI fährt
`GridCNN` und `PINNmodulusTwo` in **getrennten** Invokationen — beide Projekte
haben ein Modul namens `physics`, und in einem gemeinsamen Lauf gewänne das
erste. Das ist nachgeprüft und nicht nur vermutet: ein gemeinsamer Lauf bricht
beim Import ab.

### Die vier Zusagen, die bewiesen sind statt behauptet

1. Der Reshape wird **abgeleitet**, nicht geraten, und trifft die **gemessene**
   Geometrie: Spannweite 0.198094368 × 0.104431991 m, `dx = 10.785542 / 11.114458 mm`.

   > ⚠ **Diese Zusage war bis zum 22.09. keine.** Der Test, der sie prüfte, las
   > dieselben Konstanten, aus denen `conftest.py` sein Testgitter baut — er
   > konnte nicht fallen. Und die Konstanten waren falsch: gegen die echten
   > Koordinaten wären alle vier Vergleiche durchgefallen (`dx` um 4.6e-7, die
   > Spannweiten um 5.4e-6 bzw. 9.0e-6). Seit dem 22.09. leitet
   > `test_layout_aus_den_echten_koordinaten` das Gitter direkt aus den drei
   > CSVs in `legacy/.../coordinates/` ab — den Dateien, die im Repo liegen und
   > mit dem Cache aller siebzehn OPs bitgleich sind. **Erst dieser Test macht
   > die Zusage falsifizierbar.**
2. `dT/dx` an der Symmetrieebene ist **exakt `0.0`** — ohne Toleranz geprüft,
   über Feldskalen von 1e-8 bis 1e8. Im Zähler steht `T₁ − T₁`.
3. Die zentrale Differenz am y/z-Rand ist **exakt `0.0`**. `reflect`, nicht
   `replicate`.
4. **Bei Initialisierung rechnet `model.py` Bit für Bit `solve.py`.** `head` ist
   null initialisiert, also ist `g_θ` null. Das ist die Einlösung von
   *„f korrigiert den Löser, es ersetzt ihn nicht"*.

### Drei Befunde, die beim Bauen abgefallen sind

**Der Stern ist nicht bilanztreu** (Route R6). Rollt man adiabat aus, driftet
das Mittel um **3.6 % der Feldstreuung** und **sättigt** dann; das Feld läuft
sauber gegen eine Konstante (`std` → 4e-5). Ursache sind zwei bewusste
Entscheidungen: die nicht-konservative Form `Fo : ∇²T` und knotenzentriertes
`reflect` ohne Halbzellgewichte. Dass die Drift sättigt, ist der Beleg, dass
kein Rand *echte* Energie leckt — sie gehört aber **neben** die Physik-Latte
geschrieben, nicht darunter versteckt.

**64 × 4 sind 137 923 Parameter, nicht „~100 k."** Siehe den Kasten ganz oben.

**Ein konstanter Materialkanal wurde zu ±1 statt zu 0.** In float32 ist
`ρ·Cp = 2.25e6` überall gleich, der Mittelwert trägt aber einen Rundungsfehler
von O(0.25) — und dieser Rest geteilt durch eine genauso winzige Streuung ergibt
Rauschen mit Standardabweichung 1. Das Netz hätte es nicht von echter Struktur
unterscheiden können. Jetzt wird in float64 gescort, der konstante Fall
abgefangen, und tote Kanäle werden **gemeldet** statt still zu bleiben.

## Was fehlt

| | was | blockiert durch |
|---|---|---|
| **Ladepfad** | `train.py` an `PINNmodulusTwo/data.py` anschließen: Bündel laden, `OPTensors` füllen, `op_metrics` auswerten | `data_cache` (liegt nicht im Repo) |

> **Was der Ladepfad schon vorfindet** (22.09. gebaut, mit Tests):
> `train.modell_kwargs(args)` übersetzt die Ablationsflags an **einer** Stelle
> in die Argumente von `build_static_maps` und `GridCNN` — er muss sie nur noch
> aufrufen. Und `derive_layout` gehört mit `bundle.xn` (float64) × `L_ref`
> gefüttert, **nicht** mit `op.xn`: der ist float32 und halbiert die Reserve der
> Äquidistanzprüfung (4e-7 gemessen gegen `rtol` 1e-6).
| **Wandterm benutzbar** | `U(V̇)` kalibrieren | **Stufe 2**: `q_solid_to_fluid`, `mdot`, `cp_fluid`, `fluid_out_temp` fehlen im Bündel |
| **Physik-Latte** | Stufe 3 mit echter Wand | Wandterm |
| **`L_wall`** | der einzige Verlustterm mit gemessenem Ziel | Wandterm |
| **`C_fluid`** | Wärmekapazität des Kühlmittels im Kanal, für den Kapazitätsmodus | liegt nicht vor — **nicht raten** |

⚠ **Der Löser und das Training laufen heute adiabat.** Der Wandterm ist gebaut
und getestet, aber ohne die vier Cache-Größen nicht kalibrierbar.
`solve.rollout` trägt das als Notiz mit, `SolveResult.summary()` schreibt
„Wandterm AUS (adiabat)", und `train._wall_ghost` **wirft**, statt ein `U` zu
raten. Das ist eine **Ablation, keine Latte**, und darf auch nicht als eine
zitiert werden.

## Die Trainingsschleife — und was man an ihr nicht kaputtmachen darf

⚠ **Kein Teacher Forcing, und das wird oft falsch erzählt.**
`train.py` rollt **einmal je Epoche je OP** die *eigene* Trajektorie unter
`torch.no_grad()` aus, gesät nur von der gemessenen Anfangsbedingung, und friert
sie ein. Erst darauf laufen `inner_steps` Ein-Schritt-Updates gegen die Labels.

Der Unterschied zwischen Training und Auswertung ist also **nicht** der
Eingangszustand — beide rollen frei. Er ist:

| | Training | Auswertung |
|---|---|---|
| Trajektorie | eingefroren, je Epoche erneuert | live |
| Labels | ja, als Ziel | nein |
| Gradient | nur **ein** Schritt (Historie detached) | keiner |

**Daraus folgt der ganze Hebel von Stufe 5** — und drei Tests halten es am
*Verhalten* fest statt an der Absicht: eine Störung der Trajektorie an einem
Zeitpunkt, der weder Anker noch Lag ist, darf den Verlust **nicht** ändern;
Anker und beide Lags müssen sehr wohl durchschlagen; und eine Störung der
**Labels** vor dem Ziel darf nichts tun — genau dort entstünde Teacher Forcing.

### Was mitläuft, weil Stille hier teuer wäre

* **`[SATURATED]`** — ein festgehaltener Rollout wird gezählt und gemeldet.
  Ohne den Zähler sähe ein weglaufendes Modell aus wie langsame Konvergenz.
* **Streuungsverhältnis in Ort und Zeit.** Beide Residuen verschwinden identisch
  auf einem flachen Feld. Ein fallender Physik-Verlust ist also nur so lange ein
  Beleg für Physik, wie diese Verhältnisse nahe 1 bleiben. Fällt es mit, hat der
  Optimierer die triviale Lösung gefunden — und keine Verlustkurve zeigte das.
* **Ein abgeschalteter Term läuft als `NaN` mit, nicht als `0.0`** — damit die
  Kurve eine Lücke zeigt statt einer flachen Linie, die es nie gab.
* **Labels nur bis `split_t`.** Sonst wird aus einer ausgehaltenen Zahl eine
  Trainingszahl. Ein Test prüft es.
* **`--seeds < 3` wird angemeckert.** Ein Seed ist keine Streuung.

## Die nächsten drei Schritte

1. **`balance_check.py`, Abschnitt 4 nachholen** — Minuten, nur numpy. Der Lauf
   vom 22.09. hat `Q_ht/tot` geklärt (0.700 … 0.772, **nicht** ≈ 1) und die
   Fluidbilanz bestätigt (1.030 … 1.062); `U(V̇)` fehlt noch, weil Abschnitt 4
   an den Zeitachsen abgestürzt ist. Werkzeug repariert, Lauf steht aus.
2. **Stufe 2, der Cache-Umbau** — danach ist der Wandterm kalibrierbar, die
   Physik-Latte messbar und `L_wall` anschließbar.
3. **Den Ladepfad in `train.py` anschließen** und Konfiguration **A** als ersten
   Lauf fahren, weil sie die Latte für B und C ist.

> Schritt 1 und 2 brauchen die Rechenmaschine mit `data_raw/` und `data_cache/`.
> Schritt 3 braucht den Cache, aber **nicht** die vier neuen Größen — er kann
> also parallel zu Schritt 2 laufen, solange man die Ergebnisse als das liest,
> was sie dann sind: adiabat.