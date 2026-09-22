# Nächste Sitzung — Stand 22.09.2026

> **Was das hier ist:** der Einstieg nach einer Pause. Eine Seite, die sagt *wo
> du stehst*, *wie es läuft* und *womit du anfängst* — ohne dass du vorher
> 170 KB `FAHRPLAN.md` liest.
>
> **Was es nicht ist:** ein Ersatz für die lebenden Dokumente. Die Wahrheit
> steht weiter in [`PINNmodulusTwo/FAHRPLAN.md`](PINNmodulusTwo/FAHRPLAN.md),
> [`GridCNN/FAHRPLAN.md`](GridCNN/FAHRPLAN.md) und
> [`GridCNN/BENCHMARK.md`](GridCNN/BENCHMARK.md). Dieses Dokument verweist
> dorthin und widerspricht ihnen an drei Stellen — die stehen unter §4 und sind
> als solche gekennzeichnet.

**In dieser Sitzung nachgeprüft** (nicht abgeschrieben, sondern ausgeführt):

| | |
|---|---|
| `pytest PINNmodulusTwo/tests` | **135 passed, 1 skipped, 1 xfailed** — 15.7 s |
| `pytest GridCNN/tests` | **107 passed** — 5.5 s |
| `benchmark.py --stage 0 2 --dry-run` | **12 × OK**, keine Abweichung zum Protokoll vom 14.09. |
| `main` gegen den Arbeitsbranch | **identisch** (`1dd100c`), nichts hängt offen |
| CI auf `main` | **grün**, Lauf #82 am 22.09. 10:34 UTC |

> Die Testzahlen in den Fahrplänen sind veraltet (dort stehen 91 bzw. „100" für
> GridCNN und 133 für PINNmodulusTwo). Kein Fehler, nur nicht nachgezogen.

---

## In einem Absatz

Du hast **zwei Modelle** auf demselben Datensatz: `PINNmodulusTwo` (das PINN,
läuft, misst, verallgemeinert nicht) und `GridCNN` (der Faltungs-Surrogat,
gebaut und getestet, hat **echte Daten noch nie gesehen**). Am PINN sind zwei
Achsen durch, beide `[NOT SEPARATED]` — und dabei ist die Ursache gefunden
worden: **O18**, die Leitungsgleichung steht in der falschen Form. Am GridCNN
hängt alles an **einer Sache**: der Cache trägt drei Größen nicht, die der
Wandterm braucht. Das ist seit dem 14.09. so, es ist ein halber Tag Arbeit, und
solange es so bleibt, ist Stufe 3, Stufe 4, `L_wall` und der Vergleich der
beiden Modelle blockiert.

---

## 1 · Wo du stehst

### ⚠ Zuerst: **zwei Leitern heißen „Stufe"**

Das ist die häufigste Art, sich hier zu verlaufen, und sie kostet jedes Mal
zehn Minuten:

* **`benchmark.py`-Stufen 0–3** — das Messprotokoll in `BENCHMARK.md`. Prüft
  Reshape, Stencil, Löser. Sagt nichts über die Architektur.
* **Fahrplan-Stufen 0–6** — die Projektleiter in `GridCNN/FAHRPLAN.md`. Rangtest,
  Bilanz, Cache, Löser, CNN, BPTT, Vergleich.

Die Nummern stimmen **nicht** überein. „Stufe 2" heißt einmal *Stencil-Ordnung*
und einmal *Cache-Umbau*.

### a) `benchmark.py` — das Protokoll

**Letzter echter Lauf: 14.09., 11:25 UTC.** Seither nichts.

| Stufe | | Stand |
|---|---|---|
| 0 `workflow` | 🟢 | 8 × ok — heute nachgeprüft, unverändert |
| 1 `grid` | ⏸ | **nie gelaufen** — braucht `data_cache` |
| 2 `stencil` | 🟢 | 4 × ok — heute nachgeprüft, unverändert |
| 3 `solver` | ⏸ | **nie gelaufen** — braucht `data_cache` |

> Der Eintrag vom 22.09. in `BENCHMARK.md` ist **kein** `benchmark.py`-Lauf,
> sondern von Hand am Cache gemessen. Stufe 1 und 3 haben bis heute **keine
> einzige Zahl** produziert.

### b) GridCNN-Fahrplan — die Projektleiter

| Stufe | | Stand |
|---|---|---|
| 0 Rangtest | ✅ | **rot** (4 Moden bei 99.9 %), am 15.09. bewusst überstimmt → 16 × 3 = 11 427 Parameter |
| 1 Bilanz | 🟡 | **hier hast du aufgehört.** Zweiter Lauf am 22.09.: Abschnitt 1–3 durch, **Abschnitt 4 abgestürzt** |
| 2 Cache-Umbau | ⬜ | **nicht angefangen — der Engpass** |
| 3 Löser | ✅⏸ | gebaut, läuft **adiabat**, nie gemessen |
| 4 CNN | ✅⏸ | gebaut, Ablationsarm D am 22.09. dazu, **Ladepfad fehlt**, nie gemessen |
| 5 BPTT | ⬜ | |
| 6 Vergleich | ⬜ | ← *das ist am Ende die Aussage der Arbeit* |

**Was Stufe 1 am 22.09. ergeben hat:**

| | | |
|---|---|---|
| Fluidbilanz | **1.030 … 1.062** | 🟢 `ghost_hi` steht |
| `Q_ht/tot` | **0.700 … 0.772** | 🔴 Hypothese 1a **widerlegt** — sättigt gegen ~0.8, nicht 1 |
| `Q_ht/jr1` | **2.286 … 2.566** | 🟡 **flussabhängig** → das Tor „Q halbieren oder A verdoppeln" ist falsch gestellt |
| `tot/jr1` | **2.99 … 3.44** | ⅓ bis 40 % der Erzeugung liegt außerhalb der Wickel → **O17 belegt** |
| `U(V̇)` | **abgestürzt** | ← **genau die Frage, für die der Lauf gemacht wurde** |

Der Absturz war das Werkzeug, nicht die Daten: `balance_check.py` las die
Zeitachse von zwei der fünf CSVs und legte den Rest stillschweigend darauf.
**Repariert und committet** (`c5b16b4`), Lauf steht aus.

### c) PINNmodulusTwo — die Achsen

| Achse | | Stand |
|---|---|---|
| 0 `w_phys` 0.1 / 0 | ✅ | 10.09. — `[NOT SEPARATED]`. **Nullmessung: 5.248 ± 0.518 °C** |
| 1 δ 1.0 / 0.4 / 0.2 | ✅ | 15.09. — `[NOT SEPARATED]`. δ = 0.2 → 4.868 ± 0.650. **O8 geschlossen, O18 gefunden** |
| 2 O16 Gehäusewand | ⬜ | Tor davor: GridCNN-Stufe 1 (`U(V̇)`) |
| 3 `w_phys`/`w_bc`-Gitter | ⬜ | erst nach 1 und 2 sinnvoll |
| 4 O17 Fixpunkt | ⬜ | **nicht angefangen, braucht keine GPU, 15 Checkpoints liegen da** |

**Die drei offenen Punkte, auf die es ankommt:**

* **O18** — `physics.py:183` rechnet `Fo : ∇∇T`, also λ **außerhalb** der
  Divergenz. `(∇λ)·(∇T)` fehlt. Null im homogenen Material, dominant an einer
  Grenzfläche — und dieses Gitter hat in x **nur** Grenzflächen (363 = 121
  Spalten × 3 Knoten = die 3 Materialien). Erklärt `L_phys ~ 1/δ²`, das
  Stagnieren von `L_phys`, `[FLAT]`, und warum δ nichts tat.
* **O17** — der freilaufende Rollout hat einen **OP-unabhängigen Fixpunkt** bei
  45–50 °C. OP06 (wahr 59.55) wird zu 43.6 … 48.9, in *allen neun* Läufen der
  Achse 1. Das Modell hat die gepoolte Endtemperatur des Trainingssatzes
  gelernt, nicht die des Betriebspunkts. **Kein Gewicht verschiebt einen
  Fixpunkt.**
* **O16** — die Gehäusewand hat gar keine Randbedingung. Dieselbe Krankheit wie
  O18, an der anderen Stelle.

---

## 2 · Wie es läuft — **6 von 10**

Eine Zahl allein wäre hier irreführend, deshalb aufgeteilt:

| | Note | warum |
|---|---|---|
| **Methodik** | **9** | Das ist der starke Teil, und zwar deutlich. Seed-Streuung *gemessen*, bevor verglichen wird. Ein zirkulärer Test gefunden und geschlossen. Widerlegte Hypothesen bleiben mit Begründung stehen. Tore, die den Plan ändern statt nur den Haken. Ein Nebenbefund gegen die eigene Doktrin dokumentiert (R8). Das ist sauberer als die meisten Masterarbeiten |
| **Werkzeug & Tests** | **8** | 242 Tests, Sekunden, ohne GPU und ohne Cache. Zusagen sind *falsifizierbar* statt behauptet (`dT/dx` exakt null, Modell rechnet bei Init bitgleich den Löser). Abzug: zwei Tests maßen fünf Tage lang Rundung statt Verhalten und niemand hat das rote `main` bemerkt |
| **PINN-Ergebnis** | **3** | Zwei Achsen, zweimal `[NOT SEPARATED]`. Der Physik-Term — das *Namensgebende* — trägt messbar nichts. O17 ist kein Tuning-Problem, sondern ein Konditionierungsfehler. val-MAE ~4.9 °C bei `T_sigma` = 9.6 °C |
| **GridCNN-Ergebnis** | **2** | Vollständig gebaut, vollständig getestet, hat **nie echte Daten gesehen**. Jede Zahl bisher ist adiabat oder synthetisch |
| **Vorankommen** | **4** | ← *siehe unten* |
| **Dokumentation** | **7** | Inhaltlich exzellent, aber 172 KB + 52 KB + 40 KB. Dass dieses Dokument nötig war, ist selbst der Befund |

### Die eine Sache, die du ändern solltest

**Deine Diagnoserate ist hoch, deine Reparaturrate ist niedrig.**

* **O18** ist am 15.09. gefunden. Heute ist der 22.09. Der Term ist nicht gebaut.
* **Stufe 2** — ein halber Tag — blockiert seit dem 14.09. vier andere Dinge.
* Der 22.09. hat **fünf Commits und ~300 Zeilen Dokumentation** produziert, alle
  gut, und die eine Sache, die alles freimacht, nicht.

Das ist kein Fleiß- und kein Einsichtsproblem. Es ist ein Struktureffekt: fast
alles Interessante braucht die Rechenmaschine, und die Sitzungen finden ohne sie
statt. Also findest du stattdessen Fehler in Dokumenten — was wertvoll ist, aber
den Engpass nicht anfasst.

> **Konkret:** die nächste Sitzung **an der Maschine** macht §3 „Sitzung A"
> komplett durch, in einem Zug, und schreibt erst danach etwas auf. Nicht
> umgekehrt.

---

## 3 · Wo du weitermachst

### 🔴 Sitzung A — an der Rechenmaschine. Das Einzige, was den Engpass anfasst.

Rechne mit **1–2 h**. Die drei Schritte gehören in **einen** Zug; einzeln
verpuffen sie.

**A1 — `balance_check.py`, Abschnitt 4 nachholen.** Minuten, nur numpy.

```bash
cd ~/llmtraining               # der Linux-Rechner, NICHT der /mnt/c/... Windows-Pfad
git checkout main && git pull  # holt den Zeitachsen-Fix c5b16b4
source modulus_env/bin/activate
python3 GridCNN/tools/balance_check.py 2>&1 | tee 11_bilanz.txt
```

> ⚠ Die Fahrpläne sind sich beim Interpreter uneins: `GridCNN/FAHRPLAN.md` sagt
> `python3`, `PINNmodulusTwo/FAHRPLAN.md` §11.8 sagt ausdrücklich `python`,
> **NICHT** `python3`. Einmal nachsehen, was `modulus_env` liefert.

Abschnitt 1–3 kommen unverändert wieder (nachgerechnet). **Neu ist nur
Abschnitt 4** — und der entscheidet das offene Tor:

* **`U` liegt über die drei Flusslevel auf einer Kurve** → der flussabhängige
  Teil von `Q_ht/jr1` ist erklärt, was an konstantem Faktor übrig bleibt, ist
  Konvention. `U(V̇)` wird eine feste Funktion mit **null freien Parametern**.
* **`U` liegt nicht auf einer Kurve** → die Bezugsfläche stimmt nicht, und
  Achse 2 (O16) darf noch nicht gebaut werden.

**A2 — Stufe 2, der Cache-Umbau.** Der eigentliche Engpass. Siehe §4 —
**es ist weniger Arbeit als dokumentiert.**

Drei Größen fehlen wirklich: `q_solid_to_fluid`, `fluid_out_temp`, `cp_fluid`.
`schema_version` hochsetzen in
[`legacy/battery_surrogate_agenticWorkflow/build.yaml`](legacy/battery_surrogate_agenticWorkflow/build.yaml)
(steht auf `2`) — das invalidiert `compute_cache_key` und erzwingt den Rebuild.
Dann **alle siebzehn** OPs neu bauen.

*Tor:* `profile_report`, `coverage_report` und `energy_balance_report` müssen
**exakt dieselben** Zahlen liefern wie vorher.

**A3 — Danach sofort, weil der Cache dann warm ist:**

```bash
python3 GridCNN/benchmark.py --cache data_cache     # Stufe 1 und 3, erstmals
```

Das sind die beiden Stufen, die seit dem 14.09. auf `⏸` stehen.

### 🟡 Parallel — ohne Maschine, ohne GPU: **Achse 4 / O17**

Das billigste offene Stück im ganzen Projekt, und es sitzt genau dort, wo der
Fehler nachweislich ist.

`evaluate.py` gibt es nicht (nur eine gleichnamige Datei unter `legacy/`, die
etwas anderes tut). **Fünfzehn auskonvergierte Checkpoints** liegen da — 6 aus
Achse 0, 9 aus Achse 1. Die Messung: **Teacher Forcing gegen freien Rollout auf
OP06.** Sie trennt *„kann den Einzelschritt nicht"* von *„akkumuliert über 7 000
Schritte"* und hängt an keinem Sweep.

### 🟢 Danach, und erst danach: **O18 einbauen**

`(∂_j λ_ij)(∂_i T)` in `physics.py`. λ ist stückweise konstant, `∇λ` also
analytisch bekannt — kein numerischer Gradient auf drei Knoten. Zusammen mit
**O16**, denn beide beschreiben denselben fehlenden Übergang.

> ⚠ **Nicht mit dem skalaren Faktor 167 rechnen.** Der stammt aus dem
> *synthetischen* Cache. Am echten Gitter ist der Sprung **anisotrop**:
> in **x** von 0.755 auf 193 (**256 ×**), in **z** von 22.4 auf 193 (**8.6 ×**).
> Ein skalar dimensionierter Term träfe weder x noch z.

Dann messen: gegen dieselbe Nullmessung **5.248 ± 0.518 °C**, 3 Seeds,
60 Epochen. **Erst dann** ist `w_phys`/`w_bc` (Achse 3) eine sinnvolle Achse.

---

## 4 · Drei Funde aus dieser Sitzung

Am Code gelesen, **nicht am Cache geprüft** (hier liegt keiner). Erste Handlung
an der Maschine: einmal in ein `.npz` sehen und bestätigen.

### F1 · `total_w` liegt **schon** im Cache — Stufe 2 ist dafür nicht nötig

`assemble.py:139` schreibt `q_source = heat_source[["jr1_w", "jr2_w", "total_w"]]`.
Der Cache trägt also **drei** Spalten; `PINNmodulusTwo/data.py:421` liest nur
`[:, 0]`.

> **Das widerspricht dem Fahrplan.** `GridCNN/FAHRPLAN.md` (Stufe 2 und 1d) sagt,
> O17 sei *„messbar erst, wenn `total_w` in Stufe 2 mit in den Cache geht"*.
> Es ist bereits drin — als `q_source[:, 2]`, in jedem bestehenden Bündel, ohne
> Rebuild. **O17 lässt sich am heutigen Cache beziffern.**

### F2 · `mdot` ebenfalls — `fluid_mass_flow` ist ein Config-Kanal

`CONFIG_ORDER[4]` in `data.py:139`, gelesen in `data.py:1137`. Von den fünf für
Stufe 2 genannten Größen fehlen damit real nur **drei**:
`q_solid_to_fluid`, `fluid_out_temp`, `cp_fluid`. (Einheiten prüfen: kg/s hier
gegen l/min in der `U(V̇)`-Rechnung.)

### F3 · Der synthetische Cache verfälscht genau die Größe, um die es bei O17 geht

`tools/make_synthetic_cache.py:206` setzt
`q_source = column_stack([jr1_w, jr2_w, jr1_w + jr2_w])` — die dritte Spalte ist
dort **die Summe der beiden Wickel**. Am echten Datensatz ist sie das
ausdrücklich *nicht*: gemessen `tot/jr1 = 2.99 … 3.44` bei `jr2/jr1 = 1.000`.
Ein Test, der auf der Fixture `tot/jr1 = 2.0` sieht, prüft die Aussage nicht,
die O17 macht.

> **Kleiner Nebenfund zur Stufe-2-Anleitung:** sie nennt
> `generate_cache.py` und `opbundle_contract.md`. Die 72 Zeilen in
> `PINNmodulusTwo/generate_cache.py` sind aber nur ein Wrapper — die Arbeit sitzt
> in `legacy/.../battery_surrogate/data/{assemble,cache,raw_readers}.py`, und
> `opbundle_contract.md` existiert nur unter `legacy/.../docs/`.
> Immerhin: `raw_readers.py:55` mappt `"Heat Source Monitor (W)" → total_w`
> bereits, `fluid_mass_flow` ebenso (Zeilen 114/131).

---

## 5 · Fallen, die schon einmal zugeschlagen haben

| | |
|---|---|
| **Siebzehn OPs, nicht sechzehn** | OP01–OP16 **plus OP19** liegen im Cache. Wer sechzehn neu baut, lässt OP19 auf altem `schema_version` zurück und merkt es erst in Stufe 6 |
| **`derive_layout` mit `bundle.xn`**, nicht `op.xn` | `op.xn` ist float32 und halbiert die Reserve der Äquidistanzprüfung (4e-7 gemessen gegen `rtol` 1e-6). `bundle.xn` (float64) × `L_ref` |
| **OP16 ist nie Stützstelle für `U(V̇)`** | V̇ = 90, und es ist ein **Test**-OP. Es ist die Gegenprobe, nie die Stütze |
| **`C_fluid` nicht raten** | Liegt nicht vor. Bei elf Trajektorien zählt jeder freie Parameter, den man nicht braucht. Der `capacity`-Modus wirft deshalb, statt zu raten |
| **Eine adiabate Zahl ist keine Physik-Latte** | Sie ist eine Ablation und darf nicht als Latte zitiert werden |
| **Keine Zahl ohne Streuung daneben** | Die Latte ist ~**1 °C** (0.518 ohne / 0.882 mit Physik, auf OP06 bis 1.63). Eine Achse, die weniger bewegt, ist mit drei Seeds nicht lesbar |
| **Kein Befund aus der letzten Epoche** | Median über die letzten k Epochen. Drei Epochen sind kein Trend |
| **Kein `torch.equal` zwischen zwei Batchgrößen** | Bitgleichheit ist durch nichts garantiert — torch wählt den Faltungsalgorithmus nach Batchgröße *und* Maschine. Hat `main` fünf Tage rot gehalten |
| **`OP14` startet bei 0 °C, absichtlich** | Nicht maskieren, nicht ersetzen, nicht entfernen (O10) |
| **`OP19` nie als Auswahlkriterium** | Dauerhafte Envelope-Grenze (O11) |

---

## 6 · Was dieses Dokument weiß und was nicht

**Geprüft, in dieser Sitzung ausgeführt:** beide Testsuiten, der
Benchmark-Trockenlauf, der Git-Zustand, der CI-Zustand, und die Code-Stellen
unter §4 (gelesen).

**Nicht geprüft, aus den Dokumenten übernommen:** alle Messwerte aus Läufen auf
der Rechenmaschine — Achse 0 und 1, die Bilanzzahlen vom 22.09., die
Materialverteilung, die Laufzeiten. Hier liegt kein `data_cache` und keine GPU.

**Nicht nachgezogen:** die Testzahlen in den drei Fahrplänen (dort 91 / „100" /
133, tatsächlich 107 / 135).
