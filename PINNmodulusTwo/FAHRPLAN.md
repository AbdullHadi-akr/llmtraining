# Fahrplan — ein Projekt: OP01–OP16 trainiert, OP19 als Messvergleich

**Diese Datei ist der Einstieg.** Alles andere ist Nachschlagewerk. Wenn du nur
eine Datei liest, dann diese.

> ### Lebendes Dokument
>
> Diese Datei wird nach **jedem** Ergebnis fortgeschrieben — sie ist kein Plan,
> den man einmal schreibt und dann abarbeitet. Die Regeln dafür:
>
> * **Haken setzen, sobald ein Schritt durch ist** (`- [ ]` → `- [x]`), und die
>   gemessene Zahl in die Tabelle „Stand" darunter eintragen. Ein Haken ohne
>   Zahl ist wertlos: „gelaufen" und „das Kriterium erfüllt" sind zwei Dinge.
> * **Gemessene Zahlen ersetzen Vermutungen im Text.** Wo hier „unbekannt" oder
>   „ungemessen" steht und eine Messung vorliegt, wird der Satz umgeschrieben,
>   nicht ergänzt.
> * **Ein rotes Tor ändert den Plan, nicht nur den Haken.** Wenn Schritt 5
>   `LOSES TO` sagt, ist die Reihenfolge darunter falsch und wird neu geschrieben
>   — nicht abgehakt und ignoriert.
> * **Nichts hier ist gemessen, solange kein Haken steht.** Jede Zahl in diesem
>   Repo stammt bis heute aus einem synthetischen Bündel.

---

# Offene Punkte für die nächste Session

Zuerst lesen. Was hier steht, ist das, was beim Abbruch der letzten Sitzung
offen war — nicht neu abzuleiten.

| # | offen | wer |
|---|---|---|
| **O1** | **Es liegen Schritt-6-Ergebnisse bis Epoche 30 vor, die noch niemand ausgewertet hat.** `artifacts/metrics.txt` und (neu) `artifacts/history.csv` einschicken → Stand-Tabelle füllen | du schickst, ich werte |
| **O2** | Schritt 5b ist nie gelaufen. Falls O1 zeigt, dass `[SATURATED]` bis Epoche 30 verschwunden ist, ist 5b **hinfällig** — dann direkt Schritt 6 auswerten | ich entscheide aus O1 |
| **O3** | §9a.1 OP15: `cell_current` fehlt im Bündel. `python3 data.py` erneut laufen lassen — der Bericht sagt seit 31.08., welche der zwei Ursachen es ist | du, 2 min |
| **O4** | §9a.2 OP12 (**Training**): Profil endet bei 1440 s, Trajektorie bis 1604 s | Rückfrage an die Simulationsseite |
| **O5** | **Tote Eingangskanäle.** `soc_start` ist über alle OPs konstant 10 % (`DEAD -> forced to 0`), und die Rate-Kanäle von `c_rate` und `fluid_mass_flow` sind im Training tot — werden aber auf OP15/OP16/OP19 lebendig. Das Modell soll dort einen Kanal deuten, den es nie gesehen hat | zu entscheiden, siehe §10 |
| **O6** | Nichts an **Gewichten** ist auf Basis der Messungen geändert worden — bewusst, siehe §10 | offen bis 5b/6 |

---

# JETZT: was du als Nächstes machst

Sechs Schritte, chronologisch. Schritt 1–5 brauchen **kein GPU** und dauern
zusammen unter einer Stunde. Erst Schritt 6 kostet etwas.

**Nach jedem Schritt steht, was dich stoppen muss.** Wenn ein Stopp-Kriterium
zutrifft: nicht weitermachen, sondern die genannte Datei schicken.

- [x] **1** Code holen — 31.08.
- [x] **2** Läuft der Code? (keine Daten nötig) — 31.08., grün
- [x] **3** Cache bauen, alle sechzehn — 31.08.
- [x] **4** Stimmen die Daten? — 31.08., ein MISMATCH auf OP15 (Test-OP, unkritisch)
- [~] **5** Die Latte — 31.08. gelaufen, aber **der Lauf war nicht aussagekraeftig** (§9.3)
- [ ] **5b** Kurzlauf bei der ECHTEN Konfiguration ← **neu, vor 6**
- [ ] **6** Erster ernsthafter Lauf (GPU)

## Stand

Wird beim Abhaken ausgefüllt. Leer = noch nicht gemessen.

| Schritt | Kriterium | gemessen | Datum |
|---|---|---|---|
| 2 | `selftest.py` | **all checks passed** | 31.08. |
| 2 | `pytest` | **112 passed, 1 skipped** (83 s) | 31.08. |
| 2 | `op_registry.py` | 11 train / 2 val / 3 test, keine Warnung | 31.08. |
| 3 | 16 OPs gebaut | ja (OP19 offen) | 31.08. |
| 4 | `MISMATCH`-Zeilen | **1 — OP15, `cell_current` fehlt** | 31.08. |
| 4 | `bc_pairs` > 0 | **242** — gemessen, kein Fallback | 31.08. |
| 4 | **`A` je Lag** | **90.8 / 22.7** (bei dt = 4 s) | 31.08. |
| 4 | `dTdt_scale` | **3.534** | 31.08. |
| 4 | `T_sigma` / `T_span_ref` | 9.616 C / 1604 s | 31.08. |
| 4 | `phys_scale` / `Qsrc_scale` | 3.535 / 0.0241 | 31.08. |
| 5 | OP06 | `LOSES TO` (12.96 vs 10.82 C) — **nicht aussagekraeftig, §9.3** | 31.08. |
| 5 | OP09 | `LOSES TO` (8.71 vs 7.78 C) — dito | 31.08. |
| 5 | `[SATURATED]` | **ja, beide Epochen: 99 % / 94 % einer Trajektorie** | 31.08. |
| 5 | `spread s/t` | **4.9 / 4.2** — Rollout streut 5x so weit wie die Labels | 31.08. |
| 5b | `[SATURATED]` bei subsample 2 | | |
| 5b | `A` bei subsample 2 | | |
| 6 | `[SATURATED]` letzte Epoche | | |
| 6 | MAE OP06 / OP09 | | |
| 6 | MAE OP13 / OP15 / OP16 | | |
| 6 | MAE OP19 (Messvergleich) | | |

Alles läuft aus dem Repo-Wurzelverzeichnis:

```bash
cd /mnt/c/Users/M0245635/batterysurrogatemodell
source modulus_env/bin/activate
```

---

### - [x] Schritt 1 — Code holen (1 min)

PR #20 ist gemergt, es reicht also `main`:

```bash
git checkout main
git pull
```

> `PINNmodulusTwoExtProfiles/` verschwindet dabei — das ist gewollt, der Ordner
> ist in `PINNmodulusTwo/` aufgegangen. Falls dort noch ein `data_cache/` liegt:
> **stehen lassen**, `data.py` sucht ihn weiterhin.

**Stopp wenn:** `git status` nach dem Pull nicht sauber ist.

---

### - [x] Schritt 2 — Läuft der Code überhaupt? (2 min, keine Daten nötig)

```bash
source modulus_env/bin/activate          # <- OHNE DAS schlaegt alles fehl
python3 PINNmodulusTwo/selftest.py
python3 -m pytest PINNmodulusTwo/tests -q
python3 PINNmodulusTwo/op_registry.py
```

**Gemessen am 31.08.:** `all checks passed`, **112 passed, 1 skipped** (83 s),
und die Tabelle mit 11 train / 2 val / 3 test OPs ohne Warnung. ✅

> **Die Aktivierungszeile ist der Stolperstein.** Ohne sie laeuft alles unter
> `/usr/bin/python3` und der Fehler lautet
> `ModuleNotFoundError: No module named 'pandas'` — vier Importe tief in
> `materials.py`, wo nichts kaputt ist. `op_registry.py` laeuft trotzdem durch,
> weil es reine Standardbibliothek ist, was den Eindruck verstaerkt, es fehle
> nur eine Bibliothek.
>
> Seit dem 31.08. faengt `env_check.py` das ab und sagt stattdessen, welcher
> Interpreter laeuft und dass das venv fehlt. **Nicht** `pip install pandas` ins
> System-Python — das macht den naechsten Fehler schwerer lesbar, nicht
> leichter.

**Stopp wenn:** irgendetwas davon rot ist. → Ausgabe schicken.

---

### - [ ] Schritt 3 — Cache bauen, alle sechzehn (10–30 min)

Der Cache muss neu, weil bisher nur OP01–OP07 gebraucht wurden:

```bash
python3 PINNmodulusTwo/generate_cache.py OP01 OP02 OP03 OP04 OP05 OP06 OP07 \
        OP08 OP09 OP10 OP11 OP12 OP13 OP14 OP15 OP16 2>&1 | tee 03_cache.txt
```

Und `OP19` — den gibt es, er ist der Messvergleich. Er gehört **nicht** ins
Training und wird separat gebaut:

```bash
python3 PINNmodulusTwo/generate_cache.py OP19 2>&1 | tee -a 03_cache.txt
```

`config.yaml` hat `measurement_ops: [OP19]`, er läuft ab dann in jedem
`train.py`-Lauf als Bericht mit. Fehlt das Bündel, gibt es eine `[SKIP]`-Zeile
und sonst nichts — ein Messvergleich darf einen Trainingslauf nie blockieren.

**Stopp wenn:** ein OP nicht baut. → `03_cache.txt` schicken.

---

### - [ ] Schritt 4 — Stimmen die Daten? (2 min) ← **das erste echte Tor**

```bash
python3 PINNmodulusTwo/data.py 2>&1 | tee 04_daten.txt
```

Drei Zeilen zählen, und zwar in dieser Reihenfolge:

| worauf schauen | gut | schlecht |
|---|---|---|
| `profile_report` | keine Zeile mit `MISMATCH` | jede `MISMATCH`-Zeile. Das Plansheet ist eine Abschrift — **glaub den Bündeln, nicht der Tabelle** |
| `bc_scale=… (from N x-neighbour pairs)` | `N > 0` | `[FALLBACK 1/L_ref]`. Dann ist `w_bc` bedeutungslos |
| `A = 1/(lag_n * rate_scale) per lag: …` | **notieren, egal welcher Wert** | — |

**Gemessen am 31.08. — und die Vorhersage hier war falsch.** Es stand:
„`T_sigma` wird breiter, `dTdt_scale` kleiner, `A` damit **größer** als die
119/30 aus OP01–OP05." Gemessen ist `A` **kleiner**:

| | OP01–OP05 (alt) | OP01–OP16 (gemessen) |
|---|---|---|
| `T_span_ref` | 1474 s | **1604 s** |
| `dTdt_scale` | 2.479 | **3.534** |
| `A` bei 5 s / 20 s | 119 / 30 | **90.8 / 22.7** |

`T_sigma` ist tatsächlich breiter geworden (9.6 statt ~4.2), aber `dTdt_scale`
ist trotzdem **gestiegen**, nicht gefallen: die Profil-OPs bewegen sich schneller
als die konstanten, und das schlägt die Verbreiterung. Weniger Verstärkung heißt
weniger Abbruchrisiko in Epoche 1 — die gute Richtung.

> ⚠️ **`A` hängt am `--subsample`.** `dTdt_scale` ist die RMS einer zentralen
> Differenz **auf dem subgesampelten Gitter**: ein grobes Gitter glättet die
> Ableitung, ein feines nicht. Die 90.8/22.7 sind bei `--subsample 40`
> (dt = 4 s) gemessen, das Training läuft aber mit `subsample_time: 2`
> (dt = 0.2 s). Die maßgebliche Zahl druckt `train.py` beim Start — die aus
> Schritt 4 ist die Größenordnung, nicht der Wert.

**Stopp wenn:** ein `MISMATCH` auf einem **Trainings- oder val-OP**, oder
`FALLBACK` bei `bc_scale`. Ein `MISMATCH` auf einem **Test-OP** stoppt nicht —
er entwertet dessen Bericht, nicht das Training. → `04_daten.txt` schicken.

---

### - [ ] Schritt 5 — Die Latte (5–10 min, CPU reicht) ← **das entscheidende Tor**

```bash
python3 PINNmodulusTwo/train.py --epochs 2 --subsample 40 --device cpu \
        2>&1 | tee 05_latte.txt
```

Die MAE des Modells ist hier **egal** — zwei Epochen lernen nichts. Es geht um
die Zeile unter jedem OP:

```
  OP06 [T1-interp  ] MAE=?? C  ...
     baseline: beats|LOSES TO the trivial predictors
               (persistence=?? C, train-mean=?? C)
```

`persistence` = „das Feld ändert sich nie". `train-mean` = konstanter
Mittelwert. **Das ist die Zahl, auf die dieses Projekt seit Monaten wartet:**
schlägt das Modell „nichts tun"?

Nach zwei Epochen darf da noch `LOSES TO` stehen. Wichtig ist, dass die Zahlen
überhaupt da sind und der Lauf durchläuft.

> ⚠️ **Am 31.08. gelaufen — und die Zahlen taugen nicht als Latte.** `--subsample
> 40` bedeutet dt = 4 s und verletzt die CFL-Grenze (0.241 s) um Faktor 16.6,
> und `--delta-grid 0.2s` degeneriert unter einem 4-s-Gitter. Dieser Schritt
> zeigt also, **ob** der Lauf durchläuft, nicht **wie gut** er ist. Die echte
> Latte kommt aus 5b. Siehe §9.3.

**Stopp wenn:** `[ABORT]` — dann ist `A` zu groß, und Schritt 6 wäre
verschwendete Zeit. → `05_latte.txt` schicken, ich sage dir den Wert für
`--max-rate-amp`.

---

### - [ ] Schritt 5b — Kurzlauf bei der ECHTEN Konfiguration (~15 min je Lauf)

**Der Schritt, der entscheidet, ob Schritt 6 seine Stunden wert ist.**

Schritt 5 lief bei dt = 4 s; das Training läuft bei dt = 0.2 s. Zwei der drei
Probleme aus §9.3 verschwinden dadurch von selbst — die CFL-Verletzung und der
degenerierte Anker. Das dritte, der **weglaufende Rollout**, verschwindet nicht
automatisch. Genau das wird hier gemessen, und zwar mit **einer** Variablen
zwischen den beiden Läufen:

```bash
# 5b-1: die echte Konfiguration, nur kurz
python3 PINNmodulusTwo/train.py --epochs 3 2>&1 | tee 5b1_echt.txt

# 5b-2: dasselbe OHNE Physik- und BC-Term
python3 PINNmodulusTwo/train.py --epochs 3 --w-phys 0 --w-bc 0 \
        2>&1 | tee 5b2_ohne_physik.txt
```

Der zweite Lauf ist keine Spielerei: `README.md` sagt, der Clamp sei erst mit
`w_phys > 0` tragend — *„the physics gradient walks the weights out of the stable
region faster"*. Bei dt = 4 s war der Physik-Gradient nachweislich Rauschen; ob
er es bei dt = 0.2 s immer noch ist, trennen diese zwei Läufe.

**Zu notieren, aus beiden Läufen:**

| Zeile | warum |
|---|---|
| `A = 1/(lag_n * rate_scale) per lag: …` | bei dt = 0.2 s, das ist der **maßgebliche** Wert. Bei dt = 4 s waren es 90.8 / 22.7 |
| `[CFL …]` | muss jetzt `CFL OK` sagen |
| `[SATURATED]` je Epoche | **die Zahl, um die es geht** |
| `spread s/t` | bei 4.9/4.2 in Schritt 5; nahe 1 wäre gesund |

**Die Entscheidung danach:**

| 5b-1 | 5b-2 | heißt | dann |
|---|---|---|---|
| kein `[SATURATED]` | — | dt war das Problem | **Schritt 6 starten** |
| saturiert | sauber | der Physik-Gradient treibt es | `--w-phys 0.01`, oder `phys_scale` prüfen — **nicht** 60 Epochen |
| saturiert | saturiert | die Rekurrenz selbst | `--max-rate-amp 50`, dann `--history-mode raw` — **nicht** 60 Epochen |

**Stopp wenn:** beide saturieren. → beide Dateien schicken.

---

### - [ ] Schritt 6 — Der erste ernsthafte Lauf (Stunden, jetzt GPU)

**Erst wenn 5b grün ist** — also `[SATURATED]` verschwunden. Sechzig Epochen auf
einem Rollout, der zu 99 % im Clamp hängt, ranken das Clamp-Verhalten und nicht
das Modell.

```bash
python3 PINNmodulusTwo/train.py --epochs 60 2>&1 | tee 06_lauf.txt
```

`--device` fragt jetzt nach und listet auf, was die Maschine hat:

```
Which device should this run use?
  [1] cpu      CPU  (32 threads visible)
  [2] cuda:0   NVIDIA …  24.0 GiB   <- default
Choice [1-2, Enter = cuda:0]:
```

Über `nohup` oder ohne Terminal fragt er nicht, sondern nimmt `auto` und sagt
das. Dauerhaft festlegen: `device: cuda` in `config.yaml`.

Vier Signale im Log, in dieser Rangfolge:

| Signal | heißt | Reaktion |
|---|---|---|
| `[ABORT]` | Loss nicht-endlich | zurück zu Schritt 5 |
| `[SATURATED]` in der **letzten** Epoche | Rollout weggelaufen und festgehalten — **keine Vorhersage** | mehr Epochen → `lr` runter → längere `--rate-lags` |
| `[FLAT]` | Feld konstant; ein fallendes `L_phys` ist dann die triviale Lösung, nicht Physik | `--w-phys` / `--w-bc` senken |
| `LOSES TO` auf OP06/OP09 | schlechter als „nichts tun" | **das** ist das Problem, nicht die Gewichte |

---

## Was du mir danach schickst

Vier Dateien, alle klein:

```
03_cache.txt      (nur falls Schritt 3 gehakt hat)
04_daten.txt      <- am wichtigsten: MISMATCH, bc_scale, A
05_latte.txt      <- die baseline-Zeilen
06_lauf.txt       + PINNmodulusTwo/artifacts/metrics.txt
```

Wenn du früher stoppen musst: die Datei des Schrittes, an dem es hakt, reicht.
Sag dazu, **bei welchem Schritt** du bist — dann weiß ich, wo wir stehen, ohne
zu raten.

Danach entscheidet sich, was zuerst gebaut wird: die Seed-Schleife (§3 Phase 4),
eine Achse, oder — falls `LOSES TO` bleibt — etwas ganz anderes als ein
Benchmark.

---

## 0. Was sich am 31.08. geändert hat

Zwei Dinge, beide Vereinfachungen.

**Die acht Benchmark-Skripte sind gelöscht** — `smallBench.py`,
`bench_common.py`, `benchmark_balance.py`, `benchmark_arch.py`,
`benchmark_wphys_wbc.py` und in der Erweiterung `smokeBench.py`,
`profileBench.py`, `bench_profiles.py`. Zusammen 4735 Zeilen. Der Grund ist
nicht, dass sie falsch waren, sondern dass **kein einziges ihrer Ergebnisse auf
echten Daten gemessen war**. Ein Sweep über Konfigurationen, die alle noch nie
einen trivialen Vorhersager geschlagen haben, ist eine Rangfolge zwischen
Verlierern.

**Und es gibt nur noch ein Projekt.** `PINNmodulusTwoExtProfiles/` ist in
`PINNmodulusTwo/` aufgegangen. Die Trennung „konstante Treiber hier, Profile
dort" war nie eine echte Grenze: die Profil-Pipeline ist eine **echte
Obermenge** — ein konstanter Treiber ist ein Profil, das sich nicht bewegt.
Trainiert wird ab jetzt auf dem ganzen Plansheet, OP01–OP16, konstante Treiber
und Profile gemeinsam. `--resample point --no-driver-history` stellt die alte
Vorverarbeitung exakt wieder her, falls ein Vergleich sie je braucht.

Was `train.py` dadurch selbst kann, ohne dass ein Benchmark existieren muss:

| vorher | jetzt |
|---|---|
| `smallBench.py` druckte „the bar to beat" | `train.py` druckt persistence + Trainings-Mittel neben **jeder** OP-Zeile |
| `bench_common` baute Val-/Test-OPs | `--val-ops` / `--test-ops`, gleiche Normierung (`data.build_op` re-fittet nichts) |
| `profileBench` berichtete je Tier | `op_metrics` + `op_registry.tier_of` in jeder Zeile |
| `smokeBench` prüfte Plansheet und Abdeckung | `profile_report()` und `coverage_report()` laufen in jedem Lauf mit |
| `smallBench` warnte vor synthetischen Daten | Banner beim Start (`data.cache_is_synthetic`) |
| nur `bench_common` konnte `torch.save` | `train.py` schreibt `artifacts/model.pt` |

---

## 1. Der Datensatz — und was OP17–OP19 wirklich sind

**Trainings-/Validierungs-/Testuniversum ist OP01–OP16.** Alle sechzehn sind
Ladevorgänge (CH), alle aus derselben Batemo+StarCCM+-Simulation. **OP19 kommt
als Messvergleich dazu** — nicht als siebzehnter Trainings-OP, sondern als
eigene Frage (siehe unten). Der Split
steht in `op_registry.py` und ist dort begründet:

| Rolle | OPs | Regel |
|---|---|---|
| `--ops` (Training) | OP01–05, 07, 08, 10, 11, 12, 14 | jeder Profil-**Typ**, den ein Selektions-OP braucht, kommt hier vor |
| `--val-ops` | OP06, OP09 | konstant + Profil, je einer. Darauf darf getunt werden |
| `--test-ops` | OP13, OP15, OP16 | Extrapolations-Tier. Einmal lesen, nie darauf auswählen |

**OP17–OP19 sind kein Teil davon.** Sie stehen im Plansheet unter einer eigenen
Überschrift — „Abgleich mit Minimodul-Test" — und vergleichen gegen **gemessene**
Minimodul-Daten statt gegen die Batemo/StarCCM+-Simulation. Jede Treiberspalte
liest dort `Test Data`, es gibt also keine Plansheet-Zeile zum Abschreiben wie
bei OP01–OP16. Was das Blatt nennt, ist die Art des Versuchs:

| | Art | Lade-/Entladerichtung | Besonderheit |
|---|---|---|---|
| **OP17** | `DCH, CC` | **Entladung**, 2C | die einzige Entladung überhaupt — OP01–OP16 sind alle CH |
| **OP18** | `Fast Charge Lotus` | Ladung | `V_max` 4.3 V statt 4.35 V |
| **OP19** | `Fahrzyklus TDD.3` | WLTP (synth.), gemischt | `V_max` 4.3 V |

> **OP19 existiert** und hat eine Zeile in
> `legacy/battery_surrogate_agenticWorkflow/op_matrix.yaml`, lässt sich also mit
> `generate_cache.py` bauen.
>
> **OP17 und OP18 sind schlicht noch nicht simuliert.** Deshalb haben sie weder
> eine Zeile noch einen Rohexport — es fehlt der Simulationslauf, nicht die
> Unterstützung. Sobald sie gerechnet sind, brauchen sie **keine Codeänderung**:
> nur ihre Id in `op_registry.MEASUREMENT_OPS_AVAILABLE`.
>
> „OP01 bis OP19" heißt heute also **siebzehn** verfügbare Betriebspunkte.

Sie werden über `--measurement-ops` ausgerollt und berichtet, aber **nie**
trainiert und **nie** ausgewählt. In `config.yaml` steht bereits
`measurement_ops: [OP19]`, er läuft also in jedem Lauf mit, sobald `OP19.npz`
gebaut ist. Fehlt das Bündel, gibt es eine `[SKIP]`-Zeile und der Lauf geht
normal weiter — anders als bei `ops`/`val_ops`/`test_ops`, die hart
fehlschlagen. Ein Bericht darf ein Training nicht blockieren.

Die Zahl ist anders zu lesen als jede andere in diesem Projekt: sie mischt
Modellfehler, Messfehler und die Lücke zwischen Simulation und Prüfstand, und
nichts trennt die drei. Und OP17 wie OP19 sind härter als jeder Test-OP, aus
einem Grund, den kein Coverage-Report formuliert: **das Modell hat Entladung nie
gesehen**, und einen Fahrzyklus auch nicht. Dass sie zunächst gegen die trivialen
Vorhersager verlieren, ist eine Aussage über den Trainings-Envelope — kein
Fehler.

Der Datenpfad ist davon unabhängig: `build_op` liest jedes Bündel, das da ist,
und misst am Bündel selbst, welche Kanäle Profile sind. Sobald ein `OP17.npz`
auftaucht, braucht es **keine Codeänderung** — nur einen Eintrag in
`op_registry.MEASUREMENT_OPS_AVAILABLE`.

---

## 2. Die eine Regel

> **Nichts, was Stunden kostet, bevor das Billige gelaufen ist, das es entwerten
> könnte.**

Jede Phase hat ein **Tor**. Ist es rot, geht es nicht weiter — dann wird das Tor
repariert, nicht die nächste Phase gestartet.

---

## 3. Die Phasen

### Phase 0 — Rauchtest ohne Daten (Minuten, kein GPU)

Prüft die Mathematik, nicht das Ergebnis. Läuft auf einem frischen Checkout.

```bash
python3 PINNmodulusTwo/selftest.py            # Loss-Balance, Residuen-Skalierung
python3 -m pytest PINNmodulusTwo/tests -q     # Rollout, History-Fastpath, Checkpoint
python3 PINNmodulusTwo/op_registry.py         # der Split, ohne Daten
python3 PINNmodulusTwo/tools/rollout_divergence.py
```

**Tor G0:** alles grün.

Ohne `data_cache/` geht auch ein kompletter Trainingslauf, gegen ein
synthetisches Bündel:

```bash
python3 PINNmodulusTwo/tools/make_synthetic_cache.py
python3 PINNmodulusTwo/train.py --epochs 3 --subsample 40 \
        --ops OP01 OP02 --val-ops OP06 --test-ops
```

`train.py` druckt dabei ein Banner. Die Zahlen taugen zum Vergleich zweier
Läufe und zu sonst nichts — **nie als Ergebnis zitieren**.

### Phase 1 — Daten prüfen (Minuten, echte Daten, kein Training)

```bash
python3 PINNmodulusTwo/generate_cache.py OP01 OP02 OP03 OP04 OP05 OP06 OP07 \
        OP08 OP09 OP10 OP11 OP12 OP13 OP14 OP15 OP16
python3 PINNmodulusTwo/data.py            # Konstanten + profile_report + coverage
python3 PINNmodulusTwo/tools/data_probe.py
python3 PINNmodulusTwo/tools/interface_probe.py
```

**Tor G1:**

| Prüfung | Muss |
|---|---|
| `profile_report` | keine `MISMATCH`-Zeile. Das Plansheet ist eine Abschrift und kann falsch sein — glaub den Bündeln, nicht der Tabelle |
| `A` für `[5, 20]` | **notieren.** Das Pooling über OP01–OP16 verbreitert `T_sigma` und verkleinert damit `dTdt_scale`, also ist `A` hier **größer** als die 119/30 aus dem alten OP01–OP05-Projekt. Wie viel größer, ist ungemessen |
| `bc_scale` | aus x-Nachbarpaaren gemessen, **nicht** `[FALLBACK 1/L_ref]` |
| SNR | > 100, sonst misst der kurze Rate-Kanal Rauschen |
| Grenzflächenanteil | notieren — entscheidet über `ARCHITECTURE.md` 4.1 Option A vs. B |

### Phase 2 — Der Maßstab (Minuten) ← **hier fehlt bisher alles**

Ein kurzer Lauf, allein wegen der Latte. Die MAE des Modells ist hier egal:

```bash
python3 PINNmodulusTwo/train.py --epochs 2 --subsample 40 | tee latte.txt
```

Interessant ist nur, was neben jeder OP-Zeile steht:

```
  OP06 [T1-interp  ] MAE=?? C  RMSE=?? C  ...
     baseline: beats|LOSES TO the trivial predictors
               (persistence=?? C, train-mean=?? C)
```

`persistence` ist „das Feld ändert sich nie", `train-mean` der konstante
Mittelwert der Trainings-Labels. Beide werden auf **genau dem OP** gerechnet, um
das es geht — nie zitiert, weil die Beträge zwischen OP-Sätzen nicht übertragbar
sind.

**Tor G2:** die Latte steht als Zahl fest, auf den **echten** OP06 und OP09.
Alles ab hier wird gegen diese Zahl gelesen.

### Phase 3 — Der erste ernsthafte Lauf

```bash
python3 PINNmodulusTwo/train.py --epochs 60 --device cuda | tee lauf1.txt
```

**Tor G3**, in dieser Reihenfolge zu prüfen:

| Signal im Log | Bedeutung | Reaktion |
|---|---|---|
| `[ABORT]` | Loss nicht-endlich | `--max-rate-amp 50`, dann `--history-mode raw`. Die A-Zeile aus G1 ansehen |
| `[SATURATED]` in der letzten Epoche | der Rollout ist weggelaufen und wurde festgehalten — **keine Vorhersage** | mehr Epochen → `lr` runter → längere `--rate-lags` |
| `[FLAT]` | `spread_space`/`spread_time` unter 0.2: das Feld ist konstant, und ein konstantes Feld erfüllt Residuum und Neumann-BC exakt. Ein fallendes `L_phys` ist dann die triviale Lösung | `--w-phys` / `--w-bc` senken |
| `[DIVERGED]` | der Eval-Rollout ist nicht-endlich (bewusst ungeclampt) | wie `[SATURATED]`, nur schlimmer |
| `LOSES TO` auf einem val-OP | das Modell ist schlechter als „nichts tun" | **das** ist das Problem, nicht die Gewichte |

Dazu der Drift-Test — `pred_OP13.npz` schreibt der Lauf selbst:

```bash
python3 -c "
import numpy as np
d = np.load('PINNmodulusTwo/artifacts/pred_OP13.npz')
e = np.abs(d['T_pred'] - d['T_true']).mean(axis=1); n = len(e)
print('Wachstum', e[-(n//5):].mean() / e[1:n//5].mean())
"
```

Wachstum > 3 → Drift dominiert; das ist dann das Thema, nicht die Gewichte.

### Phase 4 — Erst jetzt wieder messen, und zwar neu gebaut

Reihenfolge des Neuaufbaus, **eine Sache pro Schritt** — das war der Fehler beim
letzten Mal:

1. **Zwei Läufe von Hand vergleichen.** Solange „ist A besser als B?" mit zwei
   `[val ]`-Zeilen beantwortbar ist, braucht es keine Maschinerie.
2. **Seeds.** Der erste echte Bedarf: eine MAE-Differenz ist wertlos ohne die
   Streuung über Seeds daneben. Eine Schleife über `--seed`, Mittelwert und Std.
   Das ist die einzige Ergänzung, die den bisherigen Ergebnissen wirklich
   gefehlt hat.
3. **Eine Achse.** Liste von `fit()`-Overrides, eine CSV-Zeile je Punkt. Kein
   Plot, kein Resume, kein Checkpoint-Merge.
4. **Plots und Resume ganz zuletzt**, nur für die Achse, die wirklich Stunden
   läuft.

Welche Achse zuerst: **Balance vor Gewichten vor Architektur.** `w_phys`
multipliziert `L_phys/EMA(L_phys)`, also eine selbstnormierte Größe — solange
der Physik-Term kollabiert, misst jeder Gewichts-Sweep den Kollaps.

Was aus dem alten Code übernommen wird, ist die **Bewertungslogik**, nicht die
Infrastruktur:

* Auswahl auf `--val-ops` als **Mittel über die Menge**, nie über einen OP.
  Konstant-genau bleiben und einem bewegten Treiber folgen sind zwei Ziele, die
  gegeneinander laufen; ein OP misst nur eines davon. Jede Einzel-MAE
  mitschleppen, sonst gewinnt eine Konfiguration den Mittelwert, indem sie eines
  der beiden ruiniert.
* **Nach Tier getrennt berichten.** Eine gemittelte Test-MAE über OP13, OP15 und
  OP16 mischt C-Raten-Extrapolation, einen ungesehenen Profiltyp und dreifachen
  Volumenstrom. Der Mittelwert dreier verschiedener Fragen beantwortet keine.
* **Kriterium ist MAE, nie `L_data`.** Die beiden ranken Konfigurationen
  nachweislich unterschiedlich.
* **Seed-Rausch-Urteil.** Ist die Spanne zwischen Konfigurationen kleiner als
  die zwischen Seeds einer Konfiguration, ist es keine Rangfolge.

Was **nicht** wiederkommt: dass jedes Skript seine eigene Kopie der Defaults
mitbringt. `config.yaml` ist die einzige Quelle.

---

## 4. Was du lokal machen musst

```bash
cd /mnt/c/Users/M0245635/batterysurrogatemodell
source modulus_env/bin/activate

# Phase 0 -- Minuten, keine Daten
python3 PINNmodulusTwo/selftest.py
python3 -m pytest PINNmodulusTwo/tests -q
python3 PINNmodulusTwo/op_registry.py

# Phase 1 -- Minuten. Der Cache braucht jetzt ALLE sechzehn.
python3 PINNmodulusTwo/generate_cache.py OP01 OP02 OP03 OP04 OP05 OP06 OP07 \
        OP08 OP09 OP10 OP11 OP12 OP13 OP14 OP15 OP16
python3 PINNmodulusTwo/data.py                | tee daten.txt
python3 PINNmodulusTwo/tools/interface_probe.py | tee interface.txt

# Phase 2 -- Minuten. Nur die baseline-Zeilen zaehlen.
python3 PINNmodulusTwo/train.py --epochs 2 --subsample 40 | tee latte.txt

# Phase 3 -- der erste ernsthafte Lauf
python3 PINNmodulusTwo/train.py --epochs 60   | tee lauf1.txt
```

**Was du mir danach schicken kannst,** damit ich weiterrechne statt zu raten:
`daten.txt`, `latte.txt`, `interface.txt`, `lauf1.txt` und
`artifacts/metrics.txt`. Fünf kleine Textdateien.

Erst wenn G2 und G3 grün sind, lohnt sich der GPU-Server — vorher kostet er nur
Geld.

---

## 5. Die Dateien

Nach dem Aufräumen: **10 Python-Dateien, 4 Dokumente.**

| Datei | Rolle |
|---|---|
| `model.py` | `LearnableSwish`, `ModulusMLP`, `RecurrentField`, `rollout` |
| `physics.py` | anisotropes Wärme-Residuum + Neumann-BC |
| `data.py` | Laden, Normierung, Profil-Resampling, Treiber-Rate-Kanäle, Reports |
| `materials.py` | die Material-CSVs |
| `op_registry.py` | Plansheet OP01–OP16, Tiers, der Split |
| `op_metrics.py` | MAE/RMSE/peak/transient/quiescent/late je OP |
| `train.py` | Trainingsschleife + Auswertung + Checkpoint |
| `device_utils.py` | Device, Seed, TF32 |
| `generate_cache.py` | rohe CSVs → `OP*.npz` |
| `selftest.py` | Arithmetik-Checks, Sekunden |
| `tests/` | Rollout-Stabilität, History-Fastpath, Buchhaltung |
| `tools/` | Sonden: Daten, Grenzflächen, Rollout-Divergenz, synthetischer Cache |
| **`FAHRPLAN.md`** | **Einstieg. Hier anfangen.** |
| `README.md` | Nachschlagewerk: Dateien, Flags, warum die Rekurrenz so aussieht |
| `ARCHITECTURE.md` | wie es intern läuft, offener Befund 4.1 |
| `README_GPU_SERVER.md` | nur aufschlagen, wenn der Server dran ist |

---

## 6. Was nicht zu tun ist

| Nicht | Warum |
|---|---|
| Die Benchmarks aus der Historie zurückholen | Sie zu haben war nie das Problem — sie ohne Maßstab zu fahren war es. Erst G2 |
| Auf OP17/OP18 warten | Noch nicht simuliert. Sie blockieren nichts — OP01–OP16 sind vollständig |
| OP19 als Test-OP zählen | Fahrzyklus, gemischt geladen/entladen, gemessen statt simuliert. Verliert anfangs zu Recht |
| `w_phys` auf 1.0 / 10.0 erhöhen | `L_phys_bal = L_phys/EMA(L_phys)` ist selbstnormiert, `w_phys` kommt darin nicht vor |
| Zahlen aus dem alten OP01–OP05-Projekt übernehmen | Andere Normierung, anderes `A`, andere `phys_scale`. Nichts überträgt sich als Betrag |
| Am Physik-Term schrauben, bevor G3 grün ist | Sonst zweite Variable im Vergleich |
| `--holdout-tail` reflexartig einschalten | Bei den CC-CV-OPs **ist** das späte Fenster der CV-Auslauf; es abzuschneiden nimmt den schwersten Teil aus dem Training |

---

## 9. Offene Befunde (31.08.)

### 9.3 Schritt 5 hat die Frage NICHT beantwortet — und warum

Der Lauf sagt auf beiden val-OPs `LOSES TO`. **Diese Zahl zaehlt nicht**, aus
drei Gruenden, und der dritte ist der eigentliche Befund.

**(a) `--subsample 40` verletzt die CFL-Grenze um Faktor 16.6.**

```
[CFL WARN] Δt=4.000s, Δt_max≈0.241s -> POTENTIALLY UNSTABLE
```

Bei dt = 4 s ist die Zeitableitung im Physik-Residuum Unsinn, also ist auch ihr
Gradient Rauschen. Das Training laeuft mit `subsample_time: 2`, dt = 0.2 s, und
damit innerhalb der Grenze. **Der Schritt-5-Lauf trainiert also eine andere
Konfiguration als der, den er freigeben soll.** Das ist ein Fehler in diesem
Fahrplan gewesen, nicht im Code.

**(b) `--delta-grid 0.2s` ist kleiner als der Datenschritt 4 s.**

```
[WARN] the anchor cannot resolve finer than the grid and will effectively act as 4s
```

Der Anker der hybriden History degeneriert. Faellt bei subsample 2 ebenfalls weg.

**(c) Der Rollout laeuft weg — und das faellt nicht automatisch weg.**

```
[SATURATED] epoch 1: OP05 365/369 steps   (98.9 %)
[SATURATED] epoch 2: OP12 376/402 steps   (93.5 %)
```

Fast die ganze Trajektorie haengt im Clamp. `train.py` sagt dazu selbst: *„it is
not a prediction, and a run that only survives because of this is not trained."*
Der Befund ist nicht ein schlechter OP, sondern das Modell: die Saettigung
**wandert** von OP05 nach OP12.

Bestaetigt von `spread s/t = 4.92/4.19`: der Rollout streut fuenfmal so weit wie
die Labels. Das ist das Gegenteil des `[FLAT]`-Falls — kein kollabiertes, sondern
ein explodierendes Feld.

**Nebenbefund:** bei 2 Epochen und einem EMA-Horizont von 10 Epochen sind die
Loss-Divisoren praktisch die aus Epoche 1 eingefrorenen —
`L_data/L_data_bal = 29696`, gesetzt vom saturierten Rollout der ersten Epoche.
Die Balance hat in diesem Lauf nie gearbeitet. Auch das verschwindet erst bei
laengeren Laeufen.

**Was der Lauf trotzdem gezeigt hat, und das ist viel:** die Pipeline laeuft von
Ende zu Ende auf echten Daten durch — 11 Trainings-OPs, 2 val, 3 test, OP19 als
Messvergleich, alle Metriken, alle Baselines, Checkpoint geschrieben. Und
`L_data` faellt von 94 auf 16, das Modell lernt also durchaus etwas.

> **Deshalb steht jetzt ein Schritt 5b vor Schritt 6.** Sechzig Epochen auf einem
> Rollout, der zu 99 % im Clamp haengt, ranken das Clamp-Verhalten und nicht das
> Modell — das sind verlorene Stunden.

### 9.4 OP19: die Latte ist dort fast unschlagbar

`persistence = 1.375 C`. Die Trajektorie bewegt sich kaum, „das Feld aendert sich
nie" ist also schon fast richtig. Dazu kommt: `tn` laeuft bis 2.18 (3496 s gegen
`T_span_ref` 1604 s), `c_rate` geht auf **-3.42** (Entladung, nie trainiert), und
`cell_current` wird negativ. `LOSES TO` ist dort erwartbar und sagt bis auf
Weiteres nichts ueber das Modell.

---

## 9a. Offene Befunde aus Schritt 4 (31.08.)

Beide kommen aus den Daten, nicht aus dem Code, und beide sind **nicht** durch
einen Codefix zu erledigen.

### 9.1 OP15: `cell_current` fehlt im Bündel

```
OP15 [held out] detected=fluid_inlet_temp,fluid_mass_flow
                sheet=cell_current,fluid_inlet_temp,fluid_mass_flow   <-- MISMATCH
```

Das Plansheet nennt OP15 „CC mit Fluidtemperaturprofil und Volumenstromprofil
**und CC-CV**". Im Bündel variiert `cell_current` nicht — der CC-CV-Auslauf ist
nicht drin.

**Blockiert nichts.** OP15 ist ein reiner Berichts-OP (`test_ops`), kein
Trainings- und kein Auswahl-OP. Was verloren geht, ist die Aussagekraft **dieses
einen** Berichts: OP15 sollte den ungesehenen Volumenstrom-Profiltyp testen, und
das tut er weiterhin — nur eben ohne den CC-CV-Anteil, den das Blatt verspricht.

**Nächster Schritt:** `python3 PINNmodulusTwo/data.py` erneut laufen lassen. Seit
dem 31.08. druckt der Bericht bei einem MISMATCH zusätzlich, was die
Upstream-Assembly für dieses Bündel als Profil *markiert* hat, und das trennt die
beiden möglichen Ursachen:

* **markiert, aber konstant** → die Profildatei fehlte oder war leer, der Kanal
  ist still auf seinen Skalar zurückgefallen. Rohexport von OP15 prüfen, OP15 neu
  bauen.
* **nie markiert** → das Blatt stimmt für OP15 nicht, oder OP15 wurde ohne dieses
  Profil exportiert.

### 9.2 Profile enden vor der Trajektorie — auch auf einem **Trainings**-OP

```
OP12 [train   ] ! fluid_inlet_temp covers 0.0..1440.0 s but the OP runs 0.1..1604.1 s
OP15 [held out] ! fluid_inlet_temp und fluid_mass_flow, dasselbe
```

Die letzten ~164 s (rund **10 %**) werden mit dem letzten Profilwert flach
gehalten. Auf OP15 ist das ein Berichtsproblem; **auf OP12 ist es
Trainingsdaten**: das Modell lernt dort 10 % lang einen Treiber, der so nie
simuliert wurde, und die Temperatur, die es dazu sehen soll, gehört zu einem
Treiber, den es nicht sieht.

Zu klären ist, ob der Simulationslauf wirklich länger war als das Profil (dann
ist das Bündel richtig und die Flachhaltung die einzig mögliche Annahme), oder
ob der Profilexport abgeschnitten wurde (dann ist er nachzuliefern).

**Bis das geklärt ist:** kein Grund, Schritt 5 aufzuhalten. Aber wenn OP12 später
auffällig schlechter ist als die anderen Trainings-OPs, steht hier, warum.

---

## 10. „Wird daraus am Ende ein Benchmark, oder füttere ich dich nur?"

Berechtigte Frage. Ehrliche Antwort: **im Moment ist es das Zweite, und das ist
nur für die ersten Schritte richtig.**

### Warum es gerade so läuft

Die ersten fünf Schritte beantworten Fragen, die **einmalig** sind und für die
sich keine Maschinerie lohnt: läuft der Code, stimmen die Bündel, wie groß ist
`A`, läuft der Rollout weg. Jede davon wird genau einmal gestellt. Ein
Sweep-Framework dafür zu bauen wäre wieder der Fehler, für den die acht
Benchmark-Skripte gelöscht wurden.

### Woran es kippt — und das ist ein hartes Kriterium

**Sobald zwei Konfigurationen verglichen werden sollen**, hört Hand-Auswerten
auf zu funktionieren, und zwar aus einem Grund, der nichts mit Bequemlichkeit zu
tun hat: eine MAE-Differenz zwischen zwei Läufen ist **nicht lesbar**, solange
die Streuung über Seeds daneben fehlt. Zwei Läufe mit 8.7 und 8.1 sind kein
Ergebnis, wenn derselbe Lauf mit einem anderen Seed zwischen 7.9 und 9.2
schwankt.

Das ist der Moment, an dem gebaut wird — nicht früher, nicht später.

### Was gebaut wird, konkret

Drei Dateien, in dieser Reihenfolge, jede einzeln nutzbar:

**1. `sweep.py` — die Seed-Schleife.** ~80 Zeilen.

```
python3 sweep.py --seeds 0 1 2 --epochs 20
  -> artifacts/sweep.csv: eine Zeile je (Konfiguration, Seed)
  -> stdout: Mittelwert und Std je Konfiguration über die val-OPs
```

Ruft `train.fit()` in einer Schleife, sonst nichts. Kein Plot, kein Resume, kein
Checkpoint-Merge. **Das allein hätte allen bisherigen Ergebnissen dieses Projekts
gefehlt.**

**2. Eine Achse.** Eine Liste von `fit()`-Overrides, eine CSV-Zeile je Punkt.
Erste Achse ist die Loss-Balance, nicht die Gewichte — solange der Physik-Term
kollabiert oder explodiert, misst ein Gewichts-Sweep das und nicht die Physik.

**3. Plots und Resume.** Zuletzt, und nur für die Achse, die wirklich Stunden
läuft.

### Die Bewertungsregeln stehen schon fest

Die müssen nicht erst gefunden werden — sie stammen aus dem gelöschten Code und
sind das Einzige daraus, was übernommen wird:

* Auswahl auf dem **Mittel über `--val-ops`**, nie über einen OP. Konstant-genau
  bleiben und einem bewegten Treiber folgen laufen gegeneinander; ein OP misst
  nur eines davon. Jede Einzel-MAE mitschleppen.
* **Nach Tier getrennt berichten.** Eine gemittelte Test-MAE über OP13/OP15/OP16
  mischt C-Raten-Extrapolation, ungesehenen Profiltyp und dreifachen Volumenstrom.
* **Kriterium ist MAE, nie `L_data`.** Die beiden ranken nachweislich verschieden.
* **Seed-Rausch-Urteil.** Spanne zwischen Konfigurationen < Spanne zwischen Seeds
  → keine Rangfolge.

### Was du dafür noch liefern musst: **einmal Schritt 5b oder 6 auswerten.**

Danach ist die Reihenfolge festgelegt und der Sweep wird gebaut. Das Füttern
endet an dieser Stelle, nicht irgendwann.

---

## 10a. Was aus den Messungen NICHT im Code gelandet ist — und warum

Ehrliche Bilanz zum 31.08. **An Gewichten, Vorverarbeitungs-Konstanten oder der
Loss-Balance ist nichts auf Basis der Messungen geändert worden.**

Geändert wurde nur, was unabhängig von den Zahlen falsch war: der
`residual_output`-Default, vier tote `config.yaml`-Schlüssel, `bc_scale` und
`phys_scale` (beim Merge regressiert), der `tier_of`-Absturz, dazu Diagnostik
(`env_check`, die `A`-Zeile, die MISMATCH-Aufschlüsselung) und Persistenz
(`history.csv`, periodischer Checkpoint).

**Warum nicht mehr:** die einzige Messung, die es gibt, kommt aus einem Lauf mit
CFL-Verletzung um Faktor 16.6, degeneriertem Anker, zu 99 % saturiertem Rollout
und einer Loss-Balance, die bei 2 Epochen und 10-Epochen-Horizont nie gearbeitet
hat (§9.3). Ein Gewicht auf dieser Grundlage zu setzen wäre geraten und würde
danach wie gemessen aussehen — genau die Sorte Zahl, wegen der die alten
Benchmarks gelöscht wurden.

**Drei Dinge, die die Daten aber schon nahelegen**, festgehalten damit sie nicht
verlorengehen:

1. **`Qsrc_scale` ist ein einziger gepoolter Divisor über OPs, deren Qsrc-RMS um
   Faktor 2.3 auseinanderliegt** (OP05 0.0158 … OP11 0.0370). `w_phys` bedeutet
   damit auf OP05 etwas anderes als auf OP11. Ein per-OP-Divisor wäre denkbar —
   ändert aber die Gleichung pro OP und muss gemessen, nicht angenommen werden.
2. **Tote Kanäle (O5).** `soc_start` ist über alle sechzehn OPs konstant 10 %,
   trägt also null Information und kostet eine Eingangsdimension. Schlimmer: die
   Rate-Kanäle von `c_rate` und `fluid_mass_flow` sind im Training tot und auf
   OP15/OP16/OP19 lebendig — das Modell soll dort einen Kanal deuten, für den es
   nie ein Beispiel gesehen hat. Das ist **kein Hyperparameter, sondern eine
   Grenze des Trainings-Envelopes**, und der Coverage-Report sagt es bei jedem
   Lauf.
3. **`--batch-bc 128` gegen 121 BC-Punkte.** Es gibt nur 121 Punkte auf `x=0`,
   der BC-Gradient ist also rauschiger als das Gewicht suggeriert. Kleine Sache,
   aber sie gehört in den ersten Balance-Sweep.

---

## 7. Abbruchkriterien

Ehrlichkeitsklausel — wann der Plan selbst falsch ist:

* **G2 zeigt `LOSES TO` auf beiden val-OPs, auch nach Phase 3** → das Problem
  sind die Daten oder die Modellklasse, nicht die Hyperparameter. Zurück auf
  Phase 1.
* **`[SATURATED]` lässt sich nicht auf 0 bringen** → der Rollout ist instabil,
  nicht untertrainiert. Dann `ARCHITECTURE.md` 3.1, nicht mehr Epochen.
* **Die Extrapolations-OPs bleiben katastrophal, während val gut ist** → das ist
  kein Fehler, das ist die Aussage. Elf Trainings-OPs sind eine harte Grenze;
  dann ist die Frage, ob der Envelope erweitert werden muss, nicht welches
  Gewicht gewinnt.

---

**Zuletzt fortgeschrieben:** 2026-08-31, Sitzungsende. Offene Punkte stehen ganz
oben; §10 beantwortet, wann aus diesem Fahrplan ein Benchmark wird, und §10a,
was aus den Messungen bewusst NICHT in den Code gewandert ist.

**Stand nach Schritt 5.** Schritt 1–4 gruen,
Schritt 5 gelaufen aber **nicht aussagekraeftig** — und das hat den Plan
geaendert: es gibt jetzt einen **Schritt 5b** vor Schritt 6.

Die Pipeline laeuft von Ende zu Ende auf echten Daten durch, und `L_data` faellt
von 94 auf 16 — das Modell lernt. Aber der Rollout haengt zu 99 % im Clamp, und
Schritt 5 lief ausserdem bei dt = 4 s statt der 0.2 s, mit denen trainiert wird.
Beides in §9.3.

**Offene Befunde** — §9:
1. §9.3 Der Rollout laeuft weg (`[SATURATED]` in beiden Epochen). **Das ist die
   Frage, an der jetzt alles haengt.**
2. §9.4 OP19s Latte ist mit persistence = 1.375 C fast unschlagbar.
3. §9a.1 OP15: `cell_current` fehlt im Buendel, obwohl das Plansheet CC-CV nennt.
4. §9a.2 OP12 (**Training**) und OP15: das `fluid_inlet_temp`-Profil endet bei
   1440 s, die Trajektorie laeuft bis 1604 s; die letzten ~10 % sind flach.

**Ausgeführt:** Testsuite (110 grün) und Ende-zu-Ende-`train.py`-Läufe gegen ein
synthetisches Bündel — Banner, Training, val/test, `op_metrics`, Coverage-Report,
Baselines, Checkpoint-Roundtrip, und ein nicht gelisteter OP als
`--measurement-ops`.

**Nicht ausgeführt:** alles auf echten Daten. `data_cache/` und
`material_properties/` liegen nur auf der Arbeitsmaschine. Die Tabelle „Stand"
ganz oben ist deshalb leer — sie ist die einzige Stelle, an der eine gemessene
Zahl in dieser Datei steht.
