# Fahrplan — ein Projekt, OP01–OP16

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

# JETZT: was du als Nächstes machst

Sechs Schritte, chronologisch. Schritt 1–5 brauchen **kein GPU** und dauern
zusammen unter einer Stunde. Erst Schritt 6 kostet etwas.

**Nach jedem Schritt steht, was dich stoppen muss.** Wenn ein Stopp-Kriterium
zutrifft: nicht weitermachen, sondern die genannte Datei schicken.

- [ ] **1** Code holen
- [ ] **2** Läuft der Code? (keine Daten nötig)
- [ ] **3** Cache bauen, alle sechzehn
- [ ] **4** Stimmen die Daten? ← erstes echtes Tor
- [ ] **5** Die Latte ← das entscheidende Tor
- [ ] **6** Erster ernsthafter Lauf (GPU)

## Stand

Wird beim Abhaken ausgefüllt. Leer = noch nicht gemessen.

| Schritt | Kriterium | gemessen | Datum |
|---|---|---|---|
| 2 | `pytest` grün | | |
| 3 | 16 OPs gebaut (+ OP19?) | | |
| 4 | `MISMATCH`-Zeilen | | |
| 4 | `bc_pairs` > 0 | | |
| 4 | **`A` je Lag** | | |
| 4 | `dTdt_scale` | | |
| 5 | OP06 `beats` / `LOSES TO` | | |
| 5 | OP09 `beats` / `LOSES TO` | | |
| 6 | `[SATURATED]` letzte Epoche | | |
| 6 | MAE OP06 / OP09 | | |
| 6 | MAE OP13 / OP15 / OP16 | | |

Alles läuft aus dem Repo-Wurzelverzeichnis:

```bash
cd /mnt/c/Users/M0245635/batterysurrogatemodell
source modulus_env/bin/activate
```

---

### - [ ] Schritt 1 — Code holen (1 min)

```bash
git fetch origin
git checkout claude/remove-benchmarks-optimize-7d1q7k
git pull
```

> Der Branch von PR #20. `PINNmodulusTwoExtProfiles/` verschwindet dabei — das
> ist gewollt, der Ordner ist in `PINNmodulusTwo/` aufgegangen. Falls dort noch
> ein `data_cache/` liegt: **stehen lassen**, `data.py` sucht ihn weiterhin.

**Stopp wenn:** `git status` nach dem Pull nicht sauber ist.

---

### - [ ] Schritt 2 — Läuft der Code überhaupt? (2 min, keine Daten nötig)

```bash
python3 PINNmodulusTwo/selftest.py
python3 -m pytest PINNmodulusTwo/tests -q
python3 PINNmodulusTwo/op_registry.py
```

**Erwartung:** `all checks passed`, `110 passed`, und eine Tabelle mit
11 train / 2 val / 3 test OPs ohne Warnung darunter.

**Stopp wenn:** irgendetwas davon rot ist. → Ausgabe schicken.

---

### - [ ] Schritt 3 — Cache bauen, alle sechzehn (10–30 min)

Der Cache muss neu, weil bisher nur OP01–OP07 gebraucht wurden:

```bash
python3 PINNmodulusTwo/generate_cache.py OP01 OP02 OP03 OP04 OP05 OP06 OP07 \
        OP08 OP09 OP10 OP11 OP12 OP13 OP14 OP15 OP16 2>&1 | tee 03_cache.txt
```

Wenn `OP19` als Rohexport vorliegt, gleich mit — er wird später gebraucht,
gehört aber **nicht** ins Training:

```bash
python3 PINNmodulusTwo/generate_cache.py OP19 2>&1 | tee -a 03_cache.txt
```

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

Zu `A`: das alte Projekt maß 119/30 auf OP01–OP05. Über OP01–OP16 gepoolt wird
`T_sigma` breiter, `dTdt_scale` kleiner und `A` damit **größer**. Wie viel
größer, weiß niemand — das ist die Zahl, die ich als Nächstes brauche.

**Stopp wenn:** `MISMATCH` oder `FALLBACK`. → `04_daten.txt` schicken.

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

**Stopp wenn:** `[ABORT]` — dann ist `A` zu groß, und Schritt 6 wäre
verschwendete Zeit. → `05_latte.txt` schicken, ich sage dir den Wert für
`--max-rate-amp`.

---

### - [ ] Schritt 6 — Der erste ernsthafte Lauf (Stunden, jetzt GPU)

Erst wenn 4 und 5 grün sind.

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
Ladevorgänge (CH), alle aus derselben Batemo+StarCCM+-Simulation. Der Split
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
trainiert und **nie** ausgewählt. In `config.yaml` steht `measurement_ops: []`;
auf `[OP19]` setzen, sobald `OP19.npz` auf der Maschine liegt.

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

**Zuletzt fortgeschrieben:** 2026-08-31 — Ersteinrichtung, noch kein Schritt
abgehakt.

**Ausgeführt:** Testsuite (110 grün) und Ende-zu-Ende-`train.py`-Läufe gegen ein
synthetisches Bündel — Banner, Training, val/test, `op_metrics`, Coverage-Report,
Baselines, Checkpoint-Roundtrip, und ein nicht gelisteter OP als
`--measurement-ops`.

**Nicht ausgeführt:** alles auf echten Daten. `data_cache/` und
`material_properties/` liegen nur auf der Arbeitsmaschine. Die Tabelle „Stand"
ganz oben ist deshalb leer — sie ist die einzige Stelle, an der eine gemessene
Zahl in dieser Datei steht.
