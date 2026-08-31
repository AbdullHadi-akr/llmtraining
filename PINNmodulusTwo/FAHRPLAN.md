# Fahrplan — ein Projekt, OP01–OP16, von vorne

**Diese Datei ist der Einstieg.** Alles andere ist Nachschlagewerk. Wenn du nur
eine Datei liest, dann diese.

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

**OP17–OP19 sind kein Teil davon und können es nicht sein.** Sie sind der
Minimodul-**Messvergleich**: gemessene Daten statt Simulation, teils Entladung
wo OP01–OP16 durchweg Ladung sind, Treiber aus Testdaten statt aus dem
Plansheet, und OP19 ist ein synthetischer Fahrzyklus. Dazu kommt:

> **OP17 und OP18 existieren in dieser Pipeline überhaupt nicht.**
> `legacy/battery_surrogate_agenticWorkflow/op_matrix.yaml` kennt OP01–OP16 und
> OP19 — sonst nichts. „OP01 bis OP19" heißt in der Praxis also **siebzehn**
> Betriebspunkte, nicht neunzehn.

OP19 ist trotzdem wertvoll, nur als andere Frage: *stimmt ein auf StarCCM+
trainiertes Modell mit einer echten Zelle überein?* Dafür gibt es
`--measurement-ops`. Diese OPs werden ausgerollt und berichtet, aber **nie**
trainiert und **nie** ausgewählt. In `config.yaml` steht `measurement_ops: []`,
weil das Bündel optional ist; auf `[OP19]` setzen, sobald `OP19.npz` auf der
Maschine liegt.

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
| OP17/OP18 suchen | Existieren in dieser Pipeline nicht. OP19 nur über `--measurement-ops`, nie im Training |
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

**Stand:** 2026-08-31
**Ausgeführt:** Testsuite (107 grün) und ein Ende-zu-Ende-`train.py`-Lauf gegen
ein synthetisches Bündel — Banner, Training, val/test, `op_metrics`,
Coverage-Report, Baselines, Checkpoint-Roundtrip. **Nicht ausgeführt:** alles auf
echten Daten. `data_cache/` und `material_properties/` liegen nur auf der
Arbeitsmaschine.
