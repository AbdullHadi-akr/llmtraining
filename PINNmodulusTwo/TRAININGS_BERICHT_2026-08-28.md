# Trainingsbericht PINNmodulusTwo — 2026-08-28

> **Update 31.08.2026 — die Benchmark-Skripte in diesem Dokument gibt es nicht
> mehr.** `smallBench.py`, `bench_common.py`, `benchmark_balance.py`,
> `benchmark_arch.py`, `benchmark_wphys_wbc.py` und in der Profil-Erweiterung
> `smokeBench.py`, `profileBench.py`, `bench_profiles.py` sind gelöscht; sie
> werden Schritt für Schritt neu aufgebaut. Jeder Befehl unten, der eines dieser
> Skripte aufruft, läuft ins Leere. Die Messungen und Befunde bleiben gültig —
> nur die Aufrufe nicht. Aktueller Einstieg:
> [`FAHRPLAN.md`](FAHRPLAN.md).

> **Korrekturstand 2026-08-28 (readme update 28.8).** Die Messwerte dieses
> Berichts sind unverändert. Mehrere Bewertungen darin waren falsch und sind
> korrigiert — betroffen waren der genannte `FAIL`-Grund (§4), der
> Baseline-Vergleich (§9), die `w_phys`/`L_phys_bal`-Diagnose (§6), die
> Overfitting-Aussage (§6) und die Nächsten Schritte (§7). Korrekturen sind als
> **[K]** markiert. `smallBench.py` wurde im selben Zug so erweitert, dass die
> Fehldeutungen bei künftigen Läufen nicht mehr möglich sind.

## Zusammenfassung

Erfolgreich neue NPZ-Daten aus CSV generiert und erstes vollständiges Training
auf echten Daten durchgeführt. Der Loss fällt über die Epochen, der Rollout
bleibt endlich.

**[K]** Beide Läufe sind mit `FAIL` beendet worden, und zwar an der
**Loss-Balance**, nicht an der Genauigkeit. Ob das Modell auf echten Daten
überhaupt besser ist als ein trivialer Vorhersager, ist mit diesem Lauf
**nicht beantwortet** — die dafür nötige Vergleichszahl wurde nie auf den echten
Daten gerechnet (§9). Das Modell ist damit weder als „knapp besser als Baseline"
noch als „schlechter" einzuordnen; es fehlt schlicht der Maßstab.

---

## 1. Datenvorbereitung

### Was wurde gemacht:
1. **Alte NPZ-Dateien gelöscht** — die vorhandenen waren veraltet/fehlerhaft
2. **17 neue NPZ-Dateien aus CSV-Rohdaten generiert**:
   - OP01–OP07 (Training + Test)
   - OP08–OP16, OP19 (zusätzliche Datenpunkte)
3. **Quelle**: `battery_surrogate_agenticWorkflow/data_raw/` → CSV-Bundles
4. **Ziel**: `PINNmodulusTwo/data_cache/` → komprimierte NumPy-Archive

### NPZ-Struktur (Beispiel OP01):
```
T shape:       (14450, 363)   # Temperaturfeld über Zeit × Gitterpunkte
xyz shape:     (363, 3)        # 3D Koordinaten der 363 Gitterpunkte
t_fast points: 14450           # Zeitschritte (dt=0.1s)
```

### Datenvalidierung (data_probe.py):
✅ **A-Wert**: 118.9 / 29.7 für rate_lags [5s, 20s] — stimmt exakt mit Übergabedokumentation
✅ **dTdt_scale**: 2.479 (0% Abweichung)
✅ **SNR**: >2000 für beide Lags → **echtes Signal**, kein Rauschen
✅ **Verdict**: Rate-Lags [5.0, 20.0] beibehalten

---

## 2. Modellarchitektur

### Netzwerk:
- **Typ**: Hybrid PINN (Physics-Informed Neural Network)
- **Architektur**: MLP mit Width=128, Depth=4
- **Parameter**: 52,485 trainierbare Gewichte
- **Input Features**:
  - 7 Konfigurationsparameter (c_rate, Temperaturen, Ströme, ...)
  - 3 statische Features (Material, Geometrie)
  - 1 Forcing-Term (Wärmequelle)
  
### Zeitintegration:
- **Methode**: BDF2 (Backward Differentiation Formula, 2. Ordnung)
- **History Mode**: Hybrid — nutzt Temperaturraten bei Δt=5s und Δt=20s
- **Amplifikation A**: 118.9× und 29.7× für die zwei Lag-Fenster
- **Zeitschritt**: Δt = 0.2s (subsample=2 von ursprünglich 0.1s)
- **CFL-Stabilität**: ✅ 0.2s < 0.241s (stabiler Bereich)

### Physik-Terms:
- **Diffusion**: Anisotrope Fourier-Tensor (λ variiert je Material)
- **Quelle**: Joule-Heizung in JR1-Region (121/363 Gitterpunkte)
- **Randbedingungen**: Neumann BC bei x=0
- **Loss-Balancing**: EMA (Exponential Moving Average), decay=0.9/epoch

---

## 3. Trainingskonfiguration

### Datensätze:
- **Training**: OP01, OP02, OP03, OP04, OP05
- **Test (held-out)**: OP07
- **Validation**: keine — **[K]** `smallBench.py` kennt keinen Validierungssplit.
  Einen `--val-op` (Default OP06) gibt es nur in `benchmark_arch.py:119`.

### Hyperparameter:
```yaml
epochs:          10
batch_data:      2048      # [K] dazu batch_phys: 256, batch_bc: 128
optimizer:       Adam
grad_clip:       1.0
learning_rate:   (adaptiv via loss balancing)
steps:           100 pro OP pro Epoche × 5 OPs = 500 steps/epoch
total_steps:     5000
```

### Loss-Komponenten:
- **L_data**: MSE zwischen Vorhersage und echter Temperatur
- **L_phys**: Residuum der PDE (Wärmeleitungsgleichung)
- **L_bc**: Randbedingungsverletzung
- **Gewichte**: w_phys = [0.0, 0.1] im Sweep getestet

---

## 4. Trainingsergebnisse

### 🔬 **Experiment 1: w_phys = 0.0 (rein datengetrieben)**

| Metrik | Epoche 1 | Epoche 5 | Epoche 10 |
|--------|----------|----------|-----------|
| L_data | 1.64e+04 | 9.87e-01 | 7.85e-01 |
| SATURATED | OP01, OP03 (>6900 steps) | — | OP03 (342 steps) |

**Finale Metriken:**
- Train MAE: **8.87°C**
- Test MAE: **13.48°C**
- Konvergenz: ✅ Ja
- Status: ❌ FAIL — **[K]** *nicht* wegen der MAE. `smallBench.py` prüft
  `test_mae < 20.0`; 13.48 °C besteht das. Der Physik-Term ist bei `w_phys=0`
  übersprungen (daher `nan`), also kann nur **`L_bc_bal`** die Balance-Prüfung
  gerissen haben. Diese Zahl wurde im Lauf nicht protokolliert und ist aus
  `artifacts/metrics.txt` nachzutragen.

**Beobachtungen:**
- Epoche 1: Starke Saturation (Rollout-Guard greift), aber Loss bleibt endlich
- Ab Epoche 2: Saturation-Count **fällt** → das ist das richtige Vorzeichen
  (`README_LOKALER_LAUF.md:171`)
- Epoche 10: noch 342/7279 Steps saturiert bei OP03. **[K]** Das ist **kein** ✅.
  `train.py:674` schreibt zu genau diesem Zähler: *„the trajectory ran away and
  was held back — it is not a prediction, and a run that only survives because
  of this is not trained."* Nichtnull in der letzten Epoche heißt: der Rollout
  läuft auf einem **Trainings**-OP weiterhin weg und wird nur vom Guard
  gehalten.

---

### 🔬 **Experiment 2: w_phys = 0.1 (mit Physik-Term)**

| Metrik | Epoche 1 | Epoche 5 | Epoche 10 |
|--------|----------|----------|-----------|
| L_data | 1.81e+04 | 9.58e-01 | 8.55e-01 |
| L_phys_bal | 3.23e-02 | 3.52e-08 | 2.69e-06 |

**Finale Metriken:**
- Train MAE: **7.65°C** (besser als w_phys=0.0!)
- Test MAE: **12.02°C** (besser als w_phys=0.0!)
- Konvergenz: ✅ Ja
- L_phys_bal: ⚠️ 2.69e-06 (Prüfbereich ist `0.01 < L_phys_bal < 100`)
- Status: ❌ FAIL — **[K]** allein wegen der Loss-Balance. Die MAE-Prüfung
  (`< 20 °C`) ist mit 12.02 °C **bestanden**.

**Beobachtungen:**
- Physik-Term **hilft**: -1.22°C Train MAE, -1.46°C Test MAE
- **[K]** `L_phys_bal` ist eine **Zeitreihe**, keine Konstante: 3.23e-02 →
  3.52e-08 (Epoche 5) → 2.69e-06 (Epoche 10). Zwischen Epoche 5 und 10 steigt
  sie um zwei Größenordnungen. Der Einbruch passiert früh und wird teilweise
  zurückgenommen — das ist Information, die in einer Einzelzahl verlorengeht.

---

### 📊 **Zusammenfassung der Ergebnisse:**

```
  w_phys |     L_data | L_phys_bal |  Train MAE |   Test MAE |   Status
----------------------------------------------------------------------
   0.000 | 7.8472e-01 |        nan |      8.87°C |     13.48°C |     FAIL
   0.100 | 8.5549e-01 | 2.6899e-06 |      7.65°C |     12.02°C |     FAIL
```

**[K]** Der Tabelle fehlt die Spalte **`L_bc_bal`** — genau die Größe, die den
`w_phys=0.0`-Lauf hat scheitern lassen. `smallBench.py` gibt sie inzwischen mit
aus; für diesen Lauf ist sie aus `artifacts/metrics.txt` nachzutragen.

**Beste Konfiguration**: w_phys=0.1
- **Verbesserung**: 10.8% Test MAE gegenüber `w_phys=0.0`
- **[K]** Das ist ein Vergleich der beiden Läufe untereinander und sagt nichts
  darüber, ob einer von beiden brauchbar ist. Beide sind `FAIL`.

---

## 5. Wo sind die Ergebnisse?

### Artefakte:
```
PINNmodulusTwo/artifacts/smallBench_convergence.png    (Konvergenz-Plot)
PINNmodulusTwo/artifacts/smallBench_results.txt        (Zusammenfassung)
/tmp/smallbench_output.txt                             (Vollständiges Log)
```

### Daten:
```
PINNmodulusTwo/data_cache/OP*.npz                      (17 NPZ-Dateien)
data_cache/OP*.npz                                      (Top-level Backup)
```

### Modell-Checkpoints:
Keine automatischen Checkpoints erstellt (smallBench speichert nur finale Metriken).
Für Checkpoints müsste `train.py` mit `--save-checkpoint` laufen.

---

## 6. Diagnostik & Probleme

### ✅ Was funktioniert:
1. **Datenpipeline**: CSV → NPZ → Training klappt
2. **Datenvalidierung**: `A = 118.9 / 29.7` und `dTdt_scale = 2.479` decken sich
   exakt mit `UEBERGABE_2026-08-27.txt` (Z. 29, 121–122). Das ist Punkt 1 von
   Schritt 2 aus `README_LOKALER_LAUF.md` und ist bestanden.
3. **Konvergenz**: Loss fällt über Epochen
4. **Kein Abbruch mehr**: Epoche 1 läuft ohne NaN durch

**[K] Nicht in dieser Liste** (stand vorher hier):
- *„Stabilität: keine NaN-Abstürze"* — das ist wörtlich der Fall, vor dem
  `model.py:738` warnt: „a clamp in the tens never binds on a model that is
  working." Endlich zu bleiben ist Verdienst des Guards, nicht des Modells.
- *„Saturation: Count fällt (Modell lernt sich selbst zu regularisieren)"* — der
  fallende Count ist ein gutes Zeichen, aber der Guard regularisiert nicht, er
  hält einen weglaufenden Rollout fest. Solange der Count > 0 ist, ist die
  Trajektorie laut `train.py:674` keine Vorhersage.

### ⚠️ Was noch problematisch ist:

1. **Die Loss-Balance reißt** — das ist der einzige Grund für beide `FAIL`.
   Prüfbereich `0.01 < L_*_bal < 100`; erreicht wurden 2.69e-06 (`L_phys_bal`,
   `w_phys=0.1`) und ein nicht protokollierter `L_bc_bal` (`w_phys=0.0`).

2. **[K] Der Physik-Term kollabiert — er ist nicht „zu schwach gewichtet".**
   Die frühere Deutung („`w_phys=0.1` ist zu klein") kann nicht stimmen.
   `train.py:600`:

   ```python
   L_phys_bal = L_phys / balance.divisor("phys", float(L_phys.detach()))
   ```

   Der Divisor ist die laufende EMA von `L_phys` selbst
   (`_LossBalancer.divisor`, `train.py:276-285`). `L_phys_bal` ist also
   `L_phys / EMA(L_phys)` — ein **selbstnormierter Quotient, in dem `w_phys`
   nicht vorkommt**. `w_phys` skaliert erst danach den Gradientenbeitrag
   (`train.py:613`). Ein höheres `w_phys` kann `L_phys_bal` deshalb
   **prinzipiell nicht** Richtung O(1) bewegen.

   Was 2.69e-06 heißt: `L_phys` ist gegenüber seinem eigenen laufenden Mittel um
   rund sechs Größenordnungen eingebrochen — ein Physik-Term, der **trivial
   erfüllt** wird. `README_MODEL_CRITIQUE.md:194-196` führt genau diesen
   degenerierten Fall als bekanntes Risiko. Mehr `w_phys` verstärkt den Druck,
   der dorthin führt.

3. **[K] Der Train-Test-Gap ist Unteranpassung, nicht Overfitting.**
   Der Gap stimmt (12.02 − 7.65 = 4.37 °C), die Deutung nicht: **Train MAE
   = 7.65 °C ist selbst schlecht.** Ein Modell, das die eigenen Trainingsdaten
   nicht besser als auf 7,65 °C trifft, ist unterangepasst. Overfitting setzt
   voraus, dass die Trainingsleistung *gut* ist. Die Unterscheidung ist nicht
   akademisch — sie führt zu entgegengesetzten Maßnahmen (mehr Regularisierung
   vs. mehr Kapazität und Budget).

4. **Der Rollout läuft weiterhin weg** — 342 saturierte Schritte auf OP03 in
   der letzten Epoche, auf einem *Trainings*-OP.

5. **[K] Es gibt keinen gültigen Maßstab für die MAE.** Siehe §9.

### 🔍 Mögliche Ursachen:
- **Nur 10 Epochen**: Zu kurz für volle Konvergenz? Bei Train-MAE 7.65 °C ist
  Unteranpassung ein realistischer Kandidat.
- **A=118.9 zu hoch**: Amplifikation macht initiale Fehler riesig
- **Residuums-Skalierung**: wenn der Physik-Term kollabiert, ist der Hebel die
  Normierung des Residuums, nicht sein Gewicht (Punkt 2)

**[K] Gestrichen:**
- *„`w_phys=0.1` zu klein: Physik-Loss wird wegbalanciert"* — siehe Punkt 2,
  mechanisch ausgeschlossen.
- *„Nur 363 Gitterpunkte: reduziertes Grid (Original hatte 6358), wahrscheinlich
  um CPU-Training zu beschleunigen"* — **die Zahl 6358 kommt im gesamten
  Repository nur in diesem Bericht vor.** 363 ist die native Sensorzahl,
  durchgehend so bezeichnet: `README_GPU_SERVER.md` Z. 406 („363 Sensoren"),
  Z. 724 („einen Absolutfehler pro Sensor, also 363 Werte"), Z. 869. Es gibt
  kein reduziertes Gitter und nichts zum Hochskalieren.

---

## 7. Nächste Schritte (laut README_LOKALER_LAUF.md)

### ✅ Erledigt:
- [x] Schritt 0: Daten platziert und verifiziert
- [x] Schritt 1: Datensonde (data_probe.py) — A-Wert bestätigt
- [x] Schritt 1: Vortests (selftest, pytest, rollout_divergence)
- [x] Schritt 2: **ein** `smallBench.py`-Lauf (als `w_phys`-Sweep)

### 📋 Offen

**[K] Schritt 2 ist nicht abgeschlossen.** `README_MODEL_CRITIQUE.md:159-186`
definiert `smallBench.py` als **A/B mit zwei Läufen** — „**der wichtigste Lauf
im ganzen Dokument**":

```bash
python3 PINNmodulusTwo/smallBench.py                       # neu
python3 PINNmodulusTwo/smallBench.py \                     # alter Stand
    --inner-steps 1 --no-residual-output --learn-gains --loss-balance legacy
```

Verglichen wird die Test-MAE beider. Gelaufen ist stattdessen ein
`w_phys`-Sweep `[0.0, 0.1]` — ein anderer Vergleich, der die Frage von Schritt A
nicht beantwortet: ob die Umbauten (Trainingsbudget, Residual-Output,
Physik-Residuum) überhaupt etwas gebracht haben. Solange das offen ist, ist die
Entscheidungstabelle in `README_MODEL_CRITIQUE.md:180-186` nicht anwendbar.

- [ ] Triviale Vorhersager auf dem **echten** OP07 rechnen (§9) — `smallBench.py`
      gibt sie inzwischen automatisch mit aus
- [ ] `L_bc_bal` aus `artifacts/metrics.txt` nachtragen (§4)
- [ ] Schritt A wirklich fahren: Baseline-Lauf gegen Default
- [ ] Drift-Test auf `artifacts/pred_OP07.npz`
      (`README_MODEL_CRITIQUE.md:206-223`)
- [ ] MAE-Zahlen in `README_ERSTER_TEST.md` Kapitel 6 eintragen und die dortigen
      synthetischen Zahlen ersetzen
- [ ] **Schritt 3 (Sweeps)** — gesperrt. `README_LOKALER_LAUF.md:193`: „Erst
      wenn Schritt 2 sauber durchläuft." Schritt 2 endete zweimal mit `FAIL`.

### 🚀 Empfohlene Next Actions

**[K] In dieser Reihenfolge.** Punkt 1 ist billig und macht alle anderen Zahlen
erst lesbar.

1. **Triviale Vorhersager auf dem echten OP07** — Persistenz `T(t) = T(0)` und
   konstanter Mittelwert der Trainingslabels. Kein Training, Minuten. Ohne das
   bedeutet „12.02 °C" nichts (§9).
2. **`L_bc_bal` nachtragen** — die wahrscheinliche Fehlerursache des
   `w_phys=0.0`-Laufs.
3. **Schritt A fahren** (Baseline-Lauf, siehe oben).
4. **Drift-Test** auf `pred_OP07.npz`. Der entscheidet, ob ein Sweep überhaupt
   etwas misst — `README_MODEL_CRITIQUE.md:225`: „Ein Gewichte-Sweep bei starker
   Drift misst hauptsächlich, welches Gewicht die Drift zufällig am wenigsten
   verstärkt — das ist die teure Art, nichts zu lernen."
5. **Dann `benchmark_balance.py --part 1`.** `smallBench.py` gibt genau das
   selbst als nächsten Schritt aus, mit Begründung im Code:

   > `# NOT the 10x10 grid: that is 100 trainings (~6-8 days) and it would`
   > `# sweep weights before anything has established what a weight means`
   > `# here. The balancing benchmark is ~4 h and settles that first.`

   Das gescheiterte Kriterium **ist** die Balance — das ist das Benchmark dafür.
6. **Längeres Training + GPU** (50–100 Epochen; CPU macht 42–98 s/Epoche). Bei
   Train-MAE 7.65 °C ist Unteranpassung plausibel.

**[K] Gestrichen:**
- *„Höheres `w_phys`: 1.0 oder 10.0, damit Physik durchkommt"* — kann
  `L_phys_bal` prinzipiell nicht bewegen (§6 Punkt 2) und verstärkt die
  Degeneration.
- *„`benchmark_arch.py` / `benchmark_wphys_wbc.py` als nächster Schritt"* —
  genau das, wovon der Code oben abrät, und durch das Gate in
  `README_LOKALER_LAUF.md:193` gesperrt.
- *„Full Grid: 6358 statt 363 Punkte"* — die 6358 existieren nicht (§6).

---

## 8. Technische Details für Reproduktion

### Environment:
```bash
Python: 3.10 (modulus_env virtualenv)
Torch: 2.12.1+cpu
Device: CPU (WSL2 on Windows)
NumPy: für NPZ-I/O
Pandas: für CSV-Lesen
```

### Kommandos zum Reproduzieren:
```bash
# 1. NPZ-Daten generieren
modulus_env/bin/python3 PINNmodulusTwo/generate_cache.py OP01 OP02 OP03 OP04 OP05 OP06 OP07

# 2. Datensonde
modulus_env/bin/python3 PINNmodulusTwo/tools/data_probe.py

# 3. Vortests
modulus_env/bin/python3 PINNmodulusTwo/selftest.py
modulus_env/bin/python3 -m pytest PINNmodulusTwo/tests -q
modulus_env/bin/python3 PINNmodulusTwo/tools/rollout_divergence.py

# 4. Training
modulus_env/bin/python3 PINNmodulusTwo/smallBench.py
```

### Konfiguration (PINNmodulusTwo/config.yaml):
Nutzt Default-Werte:
- ops: [OP01, OP02, OP03, OP04, OP05]
- test_op: OP07
- subsample_time: 2
- rate_lags: [5.0, 20.0]

---

## 9. Vergleich mit Baseline

> **[K] Dieser Abschnitt war in seiner ursprünglichen Form vollständig ungültig.**
> Drei voneinander unabhängige Fehler, jeder für sich ausreichend.

### Was ursprünglich hier stand

- „Naiver Predictor (predict mean): MAE = 11.96 °C"
- „`w_phys=0.1`: 12.02 °C → **leicht besser als Baseline** ✅ (+0.5 %)"
- „stark overfitted auf die Trainingsdaten"

### Fehler 1 — falsches Etikett

`README_ERSTER_TEST.md:387-389` führt **zwei** triviale Vorhersager:

| Vorhersager | MAE test |
|---|---|
| „Temperatur ändert sich nie", `T(t) = T(0)` | **11.96 °C** |
| „konstanter Mittelwert der Trainingslabels" | **6.60 °C** |

11.96 ist die **Persistenz**, nicht der Mittelwert. Der Mittelwert-Vorhersager
liegt bei 6.60 °C. Die Doku verlangt ausdrücklich den Vergleich gegen „das
bessere der beiden trivialen Vorhersager" — also gegen 6.60, nicht 11.96.

(Auch `T_sigma = 8.66 °C` ist unbelegt: die Zahl kommt sonst nirgends im Repo
vor, `README_ERSTER_TEST.md:699` rechnet mit `T_sigma = 5 K`.)

### Fehler 2 — Vorzeichen

Selbst gegen die falsch gewählten 11.96 °C gilt **12.02 > 11.96**. Das ist
0,5 % **schlechter**, nicht besser. Das ✅ war eine Verwechslung der
Vergleichsrichtung.

### Fehler 3 — der Vergleich ist überhaupt nicht zulässig

`README_ERSTER_TEST.md` setzt über Kapitel 6 einen Kasten:

> **Alle Zahlen in diesem Kapitel stammen von einem synthetischen Bundle** […]
> Sie sind **keine** Vorhersage der MAE auf den echten OPs.

Kapitel 9.1 wiederholt es: „Die *Richtung* ist robust […]; die *Beträge* sind es
nicht." **Die Baselines 11.96 und 6.60 sind selbst synthetisch.** Eine auf
echten Daten gemessene MAE gegen sie zu halten, ist genau die Übertragung, die
die Doku verbietet.

`README_LOKALER_LAUF.md:183-188` warnt zusätzlich vor der naheliegenden
Fehllesung: „Die 11.96 °C aus den alten Dokumenten sind eine **Baseline** […],
keine frühere Messung. Es gibt keine Verbesserung ‚von 11 auf 0.5'."

### Was tatsächlich gilt

**Es gibt für diesen Lauf keinen gültigen Maßstab.** Ob das Modell auf echten
Daten einen trivialen Vorhersager schlägt, ist unbekannt — weil die trivialen
Vorhersager auf den echten OPs nie gerechnet wurden. Weder das ✅ noch das ❌
oben hatte eine Grundlage.

Das ist ab jetzt behoben: `smallBench.py` rechnet beide trivialen Vorhersager
auf demselben Test-OP mit (`_trivial_baselines`) und gibt sie unter der Test-MAE
sowie in `artifacts/smallBench_results.txt` aus. Ein Lauf liefert die
Vergleichszahl damit gratis mit — dieselben Daten, dieselbe Metrik, kein
Transfer aus einem synthetischen Bundle.

Für die Läufe dieses Berichts ist die Zahl nachzureichen (Nachrechnen genügt,
kein Training nötig).

---

## 10. Datengröße & Rechenaufwand

### NPZ-Dateien:
- OP01: ~3.8 MB (14450 timesteps × 363 points)
- OP19: ~8.1 MB (35000 timesteps)
- **Gesamt**: ~70 MB für alle 17 OPs

### Training (CPU):
- **Epoche 1**: ~98 s (71s Rollout + 27s Inner Loop)
- **Epoche 10**: ~42 s (32s Rollout + 9s Inner Loop)
- **Gesamt**: ~10 Epochen × 50s avg = **~8 Minuten** pro w_phys
- **Full Run** (2× w_phys): ~16 Minuten

### GPU-Abschätzung:
- CPU: 42-98 s/Epoche
- GPU (A100): Erwartung 5-10 s/Epoche → **10x schneller**
- Ermöglicht: 100 Epochen in ~10 Minuten

---

## Fazit

**Status**: Die Datenpipeline steht und ist verifiziert — `A = 118.9 / 29.7`
gegen die Übergabe bestätigt. Zwei Trainingsläufe auf echten Daten sind
durchgelaufen, ohne Abbruch. Das ist der Fortschritt des Tages.

**[K] Aber**: Beide Läufe sind `FAIL`, und zwar an der **Loss-Balance** — nicht
an der Genauigkeit. Der Physik-Term **kollabiert** (er ist nicht zu schwach
gewichtet), der Train-Test-Gap ist **Unteranpassung** (nicht Overfitting), und
der Rollout läuft auf einem Trainings-OP weiterhin weg. Ob das Modell überhaupt
besser ist als „nichts tun", ist **unbekannt** — der Maßstab fehlt (§9).

**Nächster Schritt**: die trivialen Vorhersager auf dem echten OP07 rechnen
(Minuten, kein Training), dann Schritt A als echtes A/B fahren, dann
`benchmark_balance.py --part 1`. **Nicht** die Sweeps und **nicht** `w_phys`
hochdrehen — Begründung in §6 und §7.

---

**Erstellt**: 2026-08-28  
**Korrigiert**: 2026-08-28 (readme update 28.8) — Messwerte unverändert,
Bewertungen korrigiert, siehe **[K]**-Markierungen  
**Training Device**: CPU (WSL2)  
**Datensatz**: 17 OPs aus CSV, 5 Training + 1 Test  
**Bester Run**: w_phys=0.1, Test MAE = 12.02 °C — beide Läufe `FAIL`
(Loss-Balance); ohne Maßstab aus §9 ist die Zahl nicht einzuordnen
